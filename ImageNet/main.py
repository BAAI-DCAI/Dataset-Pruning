import os
import time
import random
import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from yacs.config import CfgNode
from models import build_model
from data.build import build_loader, RealLabelsImagenet
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, load_pretrained, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor


def main(config):
    data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    num_steps = config.DATA.TRAIN_SIZE // config.WORLD_SIZE // config.DATA.BATCH_SIZE_TRAIN
    logger.info(f'Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}')

    model = build_model(config)
    # logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f'number of GFLOPs: {flops / 1e9}')

    model.cuda()
    model_without_ddp = model

    optimizer = build_optimizer(config, model)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    if config.COMPILE_ENABLE:
        model = torch.compile(model)

    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, num_steps // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, num_steps)

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    real_max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT + '/ckpts')
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f'auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}')
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy, real_max_accuracy = load_checkpoint(config, model_without_ddp, optimizer,
                                                          lr_scheduler,
                                                          loss_scaler,
                                                          logger)
        if config.DATA.REAL_LABEL:
            acc1, acc5, real_acc1, real_acc5, loss = validate(config, data_loader_val, model)
            logger.info(f'Accuracy of the network on the {config.DATA.VAL_SIZE} test images: {acc1:.2f}%')
            logger.info(f'Accuracy of the network on the {config.DATA.VAL_SIZE} test images (ReaL): {real_acc1:.2f}%')
        else:
            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(f'Accuracy of the network on the {config.DATA.VAL_SIZE} test images: {acc1:.2f}%')
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        if config.DATA.REAL_LABEL:
            acc1, acc5, real_acc1, real_acc5, loss = validate(config, data_loader_val, model)
            logger.info(f'Accuracy of the network on the {config.DATA.VAL_SIZE} test images: {acc1:.2f}%')
            logger.info(f'Accuracy of the network on the {config.DATA.VAL_SIZE} test images (ReaL): {real_acc1:.2f}%')
        else:
            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(f'Accuracy of the network on the {config.DATA.VAL_SIZE} test images: {acc1:.2f}%')

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info('Start training')
    start_time = time.time()

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch,
                        mixup_fn,
                        lr_scheduler,
                        loss_scaler)

        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, real_max_accuracy, optimizer, lr_scheduler,
                            loss_scaler,
                            logger)

        if config.DATA.REAL_LABEL:
            acc1, acc5, real_acc1, real_acc5, loss = validate(config, data_loader_val, model)
            logger.info(f'Accuracy of the network on the {config.DATA.VAL_SIZE} test images: {acc1:.2f}%')
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.3f}%')
            logger.info(f'Accuracy of the network on the {config.DATA.VAL_SIZE} test images (ReaL): {real_acc1:.2f}%')
            real_max_accuracy = max(real_max_accuracy, real_acc1)
            logger.info(f'Max accuracy (ReaL): {real_max_accuracy:.3f}%')
        else:
            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(f'Accuracy of the network on the {config.DATA.VAL_SIZE} test images: {acc1:.2f}%')
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.3f}%')

        data_loader_train.reset()
        data_loader_val.reset()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = config.DATA.TRAIN_SIZE // config.WORLD_SIZE // config.DATA.BATCH_SIZE_TRAIN
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, data in enumerate(data_loader):
        samples = data[0]['image']
        targets = data[0]['label'].squeeze(-1).long()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)

        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        elder_parameters = list(model.parameters())

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0, enabled=config.AMP_ENABLE)

        if grad_norm.isinf():
            assert list(model.parameters()) == elder_parameters
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\tinfinity gradient occurred due to precision of AMP')

        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()['scale']

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None and not grad_norm.isinf() and not grad_norm.isnan():  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0 or idx + 1 == num_steps:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - 1 - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t' +
                f'used {datetime.timedelta(seconds=int(end - start))}\t' +
                f'eta {datetime.timedelta(seconds=int(etas))}\t lr {lr:.6f}\t wd {wd:.4f}\t' +
                # f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.1f})\t' +
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t' +
                (f'loss_scale {int(scaler_meter.val)} ({scaler_meter.avg:.4f})\t' if config.AMP_ENABLE else '') +
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f'EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}')


@torch.no_grad()
def validate(config, data_loader, model):
    if config.DATA.REAL_LABEL:
        real_labels = RealLabelsImagenet(real_json=config.DATA.REAL_JSON_PATH)
    else:
        real_labels = None

    num_steps = config.DATA.VAL_SIZE // config.WORLD_SIZE // config.DATA.BATCH_SIZE_VAL

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()

    for idx, data in enumerate(data_loader):
        images = data[0]['image']
        target = data[0]['label'].squeeze(-1).long()
        indices = data[0]['global_index'].squeeze(-1).long()

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        if config.DATA.DATASET == 'ImageNet-21K_Winter21' or config.DATA.DATASET == 'ImageNet-21K_Fall11':
            map_to_1k = torch.from_numpy(np.loadtxt(f'{config.DATA.MAPPING_PATH}').astype(int))
            output = output[:, map_to_1k]

        # measure accuracy and record loss
        loss = criterion(output, target)

        if real_labels is not None:
            real_labels.add_result(indices, output)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0 or idx + 1 == num_steps:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{num_steps}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

    top1a, top5a = acc1_meter.avg, acc5_meter.avg

    logger.info(f' * Acc@1 {top1a:.3f} Acc@5 {top5a:.3f}')

    if real_labels is not None:
        real_top1a, real_top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
        real_top1a = reduce_tensor(torch.Tensor([real_top1a]).cuda())[0].item()
        real_top5a = reduce_tensor(torch.Tensor([real_top5a]).cuda())[0].item()
        logger.info(f' * ReaL_Acc@1 {real_top1a:.3f} Real_Acc@5 {real_top5a:.3f}')

        return top1a, top5a, real_top1a, real_top5a, loss_meter.avg
    else:
        return top1a, top5a, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, data in enumerate(data_loader):
        images = data[0]['image']
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f'throughput averaged with 30 times')
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f'batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}')
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Coreset Selection for ImageNet')
    parser.add_argument('--config', type=str, required=True, metavar='FILE', help='path to config file')
    parser.add_argument('--coreset', type=str, required=False, metavar='FILE', help='path to coreset file')
    parser.add_argument('--pretrain', type=str, required=False, metavar='FILE', help='path to pre-train checkpoint')
    parser.add_argument('--output', type=str, required=True, metavar='FILE', help='path to output folder')

    args, unparsed = parser.parse_known_args()

    with open(args.config, 'r') as f:
        config = CfgNode.load_cfg(f)
        config.LOCAL_RANK = int(os.environ['LOCAL_RANK'])
        if args.coreset is not None:
            config.DATA.DATA_PATH_TRAIN_FILE = args.coreset
        if args.pretrain is not None:
            config.MODEL.PRETRAINED = args.pretrain
        config.OUTPUT = args.output

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f'RANK and WORLD_SIZE in environment: {rank}/{world_size}')
        config.WORLD_SIZE = world_size
        config.freeze()
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    torch.set_float32_matmul_precision('high')

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # gradient accumulation need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = config.TRAIN.BASE_LR * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = config.TRAIN.MIN_LR * config.TRAIN.ACCUMULATION_STEPS
        config.defrost()
        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr
        config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    os.makedirs(config.OUTPUT + '/ckpts', exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f'{config.MODEL.NAME}')

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, 'config.yaml')
        with open(path, 'w') as f:
            f.write(config.dump())
        logger.info(f'full config saved to {path}')

    main(config)
