import json
import warnings
import torch.distributed as dist
from timm.data import Mixup

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy, DALIGenericIterator
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    from nvidia.dali.auto_aug import auto_augment
    import numpy as np
    import fcntl
except:
    raise ImportError('Please install DALI from https://www.github.com/NVIDIA/DALI to run this program.')


def compute_dataset(config):
    config.defrost()
    with open(config.DATA.DATA_PATH_TRAIN_FILE, 'r') as f:
        config.DATA.TRAIN_SIZE = len(f.readlines())

    if config.DATA.DATASET == 'ImageNet-1K':
        config.DATA.VAL_SIZE = 50000
        config.MODEL.NUM_CLASSES = 1000
    elif config.DATA.DATASET == 'ImageNet-21K_Winter21':
        # use ImageNet-1K val temporarily
        config.DATA.VAL_SIZE = 50000
        config.MODEL.NUM_CLASSES = 19168  # 19167 + n04399382
    elif config.DATA.DATASET == 'ImageNet-21K-Processed_Winter21':
        config.DATA.VAL_SIZE = 522500
        config.MODEL.NUM_CLASSES = 10450
    elif config.DATA.DATASET == 'ImageNet-21K_Fall11':
        # use ImageNet-1K val temporarily
        config.DATA.VAL_SIZE = 50000
        config.MODEL.NUM_CLASSES = 21842  # 21841 + n04399382
    elif config.DATA.DATASET == 'ImageNet-21K-Processed_Fall11':
        config.DATA.VAL_SIZE = 561050
        config.MODEL.NUM_CLASSES = 11221
    else:
        raise NotImplementedError('We only support ImageNet Now.')
    config.freeze()


def build_loader(config):
    compute_dataset(config)

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    train_pipe = create_dali_pipeline(
        batch_size=config.DATA.BATCH_SIZE_TRAIN,
        num_threads=config.DATA.NUM_WORKERS,
        device_id=config.LOCAL_RANK,
        seed=config.SEED + dist.get_rank(),
        data_path=config.DATA.DATA_PATH_TRAIN,
        shard_id=dist.get_rank(),
        train_file=config.DATA.DATA_PATH_TRAIN_FILE,
        py_num_workers=4,
        py_start_method='spawn',
        config=config)

    train_pipe.build()

    # drop last batch
    dataloader_train = DALIClassificationIteratorForTrain(
        train_pipe,
        size=-1
    )

    print(f'local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataloader')

    val_pipe = create_dali_pipeline(
        batch_size=config.DATA.BATCH_SIZE_VAL,
        num_threads=config.DATA.NUM_WORKERS,
        device_id=config.LOCAL_RANK,
        seed=config.SEED + dist.get_rank(),
        data_path=config.DATA.DATA_PATH_VAL,
        shard_id=dist.get_rank(),
        train_file=None,
        py_num_workers=4,
        py_start_method='spawn',
        config=config)

    val_pipe.build()

    # External Source in parallel mode does not support partial batches.
    if config.DATA.VAL_SIZE % config.WORLD_SIZE != 0 or (
            config.DATA.VAL_SIZE // config.WORLD_SIZE) % config.DATA.BATCH_SIZE_VAL != 0:
        warnings.warn(
            'Not divisible; only part of validation set would be used due to drop policy (partial policy not support)')

    dataloader_val = DALIClassificationIteratorForVal(
        val_pipe,
        size=-1)

    print(f'local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataloader')

    return dataloader_train, dataloader_val, mixup_fn


@pipeline_def(enable_conditionals=True)
def create_dali_pipeline(data_path, shard_id, train_file, config):
    crop = config.DATA.IMG_SIZE
    size = config.DATA.IMG_SIZE
    num_shards = config.WORLD_SIZE

    if config.DATA.INTERPOLATION == 'cubic':
        interpolation_mode = types.INTERP_CUBIC
    elif config.DATA.INTERPOLATION == 'lanczos':
        interpolation_mode = types.INTERP_LANCZOS3
    else:
        # default linear
        interpolation_mode = types.INTERP_LINEAR

    if train_file is not None:
        jpegs, labels, file_names = fn.external_source(
            source=ExternalInputCallableForTrain(data_path=data_path,
                                                 batch_size=config.DATA.BATCH_SIZE_TRAIN,
                                                 shard_id=shard_id,
                                                 num_shards=num_shards,
                                                 data_file_path=train_file),
            num_outputs=3,
            batch=False,
            parallel=True,
            dtype=[types.UINT8, types.INT32, types.INT32])

    else:
        jpegs, labels, file_names = fn.external_source(
            source=ExternalInputCallableForVal(data_path=data_path,
                                               batch_size=config.DATA.BATCH_SIZE_VAL,
                                               shard_id=shard_id,
                                               num_shards=num_shards),
            num_outputs=3,
            batch=False,
            parallel=True,
            dtype=[types.UINT8, types.INT32, types.INT32]
        )

    dali_device = 'gpu'
    decoder_device = 'mixed'

    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920
    host_memory_padding = 140544512
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980
    preallocate_height_hint = 6430

    if train_file is not None:
        eval_images = fn.decoders.image(
            jpegs,
            device=decoder_device,
            output_type=types.RGB)

        eval_images = fn.resize(
            eval_images,
            device=dali_device,
            size=size,
            mode='not_smaller',
            interp_type=interpolation_mode,
            antialias=False)

        eval_output = eval_images.gpu()
        eval_output = fn.crop_mirror_normalize(
            eval_output,
            dtype=types.FLOAT,
            output_layout='CHW',
            crop=(crop, crop),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )

        images = fn.decoders.image_random_crop(
            jpegs,
            device=decoder_device,
            output_type=types.RGB,
            device_memory_padding=device_memory_padding,
            host_memory_padding=host_memory_padding,
            preallocate_width_hint=preallocate_width_hint,
            preallocate_height_hint=preallocate_height_hint,
            random_aspect_ratio=[3. / 4., 4. / 3.],
            random_area=[0.08, 1.0],
            num_attempts=100)

        images = fn.resize(
            images,
            device=dali_device,
            resize_x=crop,
            resize_y=crop,
            interp_type=interpolation_mode,
            antialias=False)

        images = images.gpu()

        rng = fn.random.coin_flip(probability=0.5)
        images = fn.flip(images, horizontal=rng)

        if config.AUG.AUTO_AUGMENT:
            output = auto_augment.auto_augment_image_net(images, shape=[crop, crop])
        else:
            # TODO: colorjitter rand_augment?
            output = images

    else:
        images = fn.decoders.image(
            jpegs,
            device=decoder_device,
            output_type=types.RGB)

        images = fn.resize(
            images,
            device=dali_device,
            size=size,
            mode='not_smaller',
            interp_type=interpolation_mode,
            antialias=False)

        output = images.gpu()

    output = fn.crop_mirror_normalize(
        output,
        dtype=types.FLOAT,
        output_layout='CHW',
        crop=(crop, crop),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )

    labels = labels.gpu()
    file_names = file_names.gpu()
    if train_file is not None:
        return output, eval_output, labels, file_names
    else:
        return output, labels, file_names


class DALIClassificationIteratorForTrain(DALIGenericIterator):
    def __init__(self,
                 pipelines,
                 size=-1,
                 reader_name=None,
                 auto_reset=False,
                 fill_last_batch=None,
                 dynamic_shape=False,
                 last_batch_padded=False,
                 last_batch_policy=LastBatchPolicy.FILL,
                 prepare_first_batch=False):
        super(DALIClassificationIteratorForTrain, self).__init__(pipelines,
                                                                 ['image', 'eval_image', 'label', 'global_index'],
                                                                 size,
                                                                 reader_name=reader_name,
                                                                 auto_reset=auto_reset,
                                                                 fill_last_batch=fill_last_batch,
                                                                 dynamic_shape=dynamic_shape,
                                                                 last_batch_padded=last_batch_padded,
                                                                 last_batch_policy=last_batch_policy,
                                                                 prepare_first_batch=prepare_first_batch)


class DALIClassificationIteratorForVal(DALIGenericIterator):
    def __init__(self,
                 pipelines,
                 size=-1,
                 reader_name=None,
                 auto_reset=False,
                 fill_last_batch=None,
                 dynamic_shape=False,
                 last_batch_padded=False,
                 last_batch_policy=LastBatchPolicy.FILL,
                 prepare_first_batch=False):
        super(DALIClassificationIteratorForVal, self).__init__(pipelines, ['image', 'label', 'global_index'],
                                                               size,
                                                               reader_name=reader_name,
                                                               auto_reset=auto_reset,
                                                               fill_last_batch=fill_last_batch,
                                                               dynamic_shape=dynamic_shape,
                                                               last_batch_padded=last_batch_padded,
                                                               last_batch_policy=last_batch_policy,
                                                               prepare_first_batch=prepare_first_batch)


class ExternalInputCallableForTrain:
    def __init__(self, data_path, batch_size, shard_id, num_shards, data_file_path):
        self.images_dir = data_path
        self.batch_size = batch_size
        with open(data_file_path, 'r') as f:
            file_label = [line.rstrip().split(' ') for line in f if line != '']
            self.files, self.labels = zip(*file_label)
        self.shard_id = shard_id
        self.num_shards = num_shards
        # If the dataset size is not divisible by number of shards, the trailing samples will
        # be omitted.
        self.shard_size = len(self.files) // num_shards
        self.shard_offset = self.shard_size * shard_id
        # drop last batch
        self.full_iterations = self.shard_size // batch_size
        self.perm = None
        self.last_seen_epoch = None  # so that we don't have to recompute the `self.perm` for every sample

    def __call__(self, sample_info):
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration
        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            self.perm = np.random.default_rng(seed=42 + sample_info.epoch_idx).permutation(len(self.files))

        sample_idx = self.perm[sample_info.idx_in_epoch + self.shard_offset]
        jpeg_filename = self.files[sample_idx]
        label = np.int32([self.labels[sample_idx]])
        with open(self.images_dir + jpeg_filename, 'rb') as f:
            encoded_img = np.frombuffer(f.read(), dtype=np.uint8)
        return encoded_img, label, np.int32([sample_idx])


class ExternalInputCallableForVal:
    def __init__(self, data_path, batch_size, shard_id, num_shards):
        self.images_dir = data_path
        self.batch_size = batch_size
        with open(self.images_dir + '/file_list.txt', 'r') as f:
            file_label = [line.rstrip().split(' ') for line in f if line != '']
            self.files, self.labels = zip(*file_label)
        self.shard_id = shard_id
        self.num_shards = num_shards
        # If the dataset size is not divisible by number of shards, the trailing samples will
        # be omitted.
        self.shard_size = len(self.files) // num_shards
        self.shard_offset = self.shard_size * shard_id
        # drop last batch
        self.full_iterations = self.shard_size // batch_size

    def __call__(self, sample_info):
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration

        sample_idx = sample_info.idx_in_epoch + self.shard_offset
        jpeg_filename = self.files[sample_idx]
        label = np.int32([self.labels[sample_idx]])
        with open(self.images_dir + jpeg_filename, 'rb') as f:
            encoded_img = np.frombuffer(f.read(), dtype=np.uint8)
        return encoded_img, label, np.int32([sample_idx])


''' Real labels evaluator for ImageNet
Paper: Are we done with ImageNet? - https://arxiv.org/abs/2006.07159
Based on Numpy example at https://github.com/google-research/reassessed-imagenet

Hacked together by / Copyright 2020 Ross Wightman
'''


class RealLabelsImagenet:
    def __init__(self, real_json='real.json', topk=(1, 5)):
        with open(real_json) as real_labels:
            real_labels = json.load(real_labels)
            real_labels = {i: labels for i, labels in enumerate(real_labels)}
        self.real_labels = real_labels
        self.topk = topk
        self.is_correct = {k: [] for k in topk}
        self.sample_idx = 0

    def add_result(self, indices, output):
        maxk = max(self.topk)
        _, pred_batch = output.topk(maxk, 1, True, True)
        pred_batch = pred_batch.cpu().numpy()
        cnt = 0
        for pred in pred_batch:
            index = indices[cnt].item()

            if self.real_labels[index]:
                for k in self.topk:
                    self.is_correct[k].append(
                        any([p in self.real_labels[index] for p in pred[:k]]))
            cnt += 1
            self.sample_idx += 1

    def get_accuracy(self, k=None):
        if k is None:
            return {k: float(np.mean(self.is_correct[k])) * 100 for k in self.topk}
        else:
            return float(np.mean(self.is_correct[k])) * 100
