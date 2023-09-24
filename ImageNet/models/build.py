from yacs.config import CfgNode
import models
from .resnet import resnet50
from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .conv_next import convnext_tiny


def build_model(config):
    model_type = config.MODEL.TYPE

    if model_type == 'swin':
        with open(config.MODEL.MODEL_CONFIG_PATH, 'r') as f:
            model_config = CfgNode.load_cfg(f)
            model_config.freeze()

        # accelerate layernorm
        if config.ACCELERATION.FUSED_LAYERNORM:
            try:
                import apex as amp
                layernorm = amp.normalization.FusedLayerNorm
            except:
                layernorm = None
                print('To use FusedLayerNorm, please install apex.')
        else:
            import torch.nn as nn
            layernorm = nn.LayerNorm

        model = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=model_config.PATCH_SIZE,
            in_chans=model_config.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=model_config.EMBED_DIM,
            depths=model_config.DEPTHS,
            num_heads=model_config.NUM_HEADS,
            window_size=model_config.WINDOW_SIZE,
            mlp_ratio=model_config.MLP_RATIO,
            qkv_bias=model_config.QKV_BIAS,
            qk_scale=model_config.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=model_config.APE,
            norm_layer=layernorm,
            patch_norm=model_config.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            fused_window_process=config.ACCELERATION.FUSED_WINDOW_PROCESS)

    elif model_type == 'swinv2':
        with open(config.MODEL.MODEL_CONFIG_PATH, 'r') as f:
            model_config = CfgNode.load_cfg(f)
            model_config.freeze()

        model = SwinTransformerV2(
            img_size=config.DATA.IMG_SIZE,
            patch_size=model_config.PATCH_SIZE,
            in_chans=model_config.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=model_config.EMBED_DIM,
            depths=model_config.DEPTHS,
            num_heads=model_config.NUM_HEADS,
            window_size=model_config.WINDOW_SIZE,
            mlp_ratio=model_config.MLP_RATIO,
            qkv_bias=model_config.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=model_config.APE,
            patch_norm=model_config.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            pretrained_window_sizes=model_config.PRETRAINED_WINDOW_SIZES)

    elif model_type == 'resnet':
        model = resnet50(num_classes=config.MODEL.NUM_CLASSES)

    elif model_type == 'vit':
        # accelerate layernorm
        if config.ACCELERATION.FUSED_LAYERNORM:
            try:
                import apex as amp
                layernorm = amp.normalization.FusedLayerNorm
            except:
                layernorm = None
                print('To use FusedLayerNorm, please install apex.')
        else:
            import torch.nn as nn
            layernorm = nn.LayerNorm

        model = models.vit_b_16(
            image_size=config.DATA.IMG_SIZE,
            dropout=config.MODEL.DROP_RATE,
            norm_layer=layernorm)

    elif model_type == 'convnext':
        model = convnext_tiny(
            num_classes=config.MODEL.NUM_CLASSES,
            drop_path_rate=config.MODEL.DROP_PATH_RATE)

    else:
        raise NotImplementedError(f'Unknown model: {model_type}')

    return model
