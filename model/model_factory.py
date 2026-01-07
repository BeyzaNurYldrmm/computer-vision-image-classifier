# model/model_factory.py
import torch
import timm

from model.ViT_model import ViT
from model.ConvNeXT import convnext_model
from model.swin_tr import Swin
from model.efficientNet_mdl import efficientnet
from model.BEiT import fine_tune_beit


def build_model(model_cfg: dict, num_classes: int, device: torch.device):
    """
    model_cfg:
    cfg["model"] + cfg.get("vit"/"swin"/"convnext"/"efficientnet")
    """
    name = model_cfg["name"]
    train_type = model_cfg.get("train_type", 0)

    if name == "ViT":
        model = _build_vit(model_cfg, num_classes, train_type)

    elif name == "ConvNext":
        model = _build_convnext(model_cfg, num_classes, train_type, device)

    elif name == "Swin":
        model = _build_swin(model_cfg, num_classes, train_type)

    elif name == "EfficientNet":
        model = _build_efficientnet(model_cfg, num_classes, train_type, device)

    elif name == "BEiT":
        model = _build_beit(model_cfg, num_classes, device)

    else:
        raise ValueError(f"Desteklenmeyen model: {name}")

    return model.to(device)


def _build_vit(cfg, num_classes, train_type):
    if train_type == 0:
        return ViT(
            image_size=cfg["image_size"],
            patch_size=cfg["patch_size"],
            num_classes=num_classes,
            dim=cfg["dim"],
            depth=cfg["depth"],
            heads=cfg["heads"],
            mlp_dim=cfg["mlp_dim"]
        )
    else:
        model = timm.create_model(
            "vit_large_r50_s32_224",#  <- vitc / norm -> "vit_base_patch16_224"
            pretrained=True,
            num_classes=num_classes
        )

        _freeze_all(model)
        _unfreeze_last_blocks(model, cfg.get("fine_tune_last_k_layers", 0))
        _unfreeze_head(model)
    return model


def _build_convnext(cfg, num_classes, train_type, device):
    return convnext_model(
        dims=cfg["dims"],
        cn_drop_path_rate=cfg["cn_drop_path_rate"],
        layer_scale_init_value=cfg["layer_scale_init_value"],
        head_init_scale=cfg["head_init_scale"],
        depths=cfg["cn_depths"],
        k=cfg["fine_tune_last_k_layers"],
        in_chans=3,
        num_class=num_classes,
        device=device,
        ft=bool(train_type)
    )


def _build_swin(cfg, num_classes, train_type):
    return Swin(
        train_type == 1,
        num_classes,
        cfg.get("fine_tune_last_k_layers", 0),
        cfg["image_size"],
        cfg["patch_size"],
        3,
        cfg["embed_dim"],
        cfg["depths"],
        cfg["num_heads"],
        cfg["window_size"],
        cfg["mlp_ratio"],
        cfg["qkv_bias"],
        cfg["qk_scale"],
        cfg["drop_rate"],
        cfg["attn_drop_rate"],
        cfg["sw_drop_path_rate"],
        cfg["norm_layer"],
        cfg["ape"],
        cfg["patch_norm"],
        cfg["use_checkpoint"],
        cfg["fused_window_process"]
    )


def _build_efficientnet(cfg, num_classes, train_type, device):
    return efficientnet(
        cfg["model_version"],
        train_type == 1,
        num_classes,
        device
    )

def _build_beit(cfg, num_classes, device):
    return fine_tune_beit(
        K=cfg["fine_tune_last_k_layers"], 
        num_class=num_classes,
        device=device
    )


def _freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_last_blocks(model, k: int):
    if k <= 0:
        return
    if hasattr(model, "blocks"):
        for block in model.blocks[-k:]:
            for p in block.parameters():
                p.requires_grad = True


def _unfreeze_head(model):
    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True
