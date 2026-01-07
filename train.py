# train.py
import argparse
import torch
from utils.config import load_config
from data_loading import get_data_loader, visualize_sample
from model.model_factory import build_model
from training.losses import build_criterion
from training.optimizers import build_optimizer
from training.schedulers import build_scheduler
from pipeline import Pipeline
from utils.experiment import create_experiment_dir
from training.schema import validate_config

def main():
    # -------- CLI --------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    # -------- CONFIG --------
    cfg = load_config(args.config)
    validate_config(cfg)

    # -------- DEVICE --------
    device = torch.device(
        "cuda"
        if cfg["system"]["device"] == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    # -------- EXPERIMENT DIR --------
    exp_dir = create_experiment_dir(
        cfg["model"]["name"],
        args.config
    )

    # -------- DATA --------
    loaders, num_class = get_data_loader(cfg)
    if cfg["data"]["visualize"] :
        visualize_sample(loaders["train"], cfg["data"]["num_sample_image_show"])


    # -------- MODEL --------
    model_cfg = (
        cfg["model"]
        | cfg.get("vit", {})
        | cfg.get("swin", {})
        | cfg.get("convnext", {})
        | cfg.get("efficientnet", {})
        | cfg.get("beit",{})
    )

    model = build_model(model_cfg, num_class, device)

    # -------- TRAINING COMPONENTS --------
    criterion = build_criterion(cfg["training"]["criterion"], device)
    optimizer = build_optimizer(
        cfg["training"]["optimizer"],
        model,
        cfg["training"]["lr"]
    )
    
    scheduler = build_scheduler(
        optimizer=optimizer,
        name=cfg["training"]["scheduler"],
        epochs=cfg["training"]["epochs"],
        mode=cfg["training"]["mode"],
        patience=cfg["training"]["patience"]
    )


    # -------- PIPELINE --------

    runner = Pipeline(
        model=model,
        loaders=loaders,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        exp_dir=exp_dir,
        model_name=cfg["model"]["name"]
    )

    runner.run(
        epochs=cfg["training"]["epochs"],
        patience=cfg["training"]["patience"],
        min_delta=cfg["training"]["min_delta"],
        mode=cfg["training"]["mode"],
        exp_dir=exp_dir
    )




if __name__ == "__main__":
    main()
