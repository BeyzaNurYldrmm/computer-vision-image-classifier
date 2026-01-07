from model_test import run_inference
import argparse
import torch
from utils.config import load_config
from schema_test import validate_config

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

    # -------- TEST --------
    model_name, pred,conf= run_inference(
    model_name=cfg["test"]["pred_model"],  
    weights_path=cfg["test"]["save_mdl_path"],
    image_path=cfg["test"]["img_path"],
    num_class=cfg["test"]["num_class"],
    image_size=cfg["test"]["img_size"]
    )

    print(f"Model           : {model_name}")
    print(f"Predicted class : {pred}")
    print(f"Confidence      : %{conf:.2f}")



if __name__ == "__main__":
    main()





