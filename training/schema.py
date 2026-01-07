# schema

def validate_config(cfg):
    # ---------- SYSTEM ----------
    assert "system" in cfg
    assert cfg["system"]["device"] in ["cpu", "cuda"]

    # ---------- DATA ----------
    data = cfg.get("data")
    assert data is not None
    assert isinstance(data["batch_size"], int)
    assert data["batch_size"] > 0

    # ---------- MODEL ----------
    model = cfg.get("model")
    assert model is not None
    assert model["name"] in ["ViT", "Swin", "ConvNext", "EfficientNet","BEiT"]
    assert model["train_type"] in [0, 1]

    # ---------- MODEL-SPECIFIC ----------
    if model["name"] == "ViT":
        vit = cfg.get("vit")
        assert vit is not None
        assert vit["patch_size"] > 0
        assert vit["dim"] % vit["heads"] == 0

    if model["name"] == "Swin":
        swin = cfg.get("swin")
        assert swin is not None
        assert len(swin["depths"]) == len(swin["num_heads"])

    # ---------- TRAINING ----------
    tr = cfg.get("training")
    assert tr["epochs"] > 0
    assert tr["lr"] > 0
    assert tr["optimizer"] in ["AdamW", "SGD", "Adam", "Lamb","AdaBelief","Adamax","Lars-ef"]
    assert tr["scheduler"] in ["Cosine", "Step", "Plateau", "None"]
    assert tr["criterion"] in [ "CE", "CE_LS", "BCE", "MSE", "SmoothL1"]

   


