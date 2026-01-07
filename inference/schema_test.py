def validate_config(cfg):
    # ---------- SYSTEM ----------
    assert "system" in cfg
    assert cfg["system"]["device"] in ["cpu", "cuda"]
 
    # ---------- TEST ----------
    test= cfg.get("test")