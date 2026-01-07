import timm 

def fine_tune_beit(K, num_class, device):
    model = timm.create_model(
        "beitv2_base_patch16_224",
        pretrained=True,
        num_classes=num_class
    ).to(device)

    for p in model.parameters():
        p.requires_grad = False

    for block in model.blocks[-K:]:
        for p in block.parameters():
            p.requires_grad = True

    for p in model.head.parameters():
        p.requires_grad = True

    return model
