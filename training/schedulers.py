import torch


def build_scheduler(optimizer, name, epochs, mode, patience):
    if name == "Cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )

    if name == "Step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )

    if name == "Plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            patience=patience
        )

    if name in [None, "None"]:
        return None

    raise ValueError(f"Tanımsız scheduler: {name}")
