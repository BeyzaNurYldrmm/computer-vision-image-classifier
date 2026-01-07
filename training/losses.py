# losses
import torch.nn as nn
from functools import partial

CRITERIONS = {
    "CE": nn.CrossEntropyLoss,
    "CE_LS": partial(nn.CrossEntropyLoss, label_smoothing=0.1),
    "BCE": nn.BCEWithLogitsLoss,
    "MSE": nn.MSELoss,
    "SmoothL1": nn.SmoothL1Loss,
}

def build_criterion(name, device):
    if name not in CRITERIONS:
        raise ValueError(f"Tanımsız loss tercihi. Tercih:{name}")
    return CRITERIONS[name]().to(device)

