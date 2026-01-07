# optimizers
import torch
from functools import partial
from adabelief_pytorch import AdaBelief
from torch_optimizer import LARS, Lamb, Lookahead

OPTIMIZERS = {
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
    "Adam": torch.optim.Adam,
    "AdaBelief": AdaBelief,
    "Adamax": torch.optim.Adamax,
    "Lars-ef": LARS,
    "Lamb": Lamb,
}

def build_optimizer(name, model, lr):
    if name not in OPTIMIZERS:
        raise ValueError(f"Tanımsız optimizer: {name}")

    return OPTIMIZERS[name](model.parameters(), lr=lr)

