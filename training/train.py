# training/train.py
import torch
from tqdm import tqdm

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device
):
    model.train()
    total_loss = correct = total = 0

    for x, y in tqdm(train_loader, desc="Trainig", unit="batch", colour='green'):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = out.max(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total
    print(f"Train acc:{train_acc}%,  Train loss:{train_loss}")

    model.eval()
    val_loss = val_correct = val_total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            val_loss += loss.item()
            _, pred = out.max(1)
            val_correct += (pred == y).sum().item()
            val_total += y.size(0)

    val_loss /= len(val_loader)
    val_acc = 100 * val_correct / val_total
    print(f"Val acc:{val_acc}%, Val loss: {val_loss}")

    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    }
