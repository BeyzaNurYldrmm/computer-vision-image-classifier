import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from pathlib import Path



def test_model(model, test_loader, criterion, device, exp_dir):
    model.eval()

    results = {
        "accuracy": 0.0,
        "loss": 0.0,
        "all_preds": [],
        "all_labels": [],
        "confidences": []
    }

    correct = total = total_loss = 0

    # %10 güven aralıkları
    confidence_bins = {
        f"{i}-{i+10}": {"correct": 0, "wrong": 0}
        for i in range(10, 100, 10)
    }

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing", unit="batch", colour='green'):
            x, y = x.to(device), y.to(device)
            out = model(x)

            loss = criterion(out, y)
            total_loss += loss.item()

            probs = torch.softmax(out, dim=1)
            confs, preds = probs.max(1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            results["confidences"].extend(confs.cpu().tolist())
            results["all_preds"].extend(preds.cpu().tolist())
            results["all_labels"].extend(y.cpu().tolist())

            # güven aralığı istatistikleri
            for i in range(len(y)):
                conf_percent = confs[i].item() * 100
                is_correct = (preds[i] == y[i]).item()

                for low in range(10, 100, 10):
                    high = low + 10
                    if low <= conf_percent < high:
                        key = f"{low}-{high}"
                        if is_correct:
                            confidence_bins[key]["correct"] += 1
                        else:
                            confidence_bins[key]["wrong"] += 1
                        break

    # metrikler
    results["accuracy"] = 100 * correct / total
    results["loss"] = total_loss / len(test_loader)



    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)


    log("\n===== TEST SONUÇLARI =====")
    log(f"Accuracy: %{results['accuracy']:.2f}")
    log(f"Loss: {results['loss']:.4f}")

    log("\n%10 Aralıklar ile tahminler")
    for k, v in confidence_bins.items():
        toplam = v["correct"] + v["wrong"]
        if toplam > 0:
            acc = 100 * v["correct"] / toplam
            log(
                f"%{k} | "
                f"Toplam: {toplam:4d} | "
                f"Doğru: {v['correct']:4d} | "
                f"Yanlış: {v['wrong']:4d} | "
                f"Accuracy: %{acc:6.2f}"
            )
        else:
            log(f"%{k} | Veri yok")

    report_path = Path(exp_dir) / "test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))



   
    cm= confusion_matrix(results["all_labels"],
        results["all_preds"],
        normalize="true")
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True,fmt='.2f', cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    save_path = Path(exp_dir) / "confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(0.1)
    plt.close("all")

    return results
