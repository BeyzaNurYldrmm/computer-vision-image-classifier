import json
from pathlib import Path
import torch

from training.train import train_model
from testing.test import test_model
from training.early_stopping import EarlyStopping

class Pipeline:
    """
    - Eğitim + test 
    """

    def __init__(
        self,
        model,
        loaders: dict,
        optimizer,
        criterion,
        scheduler,
        device,
        exp_dir: Path,
        model_name: str
    ):
        """
        loaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }
        """
        
        self.model = model
        self.loaders = loaders
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.exp_dir = exp_dir
        self.model_name = model_name

        # açık ve güvenli erişim
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]
        self.test_loader = loaders["test"]

   

    def run(self, epochs, patience, min_delta, mode, exp_dir):

        early_stopper = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            mode=mode,
            restore_best=True,
            verbose=True
        )

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        for epoch in range(epochs):
            metrics = train_model(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                device=self.device
            )

            # kayıt
            for k in history:
                history[k].append(metrics[k])

            # scheduler
            if self.scheduler:
                try:
                    self.scheduler.step(metrics["val_loss"])
                except TypeError:
                    self.scheduler.step()

            # early stopping
            if early_stopper.step(metrics["val_loss"], self.model):
                print(f"[Pipeline] Early stopping ->{epoch+1}. epoch ")
                break

        # ---------------- TEST ----------------
        test_results = test_model(
            self.model,
            self.test_loader,
            self.criterion,
            self.device,
            self.exp_dir
        )
        # ---------------- SAVE ----------------
        self._save_model(self.model)
        self._save_history(history)

        return {
            "history": history,
            "test": test_results
        }
    

    def _save_model(self, model):
        model_path = self.exp_dir / "model.pth"
        torch.save(model.state_dict(), model_path)

    def _save_history(self, history: dict):
        history_path = self.exp_dir / "history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

