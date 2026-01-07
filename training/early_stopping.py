import numpy as np
import torch


class EarlyStopping:
    """
    Framework seviyesinde Early Stopping.
    - mode: 'min'  -> loss gibi minimize edilen metrikler
    - mode: 'max'  -> accuracy gibi maximize edilen metrikler
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-3,
        mode: str = "min",
        restore_best: bool = True,
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.should_stop = False

        if mode not in ["min", "max"]:
            raise ValueError("mode sadece 'min' veya 'max' olabilir")

    def _is_improvement(self, current, best):
        if self.mode == "min":
            return best - current > self.min_delta
        else:  # max
            return current - best > self.min_delta

    def step(self, current_score: float, model: torch.nn.Module) -> bool:
        """
        Her epoch sonunda çağrılır.
        True dönerse eğitim durdurulmalıdır.
        """

        if self.best_score is None:
            self.best_score = current_score
            self._save_model(model)
            return False

        if self._is_improvement(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            self._save_model(model)

            if self.verbose:
                print(f"[EarlyStopping] İyileşme tespit edildi → best = {self.best_score:.6f}")

        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] İyileşme yok ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print("[EarlyStopping] Early stopping tetiklendi")

                if self.restore_best:
                    self._restore_model(model)

                return True

        return False

    def _save_model(self, model):
        if self.restore_best:
            self.best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    def _restore_model(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
