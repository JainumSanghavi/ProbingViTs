"""Probe trainer for depth regression.

Key differences from ProbeTrainer (boundary):
- Uses MSELoss instead of BCEWithLogitsLoss
- Monitors val_mae (minimize) instead of val_f1 (maximize)
- No pos_weight parameter
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class DepthProbeTrainer:
    """Trains a regression probe with early stopping on validation MAE.

    Args:
        model: Probe model (LinearProbe or MLPProbe with linear output).
        device: Torch device.
        lr: Learning rate.
        weight_decay: L2 regularization.
        patience: Early stopping patience (epochs without improvement).
        max_epochs: Maximum training epochs.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        max_epochs: int = 100,
    ):
        self.model = model.to(device)
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()

        self.best_val_mae = float("inf")
        self.best_model_state = None
        self.epochs_without_improvement = 0
        self.history = {"train_loss": [], "val_loss": [], "val_mae": []}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_path: Optional[str] = None,
    ) -> Dict:
        """Full training loop with early stopping on val_mae.

        Returns:
            Dict with best_val_mae, epochs_trained, history.
        """
        for epoch in range(self.max_epochs):
            train_loss = self._train_epoch(train_loader)
            val_metrics = self._evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_mae"].append(val_metrics["mae"])

            if val_metrics["mae"] < self.best_val_mae:
                self.best_val_mae = val_metrics["mae"]
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.epochs_without_improvement = 0

                if checkpoint_path:
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.best_model_state, checkpoint_path)
            else:
                self.epochs_without_improvement += 1

            if (epoch + 1) % 10 == 0 or self.epochs_without_improvement == 0:
                print(
                    f"  Epoch {epoch+1:3d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val MAE: {val_metrics['mae']:.4f}"
                    f"{'  *' if self.epochs_without_improvement == 0 else ''}"
                )

            if self.epochs_without_improvement >= self.patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={self.patience})")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)

        return {
            "best_val_mae": self.best_val_mae,
            "epochs_trained": epoch + 1,
            "history": self.history,
        }

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(features)
            loss = self.criterion(preds, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict:
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            preds = self.model(features)
            loss = self.criterion(preds, labels)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item()
            num_batches += 1

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        mae = float(np.abs(all_preds - all_labels).mean())
        rmse = float(np.sqrt(((all_preds - all_labels) ** 2).mean()))

        return {
            "loss": total_loss / max(num_batches, 1),
            "mae": mae,
            "rmse": rmse,
        }

    @torch.no_grad()
    def evaluate_test(self, test_loader: DataLoader) -> Dict:
        """Evaluate best model on test set."""
        return self._evaluate(test_loader)
