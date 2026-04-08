"""Probe training loop with early stopping."""

import json
import time
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


class ProbeTrainer:
    """Trains a probe model with early stopping on validation F1.

    Args:
        model: Probe model (LinearProbe, MLPProbe, or ConvProbe).
        device: Torch device.
        lr: Learning rate.
        weight_decay: L2 regularization.
        pos_weight: Positive class weight for BCEWithLogitsLoss.
        patience: Early stopping patience (epochs without improvement).
        max_epochs: Maximum training epochs.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        pos_weight: float = 1.0,
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

        pos_weight_tensor = torch.tensor([pos_weight], device=device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        # Tracking
        self.best_val_f1 = 0.0
        self.best_model_state = None
        self.epochs_without_improvement = 0
        self.history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_path: Optional[str] = None,
    ) -> Dict:
        """Full training loop with early stopping.

        Returns:
            Dictionary with best metrics and training history.
        """
        for epoch in range(self.max_epochs):
            # Training
            train_loss = self._train_epoch(train_loader)

            # Validation
            val_metrics = self._evaluate(val_loader)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            # Early stopping check
            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.epochs_without_improvement = 0

                if checkpoint_path:
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.best_model_state, checkpoint_path)
            else:
                self.epochs_without_improvement += 1

            # Print progress every 10 epochs or on improvement
            if (epoch + 1) % 10 == 0 or self.epochs_without_improvement == 0:
                print(
                    f"  Epoch {epoch+1:3d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val F1: {val_metrics['f1']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f}"
                    f"{'  *' if self.epochs_without_improvement == 0 else ''}"
                )

            if self.epochs_without_improvement >= self.patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={self.patience})")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)

        return {
            "best_val_f1": self.best_val_f1,
            "epochs_trained": epoch + 1,
            "history": self.history,
        }

    def _train_epoch(self, loader: DataLoader) -> float:
        """Train for one epoch, return average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict:
        """Evaluate on a DataLoader, return metrics dict."""
        self.model.eval()
        all_logits = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(features)
            loss = self.criterion(logits, labels)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item()
            num_batches += 1

        all_logits = torch.cat(all_logits).numpy()
        all_labels = torch.cat(all_labels).numpy()
        predictions = (all_logits > 0).astype(np.float32)

        return {
            "loss": total_loss / max(num_batches, 1),
            "accuracy": float(accuracy_score(all_labels, predictions)),
            "f1": float(f1_score(all_labels, predictions, zero_division=0)),
            "precision": float(precision_score(all_labels, predictions, zero_division=0)),
            "recall": float(recall_score(all_labels, predictions, zero_division=0)),
        }

    @torch.no_grad()
    def evaluate_test(self, test_loader: DataLoader) -> Dict:
        """Evaluate best model on test set."""
        return self._evaluate(test_loader)
