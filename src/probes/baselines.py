"""Baseline models for comparison."""

import numpy as np
import torch


class MajorityClassBaseline:
    """Always predicts the majority class (non-boundary, label=0).

    Provides lower bound for probe performance.
    """

    def __init__(self):
        self.majority_class = 0
        self.name = "majority_class"

    def fit(self, labels: np.ndarray):
        """Determine majority class from training labels."""
        self.majority_class = int(labels.mean() < 0.5)  # 0 if <50% are 1s
        return self

    def predict(self, num_samples: int) -> np.ndarray:
        """Predict majority class for all samples."""
        return np.full(num_samples, self.majority_class, dtype=np.float32)

    def evaluate(self, labels: np.ndarray) -> dict:
        """Evaluate baseline on given labels."""
        predictions = self.predict(len(labels))
        accuracy = (predictions == labels).mean()

        # F1 for majority class baseline
        if self.majority_class == 0:
            # Predicts all 0 → TP=0, FP=0, FN=sum(labels)
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            # Predicts all 1 → TP=sum(labels), FP=sum(1-labels), FN=0
            tp = labels.sum()
            fp = (1 - labels).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = 1.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }


class RandomBaseline:
    """Random predictions calibrated to dataset class distribution.

    Predicts boundary with probability equal to the training set boundary rate.
    """

    def __init__(self, seed: int = 42):
        self.boundary_rate = 0.5
        self.seed = seed
        self.name = "random"

    def fit(self, labels: np.ndarray):
        """Learn boundary rate from training labels."""
        self.boundary_rate = float(labels.mean())
        return self

    def predict(self, num_samples: int) -> np.ndarray:
        """Random predictions based on learned boundary rate."""
        rng = np.random.RandomState(self.seed)
        return (rng.random(num_samples) < self.boundary_rate).astype(np.float32)

    def evaluate(self, labels: np.ndarray) -> dict:
        """Evaluate baseline on given labels."""
        predictions = self.predict(len(labels))
        accuracy = (predictions == labels).mean()

        tp = ((predictions == 1) & (labels == 1)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }
