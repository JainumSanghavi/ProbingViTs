"""Dataset over cached ViT hidden states and patch labels."""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class HiddenStateDataset(Dataset):
    """Dataset that yields per-patch (hidden_dim,) vectors and binary labels.

    Preloads all hidden states for a single layer into contiguous tensors
    at init time, so __getitem__ is pure tensor indexing with no disk I/O.

    Args:
        hidden_dir: Path to cached hidden states, e.g., data/cached/hidden_states/pretrained/train/
        labels_dir: Path to patch labels, e.g., data/processed/patch_labels/train/
        layer: Which ViT layer to use (0-12).
        token_mode: 'patch_only' (default), 'cls_concat', or 'cls_only'.
                     For cls modes, hidden states should include CLS token.
    """

    def __init__(
        self,
        hidden_dir: str,
        labels_dir: str,
        layer: int = 0,
        token_mode: str = "patch_only",
    ):
        self.hidden_dir = Path(hidden_dir)
        self.labels_dir = Path(labels_dir)
        self.layer = layer
        self.token_mode = token_mode

        # Find all image IDs that have both hidden states and labels
        hidden_ids = {p.stem for p in self.hidden_dir.glob("*.pt")}
        label_ids = {p.stem for p in self.labels_dir.glob("*.npy")}
        self.image_ids = sorted(hidden_ids & label_ids)

        if len(self.image_ids) == 0:
            raise FileNotFoundError(
                f"No matching files found in {hidden_dir} and {labels_dir}. "
                "Run extract_hidden_states.py and preprocess_labels.py first."
            )

        self.num_patches = 196  # 14x14 for ViT-B/16

        # Preload single layer into contiguous tensors
        feature_list = []
        label_list = []
        for image_id in self.image_ids:
            hidden_path = self.hidden_dir / f"{image_id}.pt"
            label_path = self.labels_dir / f"{image_id}.npy"

            hidden_states = torch.load(hidden_path, map_location="cpu", weights_only=True)  # (13, 196, 768)
            labels = np.load(label_path)  # (196,)

            feature_list.append(hidden_states[self.layer].float())  # (196, 768)
            label_list.append(torch.from_numpy(labels).float())     # (196,)

        self.features = torch.cat(feature_list, dim=0)   # (N*196, 768)
        self.labels = torch.cat(label_list, dim=0)       # (N*196,)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

    def get_all_labels(self) -> np.ndarray:
        """Return all labels for computing class weights and sampler weights."""
        return self.labels.numpy()


class PatchLevelDataModule:
    """Wraps HiddenStateDataset with DataLoaders and class-balanced sampling.

    Args:
        cached_dir: Base path for cached hidden states (e.g., data/cached/hidden_states)
        labels_dir: Base path for patch labels (e.g., data/processed/patch_labels)
        model_type: 'pretrained' or 'random'
        layer: Which ViT layer (0-12)
        batch_size: DataLoader batch size
        num_workers: DataLoader workers
        token_mode: 'patch_only', 'cls_concat', or 'cls_only'
    """

    def __init__(
        self,
        cached_dir: str,
        labels_dir: str,
        model_type: str = "pretrained",
        layer: int = 0,
        batch_size: int = 512,
        num_workers: int = 0,
        token_mode: str = "patch_only",
    ):
        self.cached_dir = Path(cached_dir)
        self.labels_dir = Path(labels_dir)
        self.model_type = model_type
        self.layer = layer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.token_mode = token_mode

        self._datasets = {}
        self._pos_weight = None

    def _get_dataset(self, split: str) -> HiddenStateDataset:
        if split not in self._datasets:
            self._datasets[split] = HiddenStateDataset(
                hidden_dir=str(self.cached_dir / self.model_type / split),
                labels_dir=str(self.labels_dir / split),
                layer=self.layer,
                token_mode=self.token_mode,
            )
        return self._datasets[split]

    def train_dataloader(self) -> DataLoader:
        dataset = self._get_dataset("train")

        # Compute sample weights for balanced sampling
        all_labels = dataset.get_all_labels()
        pos_count = all_labels.sum()
        neg_count = len(all_labels) - pos_count

        # Weight each sample inversely proportional to its class frequency
        weight_for_0 = 1.0 / neg_count if neg_count > 0 else 1.0
        weight_for_1 = 1.0 / pos_count if pos_count > 0 else 1.0
        sample_weights = np.where(all_labels == 1, weight_for_1, weight_for_0)
        sample_weights = torch.from_numpy(sample_weights).double()

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = self._get_dataset("val")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader:
        dataset = self._get_dataset("test")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def get_pos_weight(self) -> float:
        """Compute pos_weight for BCEWithLogitsLoss from training data."""
        if self._pos_weight is None:
            dataset = self._get_dataset("train")
            all_labels = dataset.get_all_labels()
            pos_count = all_labels.sum()
            neg_count = len(all_labels) - pos_count
            self._pos_weight = float(neg_count / pos_count) if pos_count > 0 else 1.0
        return self._pos_weight
