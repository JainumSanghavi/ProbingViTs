"""DataModule for depth regression probing.

Wraps HiddenStateDataset without WeightedRandomSampler (regression, no class imbalance).
"""

from pathlib import Path
from torch.utils.data import DataLoader

from src.data.hidden_state_dataset import HiddenStateDataset


class DepthDataModule:
    """Wraps HiddenStateDataset with DataLoaders for depth regression.

    Unlike PatchLevelDataModule (boundary), no weighted sampling is applied
    since depth is a continuous regression target with no class imbalance.

    Args:
        cached_dir: Base path for cached hidden states (e.g., data/depth_cached/hidden_states)
        labels_dir: Base path for depth labels (e.g., data/depth_processed/depth_labels)
        model_type: 'pretrained' or 'random'
        layer: Which ViT layer (0-12)
        batch_size: DataLoader batch size
        num_workers: DataLoader workers
    """

    def __init__(
        self,
        cached_dir: str,
        labels_dir: str,
        model_type: str = "pretrained",
        layer: int = 0,
        batch_size: int = 512,
        num_workers: int = 0,
    ):
        self.cached_dir = Path(cached_dir)
        self.labels_dir = Path(labels_dir)
        self.model_type = model_type
        self.layer = layer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._datasets = {}

    def _get_dataset(self, split: str) -> HiddenStateDataset:
        if split not in self._datasets:
            self._datasets[split] = HiddenStateDataset(
                hidden_dir=str(self.cached_dir / self.model_type / split),
                labels_dir=str(self.labels_dir / split),
                layer=self.layer,
            )
        return self._datasets[split]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._get_dataset("train"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._get_dataset("val"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._get_dataset("test"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
