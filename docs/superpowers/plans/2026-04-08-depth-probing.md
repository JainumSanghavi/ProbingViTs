# Depth Probing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a complete depth regression probing experiment on NYU Depth V2, parallel to the existing boundary detection pipeline, producing 52 probe checkpoints and 8 visualization figures.

**Architecture:** New `configs/depth.yaml` + `scripts/depth/` + two new `src/data/` modules + `src/data/depth_dataset.py` + `src/training/depth_trainer.py`. All existing `src/` code (ViTExtractor, LinearProbe, MLPProbe, transforms, get_device, load_config) is reused unchanged. No existing files are modified.

**Tech Stack:** PyTorch, HuggingFace `datasets` + `transformers`, matplotlib, numpy, PIL, tqdm, PyYAML

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `configs/depth.yaml` | CREATE | All depth experiment hyperparameters |
| `src/data/nyu_depth.py` | CREATE | NYU Depth V2 file-based dataset loader |
| `src/data/depth_labels.py` | CREATE | Mean depth patch aggregation + normalization |
| `src/data/depth_dataset.py` | CREATE | DepthDataModule (no WeightedRandomSampler) |
| `src/training/depth_trainer.py` | CREATE | MSELoss trainer monitoring val_mae |
| `scripts/depth/download_nyu.py` | CREATE | Download HF dataset → disk |
| `scripts/depth/preprocess_depth_labels.py` | CREATE | Build (196,) depth label .npy per image |
| `scripts/depth/extract_hidden_states_depth.py` | CREATE | Cache ViT hidden states for NYU images |
| `scripts/depth/train_probes_depth.py` | CREATE | Train 52 probes (13 layers × 2 types × 2 inits) |
| `scripts/depth/evaluate_depth.py` | CREATE | MAE/RMSE per layer on test set |
| `scripts/depth/visualize_depth.py` | CREATE | All 8 figures |
| `scripts/depth/run_depth.py` | CREATE | End-to-end orchestrator |

---

## Task 1: Config

**Files:**
- Create: `configs/depth.yaml`

- [ ] **Step 1: Write `configs/depth.yaml`**

```yaml
# Dataset
dataset:
  name: "NYUDepthV2"
  raw_dir: "data/depth_raw"
  processed_dir: "data/depth_processed"
  cached_dir: "data/depth_cached"
  splits:
    train: 675   # HF train[120:]
    val: 120     # HF train[:120]
    test: 654    # HF validation split

# Model (same as boundary experiment)
model:
  name: "google/vit-base-patch16-224-in21k"
  patch_size: 16
  image_size: 224
  num_patches: 196
  hidden_dim: 768
  num_layers: 13

# Depth labels
depth_labels:
  max_depth: 10.0       # NYU Depth V2 max depth in meters
  val_split_size: 120   # images carved from HF train for validation

# Feature extraction
extraction:
  dtype: "float16"
  batch_size: 1
  models: ["pretrained", "random"]

# Probe training
training:
  probe_types: ["linear", "mlp"]
  layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  optimizer:
    name: "adam"
    lr: 0.001
    weight_decay: 0.0001
  loss: "mse"
  batch_size: 512
  max_epochs: 100
  early_stopping:
    patience: 10
    metric: "val_mae"

# MLP probe
mlp_probe:
  hidden_dim: 256
  dropout: 0.1

# Evaluation
evaluation:
  metrics: ["mae", "rmse"]

# Results
results:
  metrics_dir: "results/depth/metrics"
  figures_dir: "results/depth/figures"
  checkpoints_dir: "results/depth/checkpoints"

# Misc
seed: 42
num_workers: 0
```

- [ ] **Step 2: Verify config loads**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.utils.config import load_config
cfg = load_config('configs/depth.yaml')
assert cfg['depth_labels']['max_depth'] == 10.0
assert cfg['training']['loss'] == 'mse'
assert len(cfg['training']['layers']) == 13
print('Config OK:', cfg['dataset']['name'])
"
```

Expected output: `Config OK: NYUDepthV2`

- [ ] **Step 3: Commit**

```bash
git add configs/depth.yaml
git commit -m "add depth experiment config"
```

---

## Task 2: NYU Dataset Loader

**Files:**
- Create: `src/data/nyu_depth.py`

- [ ] **Step 1: Write `src/data/nyu_depth.py`**

```python
"""NYU Depth V2 dataset loader from saved files on disk."""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class NYUDepthDataset(Dataset):
    """NYU Depth V2 dataset for depth regression probing.

    Expected directory structure (created by download_nyu.py):
        raw_dir/images/{train,val,test}/{00000.png,...}
        raw_dir/depth/{train,val,test}/{00000.npy,...}
    """

    def __init__(self, raw_dir: str, split: str = "train", transform=None):
        self.raw_dir = Path(raw_dir)
        self.split = split
        self.transform = transform

        self.image_dir = self.raw_dir / "images" / split
        self.depth_dir = self.raw_dir / "depth" / split

        self.image_ids = sorted([p.stem for p in self.image_dir.glob("*.png")])

        if len(self.image_ids) == 0:
            raise FileNotFoundError(
                f"No images found in {self.image_dir}. "
                "Run scripts/depth/download_nyu.py first."
            )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx) -> Tuple[Image.Image, np.ndarray, str]:
        image_id = self.image_ids[idx]
        image = Image.open(self.image_dir / f"{image_id}.png").convert("RGB")
        depth = np.load(self.depth_dir / f"{image_id}.npy")  # (H, W) float32, meters
        if self.transform:
            image = self.transform(image)
        return image, depth, image_id
```

- [ ] **Step 2: Verify (after download_nyu.py is run in Task 6)**

Run this after Task 6 completes:

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.utils.config import load_config
from src.data.nyu_depth import NYUDepthDataset
cfg = load_config('configs/depth.yaml')
ds = NYUDepthDataset(cfg['dataset']['raw_dir'], split='train')
img, depth, iid = ds[0]
print(f'Image: {img.size}, Depth: {depth.shape}, range [{depth.min():.2f}, {depth.max():.2f}]m, ID: {iid}')
assert depth.dtype.name == 'float32'
assert len(ds) == cfg['dataset']['splits']['train']
print('NYUDepthDataset OK')
"
```

Expected: `Image: (640, 480), Depth: (480, 640), range [0.xx, x.xx]m, ID: 00000`

- [ ] **Step 3: Commit**

```bash
git add src/data/nyu_depth.py
git commit -m "add NYU Depth V2 dataset loader"
```

---

## Task 3: Depth Label Computation

**Files:**
- Create: `src/data/depth_labels.py`

- [ ] **Step 1: Write `src/data/depth_labels.py`**

```python
"""Convert NYU Depth V2 depth maps to ViT patch-level depth labels."""

import numpy as np
from PIL import Image

NYU_MAX_DEPTH = 10.0


def resize_depth_map(depth_map: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Resize depth map to target_size using bilinear interpolation.

    Uses bilinear (not nearest-neighbor) because depth values are continuous.
    Compare: boundary maps use nearest-neighbor to preserve binary values.
    """
    img = Image.fromarray(depth_map)
    img_resized = img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img_resized, dtype=np.float32)


def compute_depth_patch_labels(
    depth_map_224: np.ndarray,
    patch_size: int = 16,
    max_depth: float = NYU_MAX_DEPTH,
) -> np.ndarray:
    """Compute mean depth per 16x16 patch, normalized to [0, 1].

    Args:
        depth_map_224: Float (224, 224) depth map in meters.
        patch_size: ViT patch size (16 for ViT-B/16).
        max_depth: Normalization divisor (10.0m for NYU Depth V2).

    Returns:
        Flat (196,) float32 array of normalized mean depths in [0, 1].
        Multiply by max_depth to recover meters.
    """
    num_patches = depth_map_224.shape[0] // patch_size  # 14
    labels = np.zeros((num_patches, num_patches), dtype=np.float32)

    for i in range(num_patches):
        for j in range(num_patches):
            patch = depth_map_224[
                i * patch_size:(i + 1) * patch_size,
                j * patch_size:(j + 1) * patch_size,
            ]
            labels[i, j] = patch.mean()

    labels = np.clip(labels / max_depth, 0.0, 1.0)
    return labels.flatten()  # (196,)
```

- [ ] **Step 2: Verify**

```bash
python -c "
import sys; sys.path.insert(0, '.')
import numpy as np
from src.data.depth_labels import resize_depth_map, compute_depth_patch_labels

# Synthetic depth map: uniform 5m depth
depth = np.full((480, 640), 5.0, dtype='float32')
resized = resize_depth_map(depth, 224)
assert resized.shape == (224, 224), f'Expected (224,224), got {resized.shape}'

labels = compute_depth_patch_labels(resized)
assert labels.shape == (196,), f'Expected (196,), got {labels.shape}'
assert abs(labels.mean() - 0.5) < 1e-4, f'Expected 0.5 (5m/10m), got {labels.mean()}'
print('depth_labels OK — 5m depth normalized to', labels.mean())
"
```

Expected: `depth_labels OK — 5m depth normalized to 0.5`

- [ ] **Step 3: Commit**

```bash
git add src/data/depth_labels.py
git commit -m "add depth patch label computation"
```

---

## Task 4: Depth DataModule

**Files:**
- Create: `src/data/depth_dataset.py`

- [ ] **Step 1: Write `src/data/depth_dataset.py`**

```python
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
```

- [ ] **Step 2: Verify (after Tasks 6-8 complete)**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.utils.config import load_config
from src.data.depth_dataset import DepthDataModule
cfg = load_config('configs/depth.yaml')
dm = DepthDataModule(
    cached_dir=cfg['dataset']['cached_dir'] + '/hidden_states',
    labels_dir=cfg['dataset']['processed_dir'] + '/depth_labels',
    model_type='pretrained', layer=6,
    batch_size=cfg['training']['batch_size'],
)
batch = next(iter(dm.train_dataloader()))
feats, labels = batch
print(f'Features: {feats.shape}, Labels: {labels.shape}')
assert feats.shape[1] == 768
assert labels.min() >= 0.0 and labels.max() <= 1.0
print('DepthDataModule OK')
"
```

Expected: `Features: torch.Size([512, 768]), Labels: torch.Size([512])`

- [ ] **Step 3: Commit**

```bash
git add src/data/depth_dataset.py
git commit -m "add depth data module for regression probing"
```

---

## Task 5: Depth Trainer

**Files:**
- Create: `src/training/depth_trainer.py`

- [ ] **Step 1: Write `src/training/depth_trainer.py`**

```python
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
```

- [ ] **Step 2: Verify**

```bash
python -c "
import sys; sys.path.insert(0, '.')
import torch
from src.probes.linear_probe import LinearProbe
from src.training.depth_trainer import DepthProbeTrainer

probe = LinearProbe(768)
trainer = DepthProbeTrainer(probe, torch.device('cpu'))
assert trainer.criterion.__class__.__name__ == 'MSELoss'
assert trainer.best_val_mae == float('inf')
print('DepthProbeTrainer OK')
"
```

Expected: `DepthProbeTrainer OK`

- [ ] **Step 3: Commit**

```bash
git add src/training/depth_trainer.py
git commit -m "add depth probe trainer with MSE loss and val_mae early stopping"
```

---

## Task 6: Download NYU Depth V2

**Files:**
- Create: `scripts/depth/__init__.py`
- Create: `scripts/depth/download_nyu.py`

- [ ] **Step 1: Add `datasets` to `requirements.txt`**

Append to `requirements.txt`:

```
datasets
```

The HuggingFace `datasets` library is used by `download_nyu.py` to fetch NYU Depth V2. It is not in the existing requirements.

Install it:

```bash
pip install datasets
```

- [ ] **Step 2: Create `scripts/depth/__init__.py`**

```bash
touch scripts/depth/__init__.py
```

This is required so `run_depth.py` can import submodules via `from scripts.depth.download_nyu import ...`, matching the pattern used by `scripts/__init__.py` in the existing pipeline.

- [ ] **Step 3: Write `scripts/depth/download_nyu.py`**

```python
"""Download NYU Depth V2 via HuggingFace Hub and save images/depth maps to disk."""

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import load_config


def download_nyu(config: dict):
    """Download NYU Depth V2 and save RGB images + depth maps to disk."""
    from datasets import load_dataset

    raw_dir = Path(config["dataset"]["raw_dir"])
    val_size = config["depth_labels"]["val_split_size"]

    print("Loading NYU Depth V2 from HuggingFace Hub...")
    print("(~4GB download on first run; cached by HuggingFace locally after that)")

    hf_train = load_dataset("sayakpaul/nyu_depth_v2", split="train")
    hf_test = load_dataset("sayakpaul/nyu_depth_v2", split="validation")

    splits = {
        "val": hf_train.select(range(val_size)),
        "train": hf_train.select(range(val_size, len(hf_train))),
        "test": hf_test,
    }

    for split_name, ds in splits.items():
        img_dir = raw_dir / "images" / split_name
        depth_dir = raw_dir / "depth" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving {split_name}: {len(ds)} images...")

        for idx, item in enumerate(tqdm(ds, desc=split_name)):
            image_id = f"{idx:05d}"
            img_path = img_dir / f"{image_id}.png"
            depth_path = depth_dir / f"{image_id}.npy"

            if img_path.exists() and depth_path.exists():
                continue

            item["image"].save(img_path)
            depth_array = np.array(item["depth_map"], dtype=np.float32)
            np.save(depth_path, depth_array)

    print("\nDownload complete!")
    for split_name, ds in splits.items():
        print(f"  {split_name}: {len(ds)} images")


if __name__ == "__main__":
    config = load_config("configs/depth.yaml")
    download_nyu(config)
```

- [ ] **Step 4: Run download**

```bash
python scripts/depth/download_nyu.py
```

Expected: progress bars for val (120), train (675), test (654). Check files exist:

```bash
ls data/depth_raw/images/train/ | wc -l   # should print 675
ls data/depth_raw/depth/test/ | wc -l     # should print 654
```

- [ ] **Step 5: Verify image and depth shapes**

```bash
python -c "
import numpy as np
from PIL import Image
img = Image.open('data/depth_raw/images/train/00000.png')
depth = np.load('data/depth_raw/depth/train/00000.npy')
print(f'Image: {img.size}, mode: {img.mode}')
print(f'Depth: {depth.shape}, dtype: {depth.dtype}, range: [{depth.min():.2f}, {depth.max():.2f}]m')
assert img.mode == 'RGB'
assert depth.dtype == 'float32'
print('Download verified OK')
"
```

Expected: Image 640×480 RGB, depth (480, 640) float32, range in ~[0, 10]m.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt scripts/depth/__init__.py scripts/depth/download_nyu.py
git commit -m "add NYU Depth V2 download script via HuggingFace Hub"
```

---

## Task 7: Preprocess Depth Labels

**Files:**
- Create: `scripts/depth/preprocess_depth_labels.py`

- [ ] **Step 1: Write `scripts/depth/preprocess_depth_labels.py`**

```python
"""Build patch-level depth labels for all NYU Depth V2 images."""

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import load_config
from src.data.nyu_depth import NYUDepthDataset
from src.data.depth_labels import resize_depth_map, compute_depth_patch_labels


def preprocess_all_depth_labels(config: dict):
    raw_dir = config["dataset"]["raw_dir"]
    processed_dir = Path(config["dataset"]["processed_dir"]) / "depth_labels"
    max_depth = config["depth_labels"]["max_depth"]

    all_labels = []

    for split in ["train", "val", "test"]:
        split_dir = processed_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        dataset = NYUDepthDataset(raw_dir, split=split)
        print(f"\nProcessing {split} split ({len(dataset)} images)...")

        for idx in tqdm(range(len(dataset)), desc=split):
            _, depth_map, image_id = dataset[idx]

            resized = resize_depth_map(depth_map, target_size=224)
            labels = compute_depth_patch_labels(resized, max_depth=max_depth)

            save_path = split_dir / f"{image_id}.npy"
            np.save(save_path, labels)
            all_labels.append(labels)

    all_labels = np.concatenate(all_labels)
    print(f"\n--- Depth Label Statistics (normalized [0,1]) ---")
    print(f"Total patches: {len(all_labels):,}")
    print(f"Mean: {all_labels.mean():.4f}  ({all_labels.mean()*max_depth:.2f}m)")
    print(f"Std:  {all_labels.std():.4f}")
    print(f"Min:  {all_labels.min():.4f}  Max: {all_labels.max():.4f}")


if __name__ == "__main__":
    config = load_config("configs/depth.yaml")
    preprocess_all_depth_labels(config)
```

- [ ] **Step 2: Run and verify**

```bash
python scripts/depth/preprocess_depth_labels.py
```

Expected: 3 progress bars. Final stats should show mean ~0.3–0.5 (NYU indoor scenes skew near/mid depth).

```bash
python -c "
import numpy as np
labels = np.load('data/depth_processed/depth_labels/train/00000.npy')
print(f'Shape: {labels.shape}, range: [{labels.min():.3f}, {labels.max():.3f}]')
assert labels.shape == (196,)
assert labels.min() >= 0.0 and labels.max() <= 1.0
print('Labels OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/depth/preprocess_depth_labels.py
git commit -m "add depth label preprocessing script"
```

---

## Task 8: Extract Hidden States

**Files:**
- Create: `scripts/depth/extract_hidden_states_depth.py`

- [ ] **Step 1: Write `scripts/depth/extract_hidden_states_depth.py`**

```python
"""Extract and cache ViT hidden states for all NYU Depth V2 images."""

import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import load_config
from src.utils.device import get_device
from src.models.vit_extractor import ViTExtractor
from src.data.transforms import get_vit_transform
from src.data.nyu_depth import NYUDepthDataset


def extract_hidden_states(config: dict):
    device = get_device()
    print(f"Using device: {device}")

    raw_dir = config["dataset"]["raw_dir"]
    cache_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states"
    model_name = config["model"]["name"]
    use_fp16 = config["extraction"]["dtype"] == "float16"
    transform = get_vit_transform(config["model"]["image_size"])

    for model_type in config["extraction"]["models"]:
        pretrained = (model_type == "pretrained")
        print(f"\n{'='*60}")
        print(f"Extracting with {model_type} ViT...")
        print(f"{'='*60}")

        extractor = ViTExtractor(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
        )

        for split in ["train", "val", "test"]:
            dataset = NYUDepthDataset(raw_dir, split=split)
            save_dir = cache_dir / model_type / split
            save_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{split}: {len(dataset)} images")

            for idx in tqdm(range(len(dataset)), desc=f"{model_type}/{split}"):
                image, _, image_id = dataset[idx]
                save_path = save_dir / f"{image_id}.pt"

                if save_path.exists():
                    continue

                pixel_values = transform(image)
                hidden_states = extractor.extract_single(pixel_values)  # (13, 196, 768)

                if use_fp16:
                    hidden_states = hidden_states.half()

                torch.save(hidden_states.cpu(), save_path)

        del extractor
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nExtraction complete!")
    _print_cache_stats(cache_dir)


def _print_cache_stats(cache_dir: Path):
    total_size = 0
    total_files = 0
    for pt_file in cache_dir.rglob("*.pt"):
        total_size += pt_file.stat().st_size
        total_files += 1
    print(f"\nCache: {total_files} files, {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    config = load_config("configs/depth.yaml")
    extract_hidden_states(config)
```

- [ ] **Step 2: Run extraction**

```bash
python scripts/depth/extract_hidden_states_depth.py
```

This is the longest step (~10–30 min depending on hardware). Resume-safe: already-cached files are skipped.

- [ ] **Step 3: Verify**

```bash
python -c "
import torch
hs = torch.load('data/depth_cached/hidden_states/pretrained/train/00000.pt', weights_only=True)
print(f'Shape: {hs.shape}, dtype: {hs.dtype}')
assert hs.shape == (13, 196, 768)
print('Hidden states OK')
"
```

Expected: `Shape: torch.Size([13, 196, 768]), dtype: torch.float16`

- [ ] **Step 4: Commit**

```bash
git add scripts/depth/extract_hidden_states_depth.py
git commit -m "add hidden state extraction script for NYU depth experiment"
```

---

## Task 9: Train Probes

**Files:**
- Create: `scripts/depth/train_probes_depth.py`

- [ ] **Step 1: Write `scripts/depth/train_probes_depth.py`**

```python
"""Train depth regression probes across all layers, probe types, and model types."""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import load_config
from src.utils.device import get_device
from src.data.depth_dataset import DepthDataModule
from src.probes.linear_probe import get_probe
from src.training.depth_trainer import DepthProbeTrainer


def train_all_depth_probes(config: dict):
    device = get_device()
    print(f"Using device: {device}")

    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states"
    labels_dir = Path(config["dataset"]["processed_dir"]) / "depth_labels"
    metrics_dir = Path(config["results"]["metrics_dir"])
    checkpoints_dir = Path(config["results"]["checkpoints_dir"])

    layers = config["training"]["layers"]
    probe_types = config["training"]["probe_types"]
    model_types = config["extraction"]["models"]

    all_results = {}
    total_runs = len(layers) * len(probe_types) * len(model_types)
    run_idx = 0

    for model_type in model_types:
        for probe_type in probe_types:
            for layer in layers:
                run_idx += 1
                run_name = f"{model_type}_{probe_type}_layer{layer:02d}"
                print(f"\n{'='*60}")
                print(f"[{run_idx}/{total_runs}] {run_name}")
                print(f"{'='*60}")

                start_time = time.time()

                data_module = DepthDataModule(
                    cached_dir=str(cached_dir),
                    labels_dir=str(labels_dir),
                    model_type=model_type,
                    layer=layer,
                    batch_size=config["training"]["batch_size"],
                    num_workers=config["num_workers"],
                )

                probe_kwargs = {}
                if probe_type == "mlp":
                    probe_kwargs = {
                        "hidden_dim": config["mlp_probe"]["hidden_dim"],
                        "dropout": config["mlp_probe"]["dropout"],
                    }

                probe = get_probe(
                    probe_type,
                    input_dim=config["model"]["hidden_dim"],
                    **probe_kwargs,
                )

                num_params = sum(p.numel() for p in probe.parameters())
                print(f"  Probe params: {num_params:,}")

                checkpoint_path = str(checkpoints_dir / f"{run_name}.pt")
                trainer = DepthProbeTrainer(
                    model=probe,
                    device=device,
                    lr=config["training"]["optimizer"]["lr"],
                    weight_decay=config["training"]["optimizer"]["weight_decay"],
                    patience=config["training"]["early_stopping"]["patience"],
                    max_epochs=config["training"]["max_epochs"],
                )

                results = trainer.train(
                    data_module.train_dataloader(),
                    data_module.val_dataloader(),
                    checkpoint_path=checkpoint_path,
                )

                elapsed = time.time() - start_time
                print(f"  Training time: {elapsed:.1f}s")
                print(f"  Best val MAE: {results['best_val_mae']:.4f}")

                all_results[run_name] = {
                    "model_type": model_type,
                    "probe_type": probe_type,
                    "layer": layer,
                    "num_params": num_params,
                    "best_val_mae": results["best_val_mae"],
                    "epochs_trained": results["epochs_trained"],
                    "training_time": elapsed,
                }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    results_path = metrics_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {results_path}")
    _print_summary(all_results)


def _print_summary(results: dict):
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"{'Run':<40} {'Val MAE':>8} {'Epochs':>8} {'Time':>8}")
    print("-" * 80)
    for name, r in sorted(results.items()):
        print(
            f"{name:<40} "
            f"{r['best_val_mae']:>8.4f} "
            f"{r['epochs_trained']:>8d} "
            f"{r['training_time']:>7.1f}s"
        )


if __name__ == "__main__":
    config = load_config("configs/depth.yaml")
    torch.manual_seed(config["seed"])
    train_all_depth_probes(config)
```

- [ ] **Step 2: Run training**

```bash
python scripts/depth/train_probes_depth.py
```

52 runs total. Expect val MAE to decrease from ~0.15–0.20 at early layers to a minimum somewhere in layers 6–10 for the pretrained ViT.

- [ ] **Step 3: Verify checkpoints**

```bash
ls results/depth/checkpoints/ | wc -l   # should print 52
python -c "
import json
with open('results/depth/metrics/training_results.json') as f:
    r = json.load(f)
print(f'{len(r)} runs recorded')
best = min(r.items(), key=lambda x: x[1]['best_val_mae'])
print(f'Best: {best[0]}, val MAE={best[1][\"best_val_mae\"]:.4f}')
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/depth/train_probes_depth.py
git commit -m "add depth probe training script"
```

---

## Task 10: Evaluate Probes

**Files:**
- Create: `scripts/depth/evaluate_depth.py`

- [ ] **Step 1: Write `scripts/depth/evaluate_depth.py`**

```python
"""Evaluate all trained depth probes on the test set."""

import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import load_config
from src.utils.device import get_device
from src.data.depth_dataset import DepthDataModule
from src.probes.linear_probe import get_probe
from src.training.depth_trainer import DepthProbeTrainer


def evaluate_all_depth_probes(config: dict, device: torch.device) -> dict:
    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states"
    labels_dir = Path(config["dataset"]["processed_dir"]) / "depth_labels"
    checkpoints_dir = Path(config["results"]["checkpoints_dir"])

    layers = config["training"]["layers"]
    probe_types = config["training"]["probe_types"]
    model_types = config["extraction"]["models"]

    results = {}

    for model_type in model_types:
        results[model_type] = {}
        for probe_type in probe_types:
            results[model_type][probe_type] = {}

            for layer in tqdm(layers, desc=f"{model_type}/{probe_type}"):
                run_name = f"{model_type}_{probe_type}_layer{layer:02d}"
                checkpoint_path = checkpoints_dir / f"{run_name}.pt"

                if not checkpoint_path.exists():
                    print(f"  Warning: checkpoint not found for {run_name}")
                    continue

                data_module = DepthDataModule(
                    cached_dir=str(cached_dir),
                    labels_dir=str(labels_dir),
                    model_type=model_type,
                    layer=layer,
                    batch_size=config["training"]["batch_size"],
                    num_workers=config["num_workers"],
                )

                probe_kwargs = {}
                if probe_type == "mlp":
                    probe_kwargs = {
                        "hidden_dim": config["mlp_probe"]["hidden_dim"],
                        "dropout": config["mlp_probe"]["dropout"],
                    }

                probe = get_probe(probe_type, input_dim=config["model"]["hidden_dim"], **probe_kwargs)
                state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                probe.load_state_dict(state_dict)

                trainer = DepthProbeTrainer(model=probe, device=device)
                metrics = trainer.evaluate_test(data_module.test_dataloader())

                results[model_type][probe_type][layer] = metrics

    return results


def main():
    config = load_config("configs/depth.yaml")
    device = get_device()
    print(f"Using device: {device}")

    print("\nEvaluating all depth probes on test set...")
    results = evaluate_all_depth_probes(config, device)

    metrics_dir = Path(config["results"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for mt in results:
        serializable[mt] = {}
        for pt in results[mt]:
            serializable[mt][pt] = {str(k): v for k, v in results[mt][pt].items()}

    with open(metrics_dir / "depth_test_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {metrics_dir / 'depth_test_results.json'}")

    print("\nBest test MAE per model/probe type (pretrained ViT):")
    for pt in results.get("pretrained", {}):
        if results["pretrained"][pt]:
            best_layer = min(results["pretrained"][pt], key=lambda l: results["pretrained"][pt][l]["mae"])
            best_mae = results["pretrained"][pt][best_layer]["mae"]
            print(f"  {pt}: layer {best_layer}, MAE={best_mae:.4f} ({best_mae*10:.3f}m)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run evaluation**

```bash
python scripts/depth/evaluate_depth.py
```

- [ ] **Step 3: Verify results file**

```bash
python -c "
import json
with open('results/depth/metrics/depth_test_results.json') as f:
    r = json.load(f)
# Check structure
assert 'pretrained' in r and 'random' in r
assert 'linear' in r['pretrained'] and 'mlp' in r['pretrained']
assert len(r['pretrained']['linear']) == 13
print('Results structure OK')
print('Sample — pretrained/linear/layer 6:', r['pretrained']['linear']['6'])
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/depth/evaluate_depth.py
git commit -m "add depth probe evaluation script"
```

---

## Task 11: Visualizations

**Files:**
- Create: `scripts/depth/visualize_depth.py`

- [ ] **Step 1: Write `scripts/depth/visualize_depth.py`**

```python
"""Generate all depth probing visualization figures (8 total)."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import load_config
from src.utils.device import get_device
from src.probes.linear_probe import get_probe
from src.training.depth_trainer import DepthProbeTrainer
from src.data.depth_dataset import DepthDataModule


def _load_results(config: dict):
    """Load test results JSON, converting str keys to int."""
    path = Path(config["results"]["metrics_dir"]) / "depth_test_results.json"
    with open(path) as f:
        raw = json.load(f)
    results = {}
    for mt in raw:
        results[mt] = {}
        for pt in raw[mt]:
            results[mt][pt] = {int(k): v for k, v in raw[mt][pt].items()}
    return results


def _get_best_layer(results: dict, model_type: str = "pretrained", probe_type: str = "linear") -> int:
    """Return layer with lowest test MAE for given model/probe type."""
    layer_metrics = results[model_type][probe_type]
    return min(layer_metrics, key=lambda l: layer_metrics[l]["mae"])


def _run_inference(config, device, model_type, probe_type, layer, split="test"):
    """Load checkpoint and collect all predictions + labels for a split."""
    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states"
    labels_dir = Path(config["dataset"]["processed_dir"]) / "depth_labels"
    checkpoints_dir = Path(config["results"]["checkpoints_dir"])

    run_name = f"{model_type}_{probe_type}_layer{layer:02d}"
    probe_kwargs = {}
    if probe_type == "mlp":
        probe_kwargs = {
            "hidden_dim": config["mlp_probe"]["hidden_dim"],
            "dropout": config["mlp_probe"]["dropout"],
        }

    probe = get_probe(probe_type, input_dim=config["model"]["hidden_dim"], **probe_kwargs)
    state_dict = torch.load(checkpoints_dir / f"{run_name}.pt", map_location="cpu", weights_only=True)
    probe.load_state_dict(state_dict)

    dm = DepthDataModule(
        cached_dir=str(cached_dir),
        labels_dir=str(labels_dir),
        model_type=model_type,
        layer=layer,
        batch_size=config["training"]["batch_size"],
        num_workers=config["num_workers"],
    )
    loader = dm.test_dataloader() if split == "test" else dm.val_dataloader()

    trainer = DepthProbeTrainer(model=probe, device=device)
    trainer.model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for feats, lbls in loader:
            preds = trainer.model(feats.to(device)).cpu()
            all_preds.append(preds)
            all_labels.append(lbls)

    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


# ── Figure 1: Layerwise MAE ──────────────────────────────────────────────────

def plot_layerwise_mae(results: dict, figures_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {
        ("pretrained", "linear"): {"color": "blue", "marker": "o", "linestyle": "-"},
        ("pretrained", "mlp"):    {"color": "blue", "marker": "s", "linestyle": "--"},
        ("random",     "linear"): {"color": "red",  "marker": "o", "linestyle": "-"},
        ("random",     "mlp"):    {"color": "red",  "marker": "s", "linestyle": "--"},
    }
    for model_type in results:
        for probe_type in results[model_type]:
            layer_metrics = results[model_type][probe_type]
            layers = sorted(layer_metrics.keys())
            maes = [layer_metrics[l]["mae"] for l in layers]
            style = styles.get((model_type, probe_type), {})
            ax.plot(layers, maes, label=f"{model_type} {probe_type}", linewidth=2, markersize=6, **style)

    ax.set_xlabel("ViT Layer", fontsize=12)
    ax.set_ylabel("MAE (normalized depth [0,1])", fontsize=12)
    ax.set_title("Depth Probe: MAE by Layer", fontsize=14)
    ax.set_xticks(range(13))
    ax.set_xticklabels([f"L{i}" for i in range(13)])
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "layerwise_mae.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: layerwise_mae.png")


# ── Figure 2: Pretrained vs Random ──────────────────────────────────────────

def plot_pretrained_vs_random(results: dict, figures_dir: Path):
    if "pretrained" not in results or "random" not in results:
        print("  Skipping pretrained_vs_random_depth.png (missing model types)")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_type, color in [("pretrained", "blue"), ("random", "red")]:
        if "linear" in results[model_type]:
            lm = results[model_type]["linear"]
            layers = sorted(lm.keys())
            ax.plot(layers, [lm[l]["mae"] for l in layers], color=color,
                    marker="o", linewidth=2, label=f"{model_type} (linear)")
    ax.set_xlabel("ViT Layer", fontsize=12)
    ax.set_ylabel("MAE (normalized depth [0,1])", fontsize=12)
    ax.set_title("Depth Probe: Pretrained vs Random ViT", fontsize=14)
    ax.set_xticks(range(13))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "pretrained_vs_random_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: pretrained_vs_random_depth.png")


# ── Figure 3: Cross-Task Layerwise Comparison ────────────────────────────────

def plot_cross_task(depth_results: dict, figures_dir: Path):
    boundary_path = Path("results/metrics/test_results.json")
    if not boundary_path.exists():
        print("  Skipping cross_task_layerwise.png (boundary results not found at results/metrics/test_results.json)")
        return

    with open(boundary_path) as f:
        boundary_raw = json.load(f)
    boundary_results = {}
    for mt in boundary_raw:
        boundary_results[mt] = {}
        for pt in boundary_raw[mt]:
            boundary_results[mt][pt] = {int(k): v for k, v in boundary_raw[mt][pt].items()}

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Boundary F1 on left axis (pretrained linear)
    if "pretrained" in boundary_results and "linear" in boundary_results["pretrained"]:
        lm = boundary_results["pretrained"]["linear"]
        layers = sorted(lm.keys())
        ax1.plot(layers, [lm[l]["f1"] for l in layers],
                 color="green", marker="o", linewidth=2, label="Boundary F1 (pretrained linear)")

    # Depth MAE on right axis (pretrained linear), inverted so "up" = better
    if "pretrained" in depth_results and "linear" in depth_results["pretrained"]:
        lm = depth_results["pretrained"]["linear"]
        layers = sorted(lm.keys())
        ax2.plot(layers, [lm[l]["mae"] for l in layers],
                 color="purple", marker="s", linewidth=2, linestyle="--", label="Depth MAE (pretrained linear)")

    ax1.set_xlabel("ViT Layer", fontsize=12)
    ax1.set_ylabel("Boundary F1 (higher = better)", fontsize=12, color="green")
    ax2.set_ylabel("Depth MAE (lower = better)", fontsize=12, color="purple")
    ax1.set_xticks(range(13))
    ax1.set_xticklabels([f"L{i}" for i in range(13)])
    ax1.tick_params(axis="y", labelcolor="green")
    ax2.tick_params(axis="y", labelcolor="purple")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)
    ax1.set_title("Boundary vs Depth Encoding Across ViT Layers", fontsize=14)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(figures_dir / "cross_task_layerwise.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: cross_task_layerwise.png")


# ── Figure 4: Qualitative (original / GT depth / predicted depth) ────────────

def plot_qualitative(config: dict, device: torch.device, results: dict, figures_dir: Path):
    best_layer = _get_best_layer(results)
    raw_dir = Path(config["dataset"]["raw_dir"])
    labels_dir = Path(config["dataset"]["processed_dir"]) / "depth_labels" / "test"
    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states" / "pretrained" / "test"

    image_ids = sorted([p.stem for p in (raw_dir / "images" / "test").glob("*.png")])[:5]

    run_name = f"pretrained_linear_layer{best_layer:02d}"
    probe = get_probe("linear", input_dim=config["model"]["hidden_dim"])
    state = torch.load(Path(config["results"]["checkpoints_dir"]) / f"{run_name}.pt",
                       map_location="cpu", weights_only=True)
    probe.load_state_dict(state)
    probe.eval()

    fig, axes = plt.subplots(5, 3, figsize=(12, 20))

    for row, image_id in enumerate(image_ids):
        image = Image.open(raw_dir / "images" / "test" / f"{image_id}.png").convert("RGB")
        gt_labels = np.load(labels_dir / f"{image_id}.npy").reshape(14, 14)
        hidden = torch.load(cached_dir / f"{image_id}.pt", weights_only=True)
        feats = hidden[best_layer].float()  # (196, 768)

        with torch.no_grad():
            pred = probe(feats).numpy().reshape(14, 14)

        # Upsample 14x14 → 224x224
        gt_up = np.array(Image.fromarray(gt_labels).resize((224, 224), Image.BILINEAR))
        pred_up = np.array(Image.fromarray(pred).resize((224, 224), Image.BILINEAR))

        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Original" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_up, cmap="plasma", vmin=0, vmax=1)
        axes[row, 1].set_title(f"GT Depth (norm.)" if row == 0 else "")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_up, cmap="plasma", vmin=0, vmax=1)
        axes[row, 2].set_title(f"Predicted (L{best_layer})" if row == 0 else "")
        axes[row, 2].axis("off")

    plt.suptitle(f"Depth Probe Qualitative — Best Layer L{best_layer}", fontsize=14)
    plt.tight_layout()
    fig.savefig(figures_dir / f"qualitative_depth_layer{best_layer:02d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: qualitative_depth_layer{best_layer:02d}.png")


# ── Figure 5: Heatmap Grid (layers 0,3,6,9,12) ───────────────────────────────

def plot_heatmap_grid(config: dict, device: torch.device, figures_dir: Path):
    target_layers = [0, 3, 6, 9, 12]
    raw_dir = Path(config["dataset"]["raw_dir"])
    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states" / "pretrained" / "test"
    checkpoints_dir = Path(config["results"]["checkpoints_dir"])

    image_ids = sorted([p.stem for p in (raw_dir / "images" / "test").glob("*.png")])[:5]

    probes = {}
    for layer in target_layers:
        run_name = f"pretrained_linear_layer{layer:02d}"
        p = get_probe("linear", input_dim=config["model"]["hidden_dim"])
        state = torch.load(checkpoints_dir / f"{run_name}.pt", map_location="cpu", weights_only=True)
        p.load_state_dict(state)
        p.eval()
        probes[layer] = p

    n_rows, n_cols = len(image_ids), len(target_layers)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

    for row, image_id in enumerate(image_ids):
        hidden = torch.load(cached_dir / f"{image_id}.pt", weights_only=True)
        image = Image.open(raw_dir / "images" / "test" / f"{image_id}.png").convert("RGB")
        img_arr = np.array(image.resize((224, 224)))

        for col, layer in enumerate(target_layers):
            feats = hidden[layer].float()  # (196, 768)
            with torch.no_grad():
                pred = probes[layer](feats).numpy().reshape(14, 14)
            pred_up = np.array(Image.fromarray(pred).resize((224, 224), Image.BILINEAR))

            ax = axes[row, col]
            ax.imshow(img_arr, alpha=0.5)
            im = ax.imshow(pred_up, cmap="plasma", alpha=0.6, vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Layer {layer}", fontsize=10)

    plt.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02,
                 label="Predicted depth (normalized)")
    plt.suptitle("Depth Probe Heatmap Grid — Pretrained Linear", fontsize=13)
    plt.tight_layout()
    fig.savefig(figures_dir / "depth_heatmap_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: depth_heatmap_grid.png")


# ── Figure 6: Predicted vs Actual Scatter ────────────────────────────────────

def plot_scatter(config: dict, device: torch.device, results: dict, figures_dir: Path):
    best_layer = _get_best_layer(results)
    preds, labels = _run_inference(config, device, "pretrained", "linear", best_layer)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(labels, preds, alpha=0.03, s=1, color="steelblue")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual depth (normalized)", fontsize=12)
    ax.set_ylabel("Predicted depth (normalized)", fontsize=12)
    ax.set_title(f"Depth Probe Calibration — Layer {best_layer} (linear, pretrained)", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    mae = float(np.abs(preds - labels).mean())
    ax.text(0.05, 0.92, f"MAE = {mae:.4f} ({mae*10:.3f}m)", transform=ax.transAxes, fontsize=11)
    plt.tight_layout()
    fig.savefig(figures_dir / "depth_scatter_best_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: depth_scatter_best_layer.png")


# ── Figure 7: MAE by Depth Bin ───────────────────────────────────────────────

def plot_mae_by_bin(config: dict, device: torch.device, results: dict, figures_dir: Path):
    best_layer = _get_best_layer(results)
    preds, labels = _run_inference(config, device, "pretrained", "linear", best_layer)

    bins = [(0.0, 0.3, "Near\n(0–3m)"), (0.3, 0.6, "Mid\n(3–6m)"), (0.6, 1.0, "Far\n(6–10m)")]
    bin_maes = []
    bin_labels = []
    bin_counts = []

    for lo, hi, label in bins:
        mask = (labels >= lo) & (labels < hi)
        if mask.sum() > 0:
            bin_maes.append(float(np.abs(preds[mask] - labels[mask]).mean()))
            bin_counts.append(int(mask.sum()))
        else:
            bin_maes.append(0.0)
            bin_counts.append(0)
        bin_labels.append(label)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(bin_labels, bin_maes, color=["#4c72b0", "#dd8452", "#55a868"], edgecolor="black", linewidth=0.8)
    for bar, count in zip(bars, bin_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"n={count:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("MAE (normalized depth [0,1])", fontsize=12)
    ax.set_title(f"Depth Probe MAE by Depth Range — Layer {best_layer} (linear, pretrained)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "mae_by_depth_bin.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: mae_by_depth_bin.png")


# ── Figure 8: Patch Error Map ─────────────────────────────────────────────────

def plot_patch_error_map(config: dict, device: torch.device, results: dict, figures_dir: Path):
    best_layer = _get_best_layer(results)
    raw_dir = Path(config["dataset"]["raw_dir"])
    labels_dir = Path(config["dataset"]["processed_dir"]) / "depth_labels" / "test"
    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states" / "pretrained" / "test"
    checkpoints_dir = Path(config["results"]["checkpoints_dir"])

    image_ids = sorted([p.stem for p in (raw_dir / "images" / "test").glob("*.png")])[:5]

    run_name = f"pretrained_linear_layer{best_layer:02d}"
    probe = get_probe("linear", input_dim=config["model"]["hidden_dim"])
    state = torch.load(checkpoints_dir / f"{run_name}.pt", map_location="cpu", weights_only=True)
    probe.load_state_dict(state)
    probe.eval()

    fig, axes = plt.subplots(5, 3, figsize=(12, 20))

    for row, image_id in enumerate(image_ids):
        image = Image.open(raw_dir / "images" / "test" / f"{image_id}.png").convert("RGB")
        gt = np.load(labels_dir / f"{image_id}.npy").reshape(14, 14)
        hidden = torch.load(cached_dir / f"{image_id}.pt", weights_only=True)

        with torch.no_grad():
            pred = probe(hidden[best_layer].float()).numpy().reshape(14, 14)

        error = np.abs(pred - gt)
        error_up = np.array(Image.fromarray(error).resize((224, 224), Image.BILINEAR))
        img_224 = np.array(image.resize((224, 224)))

        axes[row, 0].imshow(img_224)
        axes[row, 0].set_title("Original" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(np.array(Image.fromarray(gt).resize((224, 224), Image.BILINEAR)),
                            cmap="plasma", vmin=0, vmax=1)
        axes[row, 1].set_title("GT Depth" if row == 0 else "")
        axes[row, 1].axis("off")

        im = axes[row, 2].imshow(error_up, cmap="hot", vmin=0, vmax=0.3)
        axes[row, 2].set_title(f"Abs Error (L{best_layer})" if row == 0 else "")
        axes[row, 2].axis("off")

    plt.colorbar(im, ax=axes[:, 2], orientation="vertical", fraction=0.05, pad=0.02,
                 label="|pred - GT| (normalized)")
    plt.suptitle(f"Per-Patch Depth Error Map — Layer {best_layer}", fontsize=14)
    plt.tight_layout()
    fig.savefig(figures_dir / "patch_error_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: patch_error_map.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    config = load_config("configs/depth.yaml")
    device = get_device()
    figures_dir = Path(config["results"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = _load_results(config)

    print("Figure 1: Layerwise MAE...")
    plot_layerwise_mae(results, figures_dir)

    print("Figure 2: Pretrained vs Random...")
    plot_pretrained_vs_random(results, figures_dir)

    print("Figure 3: Cross-task comparison...")
    plot_cross_task(results, figures_dir)

    print("Figure 4: Qualitative...")
    plot_qualitative(config, device, results, figures_dir)

    print("Figure 5: Heatmap grid...")
    plot_heatmap_grid(config, device, figures_dir)

    print("Figure 6: Scatter plot...")
    plot_scatter(config, device, results, figures_dir)

    print("Figure 7: MAE by depth bin...")
    plot_mae_by_bin(config, device, results, figures_dir)

    print("Figure 8: Patch error map...")
    plot_patch_error_map(config, device, results, figures_dir)

    print(f"\nAll figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run visualization**

```bash
python scripts/depth/visualize_depth.py
```

- [ ] **Step 3: Verify all 8 figures exist**

```bash
ls results/depth/figures/
```

Expected files: `layerwise_mae.png`, `pretrained_vs_random_depth.png`, `cross_task_layerwise.png`, `qualitative_depth_layer{N:02d}.png`, `depth_heatmap_grid.png`, `depth_scatter_best_layer.png`, `mae_by_depth_bin.png`, `patch_error_map.png`

- [ ] **Step 4: Commit**

```bash
git add scripts/depth/visualize_depth.py
git commit -m "add depth visualization script with 8 figures"
```

---

## Task 12: Orchestrator

**Files:**
- Create: `scripts/depth/run_depth.py`

- [ ] **Step 1: Write `scripts/depth/run_depth.py`**

```python
"""End-to-end depth probing pipeline: download → preprocess → extract → train → evaluate → visualize."""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run full depth probing pipeline")
    parser.add_argument("--config", default="configs/depth.yaml", help="Config file path")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip label preprocessing")
    parser.add_argument("--skip-extract", action="store_true", help="Skip feature extraction")
    parser.add_argument("--skip-train", action="store_true", help="Skip probe training")
    args = parser.parse_args()

    config = load_config(args.config)
    total_start = time.time()

    if not args.skip_download:
        print("\n" + "=" * 60)
        print("STEP 1: Downloading NYU Depth V2")
        print("=" * 60)
        from scripts.depth.download_nyu import download_nyu
        download_nyu(config)

    if not args.skip_preprocess:
        print("\n" + "=" * 60)
        print("STEP 2: Preprocessing depth labels")
        print("=" * 60)
        from scripts.depth.preprocess_depth_labels import preprocess_all_depth_labels
        preprocess_all_depth_labels(config)

    if not args.skip_extract:
        print("\n" + "=" * 60)
        print("STEP 3: Extracting ViT hidden states")
        print("=" * 60)
        from scripts.depth.extract_hidden_states_depth import extract_hidden_states
        extract_hidden_states(config)

    if not args.skip_train:
        print("\n" + "=" * 60)
        print("STEP 4: Training depth probes")
        print("=" * 60)
        torch.manual_seed(config["seed"])
        from scripts.depth.train_probes_depth import train_all_depth_probes
        train_all_depth_probes(config)

    print("\n" + "=" * 60)
    print("STEP 5: Evaluating probes")
    print("=" * 60)
    from scripts.depth.evaluate_depth import main as evaluate_main
    evaluate_main()

    print("\n" + "=" * 60)
    print("STEP 6: Generating visualizations")
    print("=" * 60)
    from scripts.depth.visualize_depth import main as visualize_main
    visualize_main()

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"DEPTH PIPELINE COMPLETE in {elapsed / 60:.1f} minutes")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs with all skips**

```bash
python scripts/depth/run_depth.py --skip-download --skip-preprocess --skip-extract --skip-train
```

Expected: Runs steps 5 (evaluate) and 6 (visualize) without error.

- [ ] **Step 3: Commit**

```bash
git add scripts/depth/run_depth.py
git commit -m "add end-to-end depth probing orchestrator"
```

---

## Running the Full Pipeline

```bash
# Full run from scratch:
python scripts/depth/run_depth.py

# Skip completed steps (resume after interruption):
python scripts/depth/run_depth.py --skip-download --skip-preprocess
python scripts/depth/run_depth.py --skip-download --skip-preprocess --skip-extract
python scripts/depth/run_depth.py --skip-download --skip-preprocess --skip-extract --skip-train
```
