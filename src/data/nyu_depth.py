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
