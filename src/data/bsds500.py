"""BSDS500 raw dataset loading."""

import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset


class BSDS500Dataset(Dataset):
    """BSDS500 dataset for boundary detection.

    Expected directory structure:
        data_dir/data/images/{train,val,test}/*.jpg
        data_dir/data/groundTruth/{train,val,test}/*.mat
    """

    def __init__(self, data_dir: str, split: str = "train", transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        self.image_dir = self.data_dir / "data" / "images" / split
        self.gt_dir = self.data_dir / "data" / "groundTruth" / split

        # Get sorted list of image IDs
        self.image_ids = sorted([
            p.stem for p in self.image_dir.glob("*.jpg")
        ])

        if len(self.image_ids) == 0:
            raise FileNotFoundError(
                f"No images found in {self.image_dir}. "
                "Run scripts/download_bsds500.py first."
            )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx) -> Tuple[Image.Image, List[np.ndarray], str]:
        image_id = self.image_ids[idx]

        # Load image
        image_path = self.image_dir / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")

        # Load ground truth boundary maps from .mat file
        gt_path = self.gt_dir / f"{image_id}.mat"
        boundary_maps = load_boundary_maps(gt_path)

        if self.transform:
            image = self.transform(image)

        return image, boundary_maps, image_id


def load_boundary_maps(mat_path: str) -> List[np.ndarray]:
    """Load boundary maps from a BSDS500 .mat file.

    Each .mat file contains 'groundTruth' with multiple annotator boundaries.
    Returns list of binary (H, W) arrays, one per annotator.
    """
    mat = loadmat(str(mat_path))
    gt = mat["groundTruth"]

    maps = []
    num_annotators = gt.shape[1]
    for i in range(num_annotators):
        # Each annotator has a 'Boundaries' field
        boundary = gt[0, i]["Boundaries"][0, 0]
        # Convert to binary (some may have soft boundaries)
        binary_map = (boundary > 0).astype(np.float32)
        maps.append(binary_map)

    return maps
