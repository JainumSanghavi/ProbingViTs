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
