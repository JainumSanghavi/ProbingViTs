"""Build patch-level boundary labels for all BSDS500 images."""

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import load_config
from src.data.bsds500 import BSDS500Dataset
from src.data.patch_labels import build_patch_labels_for_image, compute_class_weights


def preprocess_all_labels(config: dict):
    """Build and save patch labels for all images."""
    raw_dir = config["dataset"]["raw_dir"]
    processed_dir = Path(config["dataset"]["processed_dir"]) / "patch_labels"
    annotator_thresh = config["patch_labels"]["annotator_threshold"]
    boundary_thresh = config["patch_labels"]["boundary_threshold"]

    all_labels = []

    for split in ["train", "val", "test"]:
        split_dir = processed_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        dataset = BSDS500Dataset(raw_dir, split=split)
        print(f"\nProcessing {split} split ({len(dataset)} images)...")

        for idx in tqdm(range(len(dataset))):
            image, boundary_maps, image_id = dataset[idx]

            labels = build_patch_labels_for_image(
                boundary_maps,
                annotator_threshold=annotator_thresh,
                boundary_threshold=boundary_thresh,
            )

            save_path = split_dir / f"{image_id}.npy"
            np.save(save_path, labels)
            all_labels.append(labels)

    # Print class balance statistics
    all_labels = np.concatenate(all_labels)
    pos_ratio = all_labels.mean()
    pos_weight = compute_class_weights(all_labels)

    print(f"\n--- Class Balance Statistics ---")
    print(f"Total patches: {len(all_labels)}")
    print(f"Boundary patches: {all_labels.sum():.0f} ({pos_ratio*100:.1f}%)")
    print(f"Non-boundary patches: {(1-pos_ratio)*len(all_labels):.0f} ({(1-pos_ratio)*100:.1f}%)")
    print(f"Computed pos_weight: {pos_weight:.2f}")


if __name__ == "__main__":
    config = load_config()
    preprocess_all_labels(config)
