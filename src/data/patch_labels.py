"""Convert BSDS500 boundary annotations to ViT patch-level labels."""

import numpy as np
from typing import List, Optional


def aggregate_boundary_map(
    maps: List[np.ndarray],
    threshold: float = 0.5
) -> np.ndarray:
    """Aggregate multiple annotator boundary maps via majority vote.

    Args:
        maps: List of binary (H, W) boundary maps from different annotators.
        threshold: Fraction of annotators that must agree for a pixel to be boundary.

    Returns:
        Binary (H, W) consensus boundary map.
    """
    stacked = np.stack(maps, axis=0)  # (num_annotators, H, W)
    agreement = stacked.mean(axis=0)   # fraction of annotators marking boundary
    consensus = (agreement >= threshold).astype(np.float32)
    return consensus


def resize_boundary_map(
    boundary_map: np.ndarray,
    target_size: int = 224
) -> np.ndarray:
    """Resize boundary map to target_size x target_size using nearest interpolation.

    IMPORTANT: Use nearest interpolation to avoid creating soft boundaries
    from bilinear/bicubic interpolation on binary maps.
    """
    from PIL import Image

    h, w = boundary_map.shape
    img = Image.fromarray(boundary_map)
    img_resized = img.resize((target_size, target_size), Image.NEAREST)
    return np.array(img_resized)


def compute_patch_labels(
    boundary_map_224: np.ndarray,
    patch_size: int = 16,
    threshold: float = 0.0
) -> np.ndarray:
    """Convert a 224x224 boundary map to 14x14 patch-level binary labels.

    Args:
        boundary_map_224: Binary (224, 224) boundary map.
        patch_size: ViT patch size (16 for ViT-B/16).
        threshold: Fraction of boundary pixels needed in a patch.
                   0.0 means ANY boundary pixel -> label=1.

    Returns:
        Flat (196,) binary label array.
    """
    num_patches_per_side = boundary_map_224.shape[0] // patch_size  # 14
    labels = np.zeros((num_patches_per_side, num_patches_per_side), dtype=np.float32)

    for i in range(num_patches_per_side):
        for j in range(num_patches_per_side):
            patch = boundary_map_224[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size
            ]
            boundary_fraction = patch.mean()
            labels[i, j] = float(boundary_fraction > threshold)

    return labels.flatten()  # (196,)


def build_patch_labels_for_image(
    boundary_maps: List[np.ndarray],
    annotator_threshold: float = 0.5,
    boundary_threshold: float = 0.0,
    patch_size: int = 16,
    target_size: int = 224,
) -> np.ndarray:
    """Full pipeline: raw annotator maps -> flat (196,) patch labels.

    Args:
        boundary_maps: List of (H, W) binary maps from annotators.
        annotator_threshold: Majority vote threshold.
        boundary_threshold: Per-patch boundary fraction threshold.
        patch_size: ViT patch size.
        target_size: Image resize target.

    Returns:
        Flat (196,) binary label array.
    """
    consensus = aggregate_boundary_map(boundary_maps, annotator_threshold)
    resized = resize_boundary_map(consensus, target_size)
    labels = compute_patch_labels(resized, patch_size, boundary_threshold)
    return labels


def compute_class_weights(all_labels: np.ndarray) -> float:
    """Compute pos_weight for BCEWithLogitsLoss from all labels.

    Args:
        all_labels: Array of all binary labels.

    Returns:
        pos_weight: ratio of negative to positive samples.
    """
    num_positive = all_labels.sum()
    num_negative = len(all_labels) - num_positive
    if num_positive == 0:
        return 1.0
    pos_weight = num_negative / num_positive
    return float(pos_weight)
