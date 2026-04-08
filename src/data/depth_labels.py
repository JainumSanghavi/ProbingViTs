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
