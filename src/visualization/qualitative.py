"""Qualitative visualization: boundary prediction overlays on images."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def visualize_predictions(
    config: dict,
    device: torch.device,
    layer: int = 6,
    num_images: int = 5,
    save_dir: str = "results/figures",
):
    """Overlay predicted boundary patches on original images.

    Shows original image, ground truth patches, and predicted patches side by side.
    """
    from src.probes.linear_probe import get_probe

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(config["dataset"]["raw_dir"])
    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states"
    labels_dir = Path(config["dataset"]["processed_dir"]) / "patch_labels"
    checkpoints_dir = Path(config["results"]["checkpoints_dir"])

    # Load probe
    run_name = f"pretrained_linear_layer{layer:02d}"
    checkpoint_path = checkpoints_dir / f"{run_name}.pt"
    if not checkpoint_path.exists():
        print(f"  No checkpoint for {run_name}")
        return

    probe = get_probe("linear", input_dim=config["model"]["hidden_dim"])
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    probe.load_state_dict(state_dict)
    probe.to(device)
    probe.eval()

    # Get test images
    test_img_dir = raw_dir / "data" / "images" / "test"
    test_hidden_dir = cached_dir / "pretrained" / "test"
    test_labels_dir = labels_dir / "test"

    image_ids = sorted([p.stem for p in test_hidden_dir.glob("*.pt")])[:num_images]

    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
    if num_images == 1:
        axes = axes[np.newaxis, :]

    for idx, image_id in enumerate(image_ids):
        # Load original image
        img = Image.open(test_img_dir / f"{image_id}.jpg").convert("RGB")
        img_resized = img.resize((224, 224))

        # Load ground truth labels
        gt_labels = np.load(test_labels_dir / f"{image_id}.npy")
        gt_grid = gt_labels.reshape(14, 14)

        # Get predictions
        hidden = torch.load(test_hidden_dir / f"{image_id}.pt", map_location="cpu", weights_only=True)
        features = hidden[layer].float()  # (196, 768)
        with torch.no_grad():
            logits = probe(features.unsqueeze(0).to(device))  # (1, 196)
        pred_labels = (logits.cpu().squeeze().numpy() > 0).astype(np.float32)
        pred_grid = pred_labels.reshape(14, 14)

        # Plot
        axes[idx, 0].imshow(img_resized)
        axes[idx, 0].set_title(f"Original ({image_id})")
        axes[idx, 0].axis("off")

        # Ground truth overlay
        axes[idx, 1].imshow(img_resized)
        _overlay_patches(axes[idx, 1], gt_grid, color="red", alpha=0.35)
        axes[idx, 1].set_title("Ground Truth Boundaries")
        axes[idx, 1].axis("off")

        # Prediction overlay
        axes[idx, 2].imshow(img_resized)
        _overlay_patches(axes[idx, 2], pred_grid, color="blue", alpha=0.35)
        axes[idx, 2].set_title(f"Predicted (Layer {layer})")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    fig.savefig(save_dir / f"qualitative_layer{layer:02d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: qualitative_layer{layer:02d}.png")


def visualize_heatmap_grid(
    config: dict,
    device: torch.device,
    layers: list[int] | None = None,
    num_images: int = 5,
    save_dir: str = "results/figures",
):
    """Show boundary probability heatmaps across ViT layers for each test image.

    Creates a grid: rows = images, columns = Original | GT | Layer 0 | 3 | 6 | 9 | 12.
    Each layer cell shows the probe's sigmoid probabilities as a heatmap overlaid on the image.
    """
    from src.probes.linear_probe import get_probe

    if layers is None:
        layers = [0, 3, 6, 9, 12]

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = Path(config["dataset"]["raw_dir"])
    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states"
    labels_dir = Path(config["dataset"]["processed_dir"]) / "patch_labels"
    checkpoints_dir = Path(config["results"]["checkpoints_dir"])
    hidden_dim = config["model"]["hidden_dim"]

    # Load probes for each layer
    probes = {}
    for layer in layers:
        run_name = f"pretrained_linear_layer{layer:02d}"
        checkpoint_path = checkpoints_dir / f"{run_name}.pt"
        if not checkpoint_path.exists():
            print(f"  No checkpoint for {run_name}, skipping layer {layer}")
            continue
        probe = get_probe("linear", input_dim=hidden_dim)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        probe.load_state_dict(state_dict)
        probe.to(device)
        probe.eval()
        probes[layer] = probe

    if not probes:
        print("  No probe checkpoints found, skipping heatmap grid.")
        return

    available_layers = sorted(probes.keys())

    # Get test images
    test_img_dir = raw_dir / "data" / "images" / "test"
    test_hidden_dir = cached_dir / "pretrained" / "test"
    test_labels_dir = labels_dir / "test"

    image_ids = sorted([p.stem for p in test_hidden_dir.glob("*.pt")])[:num_images]
    num_cols = 2 + len(available_layers)  # Original + GT + one per layer

    fig, axes = plt.subplots(
        len(image_ids), num_cols, figsize=(3 * num_cols, 3 * len(image_ids))
    )
    if len(image_ids) == 1:
        axes = axes[np.newaxis, :]

    for idx, image_id in enumerate(image_ids):
        # Load original image
        img = Image.open(test_img_dir / f"{image_id}.jpg").convert("RGB")
        img_resized = img.resize((224, 224))
        img_arr = np.array(img_resized)

        # Load ground truth labels
        gt_labels = np.load(test_labels_dir / f"{image_id}.npy")
        gt_grid = gt_labels.reshape(14, 14)

        # Load hidden states
        hidden = torch.load(
            test_hidden_dir / f"{image_id}.pt", map_location="cpu", weights_only=True
        )

        # Column 0: Original
        axes[idx, 0].imshow(img_arr)
        axes[idx, 0].set_title(f"Original" if idx > 0 else "Original")
        axes[idx, 0].axis("off")
        if idx == 0:
            axes[idx, 0].set_title("Original")

        # Column 1: Ground truth as heatmap
        gt_upsampled = torch.nn.functional.interpolate(
            torch.from_numpy(gt_grid).float().unsqueeze(0).unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()
        axes[idx, 1].imshow(img_arr)
        im_gt = axes[idx, 1].imshow(gt_upsampled, cmap="hot", alpha=0.5, vmin=0, vmax=1)
        axes[idx, 1].set_title("GT" if idx > 0 else "GT")
        axes[idx, 1].axis("off")
        if idx == 0:
            axes[idx, 1].set_title("GT")

        # Columns 2+: Layer heatmaps
        for col_offset, layer in enumerate(available_layers):
            col = 2 + col_offset
            features = hidden[layer].float()  # (196, 768)
            with torch.no_grad():
                logits = probes[layer](features.unsqueeze(0).to(device))  # (1, 196)
            probs = torch.sigmoid(logits).cpu().squeeze().numpy()  # (196,)
            prob_grid = probs.reshape(14, 14)

            # Upsample to 224x224
            prob_upsampled = torch.nn.functional.interpolate(
                torch.from_numpy(prob_grid).float().unsqueeze(0).unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()

            axes[idx, col].imshow(img_arr)
            im = axes[idx, col].imshow(
                prob_upsampled, cmap="hot", alpha=0.5, vmin=0, vmax=1
            )
            axes[idx, col].set_title(f"Layer {layer}")
            axes[idx, col].axis("off")

    # Add a single colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="P(boundary)")

    plt.savefig(save_dir / "heatmap_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: heatmap_grid.png")


def _overlay_patches(ax, grid: np.ndarray, color: str = "red", alpha: float = 0.3):
    """Overlay colored patches on an axis where grid=1."""
    patch_size = 16  # 224 / 14
    for i in range(14):
        for j in range(14):
            if grid[i, j] > 0:
                rect = mpatches.Rectangle(
                    (j * patch_size, i * patch_size),
                    patch_size, patch_size,
                    linewidth=0, facecolor=color, alpha=alpha,
                )
                ax.add_patch(rect)
