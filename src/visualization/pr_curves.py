"""Precision-recall curve visualization."""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score


def plot_pr_curves(
    config: dict,
    device: torch.device,
    layers_to_plot: List[int] = [0, 3, 6, 9, 12],
    save_dir: str = "results/figures",
):
    """Plot precision-recall curves for selected layers.

    Args:
        config: Configuration dict.
        device: Torch device.
        layers_to_plot: Which layers to include in the plot.
        save_dir: Where to save the figure.
    """
    from src.data.hidden_state_dataset import PatchLevelDataModule
    from src.probes.linear_probe import get_probe
    from src.evaluation.metrics import evaluate_probe

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states"
    labels_dir = Path(config["dataset"]["processed_dir"]) / "patch_labels"
    checkpoints_dir = Path(config["results"]["checkpoints_dir"])

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(layers_to_plot)))

    for layer, color in zip(layers_to_plot, colors):
        run_name = f"pretrained_linear_layer{layer:02d}"
        checkpoint_path = checkpoints_dir / f"{run_name}.pt"

        if not checkpoint_path.exists():
            continue

        # Load probe
        probe = get_probe("linear", input_dim=config["model"]["hidden_dim"])
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        probe.load_state_dict(state_dict)
        probe.to(device)

        # Get test predictions
        data_module = PatchLevelDataModule(
            cached_dir=str(cached_dir),
            labels_dir=str(labels_dir),
            model_type="pretrained",
            layer=layer,
            batch_size=config["training"]["batch_size"],
            num_workers=config["num_workers"],
        )

        metrics, probs, labels = evaluate_probe(
            probe, data_module.test_dataloader(), device,
        )

        # Plot PR curve
        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        ax.plot(recall, precision, color=color, linewidth=2,
                label=f"Layer {layer} (AP={ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves by Layer (Pretrained, Linear Probe)", fontsize=13)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    fig.savefig(save_dir / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: pr_curves.png")
