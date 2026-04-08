"""Generate layerwise performance plots — the main results figure."""

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_layerwise_metrics(
    results: Dict,
    baseline_metrics: Dict,
    save_dir: str,
    metrics_to_plot: list = ["f1", "accuracy"],
):
    """Plot probe performance across layers with baselines.

    Creates the main results figure showing how boundary detection
    accuracy varies across ViT layers.

    Args:
        results: Nested dict results[model_type][probe_type][layer] = metrics
        baseline_metrics: Dict with baseline results
        save_dir: Directory to save figures
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for metric_name in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each model_type x probe_type combination
        styles = {
            ("pretrained", "linear"): {"color": "blue", "marker": "o", "linestyle": "-"},
            ("pretrained", "mlp"): {"color": "blue", "marker": "s", "linestyle": "--"},
            ("random", "linear"): {"color": "red", "marker": "o", "linestyle": "-"},
            ("random", "mlp"): {"color": "red", "marker": "s", "linestyle": "--"},
        }

        for model_type in results:
            for probe_type in results[model_type]:
                layer_metrics = results[model_type][probe_type]
                layers = sorted(layer_metrics.keys())
                values = [layer_metrics[l][metric_name] for l in layers]

                style = styles.get((model_type, probe_type), {})
                label = f"{model_type} {probe_type}"
                ax.plot(layers, values, label=label, linewidth=2, markersize=6, **style)

        # Plot baselines as horizontal lines
        if "majority_class" in baseline_metrics:
            val = baseline_metrics["majority_class"].get(metric_name, 0)
            ax.axhline(y=val, color="gray", linestyle=":", linewidth=1.5,
                       label=f"majority class ({val:.3f})")

        if "random" in baseline_metrics:
            val = baseline_metrics["random"].get(metric_name, 0)
            ax.axhline(y=val, color="gray", linestyle="-.", linewidth=1.5,
                       label=f"random ({val:.3f})")

        ax.set_xlabel("ViT Layer", fontsize=12)
        ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=12)
        ax.set_title(f"Boundary Detection Probe: {metric_name.replace('_', ' ').title()} by Layer", fontsize=14)
        ax.set_xticks(range(13))
        ax.set_xticklabels([f"L{i}" for i in range(13)])
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_dir / f"layerwise_{metric_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: layerwise_{metric_name}.png")


def plot_layerwise_comparison(
    results: Dict,
    save_dir: str,
):
    """Side-by-side comparison: pretrained vs random across layers."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if "pretrained" not in results or "random" not in results:
        print("  Skipping comparison plot (need both pretrained and random results)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, metric in enumerate(["f1", "accuracy"]):
        ax = axes[idx]

        for model_type, color in [("pretrained", "blue"), ("random", "red")]:
            if "linear" in results[model_type]:
                layer_metrics = results[model_type]["linear"]
                layers = sorted(layer_metrics.keys())
                values = [layer_metrics[l][metric] for l in layers]
                ax.plot(layers, values, color=color, marker="o", linewidth=2,
                       label=f"{model_type} (linear)")

        ax.set_xlabel("ViT Layer", fontsize=11)
        ax.set_ylabel(metric.title(), fontsize=11)
        ax.set_title(f"Pretrained vs Random: {metric.title()}", fontsize=12)
        ax.set_xticks(range(13))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_dir / "pretrained_vs_random.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: pretrained_vs_random.png")
