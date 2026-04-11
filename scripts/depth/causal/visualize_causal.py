"""Generate causal analysis figures (5 total)."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.utils.config import load_config
from src.probes.linear_probe import get_probe
from src.data.depth_dataset import DepthDataModule


def _load_ablation(config):
    with open(Path(config["results"]["metrics_dir"]) / "ablation_results.json") as f:
        return json.load(f)


def _load_patching(config):
    with open(Path(config["results"]["metrics_dir"]) / "patching_results.json") as f:
        return json.load(f)


def plot_ablation_gap(ablation, figures_dir):
    """Figure 1: Ablation gap by layer — bar chart with random-direction error bars."""
    layers = sorted(int(k) for k in ablation["probe_direction_ablation"].keys())
    probe_gaps = [ablation["probe_direction_ablation"][str(l)]["gap_pct"] for l in layers]
    rand_gaps = []
    rand_stds = []
    for l in layers:
        r = ablation["random_direction_ablation"][str(l)]
        orig = ablation["probe_direction_ablation"][str(l)]["original_mae"]
        rand_gap_pct = 100.0 * (r["mean_ablated_mae"] - orig) / orig
        rand_std_pct = 100.0 * r["std_ablated_mae"] / orig
        rand_gaps.append(rand_gap_pct)
        rand_stds.append(rand_std_pct)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(layers))
    width = 0.35

    bars1 = ax.bar(x - width/2, probe_gaps, width, label="Probe direction", color="steelblue", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, rand_gaps, width, yerr=rand_stds, capsize=3,
                    label="Random direction (mean ± std)", color="lightcoral", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("ViT Layer", fontsize=12)
    ax.set_ylabel("MAE Increase (%)", fontsize=12)
    ax.set_title("Probe Direction Ablation: Depth Signal Concentration", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "ablation_gap_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: ablation_gap_by_layer.png")


def plot_dose_response(ablation, figures_dir):
    """Figure 2: Dose-response curve at layer 8."""
    dose = ablation["dose_response"].get("8", {})
    if not dose:
        print("  Skipping dose_response_layer08.png (no data)")
        return

    alphas = sorted(float(k) for k in dose.keys())
    maes = [dose[str(a)] for a in alphas]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, maes, "o-", color="steelblue", linewidth=2, markersize=6)
    ax.axhline(y=maes[0], color="green", linestyle="--", alpha=0.5, label=f"Original MAE ({maes[0]:.4f})")
    ax.axhline(y=maes[-1], color="red", linestyle="--", alpha=0.5, label=f"Fully ablated ({maes[-1]:.4f})")
    ax.fill_between(alphas, maes[0], maes, alpha=0.15, color="steelblue")

    ax.set_xlabel("Ablation strength (α)", fontsize=12)
    ax.set_ylabel("MAE (normalized depth)", fontsize=12)
    ax.set_title("Dose-Response: Gradual Depth Direction Ablation (Layer 8)", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "dose_response_layer08.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: dose_response_layer08.png")


def plot_ablation_qualitative(config, ablation, figures_dir):
    """Figure 3: Qualitative ablation — original vs ablated predicted depth for 3 images."""
    best_layer = 8
    ckpt = Path(config["results"]["checkpoints_dir"]) / f"pretrained_linear_layer{best_layer:02d}.pt"
    state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    probe_w = state["linear.weight"].squeeze(0).float()
    probe_b = float(state["linear.bias"])
    w_hat = probe_w / probe_w.norm()

    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states" / "pretrained" / "test"
    labels_dir = Path(config["dataset"]["processed_dir"]) / "depth_labels" / "test"
    raw_dir = Path(config["dataset"]["raw_dir"])

    image_ids = sorted([p.stem for p in (raw_dir / "images" / "test").glob("*.png")])[:3]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for row, img_id in enumerate(image_ids):
        from PIL import Image
        image = Image.open(raw_dir / "images" / "test" / f"{img_id}.png").convert("RGB")
        hidden = torch.load(cached_dir / f"{img_id}.pt", map_location="cpu", weights_only=True)
        gt = np.load(labels_dir / f"{img_id}.npy").reshape(14, 14)

        feats = hidden[best_layer].float()  # (196, 768)
        with torch.no_grad():
            orig_pred = (feats @ probe_w + probe_b).numpy().reshape(14, 14)

        # Ablate
        proj = (feats @ w_hat).unsqueeze(1) * w_hat.unsqueeze(0)
        feats_abl = feats - proj
        with torch.no_grad():
            abl_pred = (feats_abl @ probe_w + probe_b).numpy().reshape(14, 14)

        # Upsample
        from PIL import Image as PILImage
        def up(arr):
            return np.array(PILImage.fromarray(arr.astype(np.float32), mode='F').resize((224, 224), PILImage.BILINEAR))

        axes[row, 0].imshow(up(orig_pred), cmap="plasma", vmin=0, vmax=1)
        axes[row, 0].set_title("Original prediction" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(up(abl_pred), cmap="plasma", vmin=0, vmax=1)
        axes[row, 1].set_title("After ablation" if row == 0 else "")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(up(gt), cmap="plasma", vmin=0, vmax=1)
        axes[row, 2].set_title("Ground truth" if row == 0 else "")
        axes[row, 2].axis("off")

    plt.suptitle(f"Probe Direction Ablation — Layer {best_layer}", fontsize=14)
    plt.tight_layout()
    fig.savefig(figures_dir / "ablation_qualitative.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: ablation_qualitative.png")


def plot_patch_heatmap(patching, figures_dir):
    """Figure 4: Patch effect heatmap (12x13 lower triangular)."""
    layers = list(range(13))
    intervention_layers = list(range(1, 13))
    matrix = np.full((12, 13), np.nan)

    for L_idx, L in enumerate(intervention_layers):
        for T in layers:
            if T < L:
                continue
            key = f"L{L:02d}_T{T:02d}"
            if key in patching["patch_effects"]:
                matrix[L_idx, T] = patching["patch_effects"][key]["mean"]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1.1, aspect="auto")
    ax.set_xlabel("Target Layer (T)", fontsize=12)
    ax.set_ylabel("Intervention Layer (L)", fontsize=12)
    ax.set_xticks(range(13))
    ax.set_xticklabels([f"L{l}" for l in range(13)])
    ax.set_yticks(range(12))
    ax.set_yticklabels([f"L{l}" for l in range(1, 13)])
    ax.set_title("Activation Patching: Causal Influence Matrix", fontsize=14)

    # Add text annotations
    for i in range(12):
        for j in range(13):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=7,
                       color="white" if matrix[i, j] > 0.6 else "black")

    plt.colorbar(im, ax=ax, label="Patch Effect (0=no shift, 1=full shift)")
    plt.tight_layout()
    fig.savefig(figures_dir / "patch_effect_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: patch_effect_heatmap.png")


def plot_patch_diagonal(patching, figures_dir):
    """Figure 5: Diagonal patch effect (T=L) line plot."""
    intervention_layers = list(range(1, 13))
    diag = []
    for L in intervention_layers:
        key = f"L{L:02d}_T{L:02d}"
        if key in patching["patch_effects"]:
            diag.append(patching["patch_effects"][key]["mean"])
        else:
            diag.append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(intervention_layers, diag, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Perfect causal effect")
    ax.set_xlabel("Layer (L = T)", fontsize=12)
    ax.set_ylabel("Patch Effect at T=L", fontsize=12)
    ax.set_title("Immediate Causal Effect of Activation Patching", fontsize=13)
    ax.set_xticks(intervention_layers)
    ax.set_xticklabels([f"L{l}" for l in intervention_layers])
    ax.set_ylim(0.9, 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "patch_effect_diagonal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: patch_effect_diagonal.png")


def main():
    config = load_config("configs/depth.yaml")
    figures_dir = Path(config["results"]["figures_dir"]) / "causal"
    figures_dir.mkdir(parents=True, exist_ok=True)

    ablation = _load_ablation(config)
    patching = _load_patching(config)

    print("Figure 1: Ablation gap by layer...")
    plot_ablation_gap(ablation, figures_dir)

    print("Figure 2: Dose-response...")
    plot_dose_response(ablation, figures_dir)

    print("Figure 3: Ablation qualitative...")
    plot_ablation_qualitative(config, ablation, figures_dir)

    print("Figure 4: Patch effect heatmap...")
    plot_patch_heatmap(patching, figures_dir)

    print("Figure 5: Patch effect diagonal...")
    plot_patch_diagonal(patching, figures_dir)

    print(f"\nAll causal figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
