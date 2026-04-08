"""Generate all depth probing visualization figures (8 total)."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import load_config
from src.utils.device import get_device
from src.probes.linear_probe import get_probe
from src.training.depth_trainer import DepthProbeTrainer
from src.data.depth_dataset import DepthDataModule


def _load_results(config: dict):
    path = Path(config["results"]["metrics_dir"]) / "depth_test_results.json"
    with open(path) as f:
        raw = json.load(f)
    results = {}
    for mt in raw:
        results[mt] = {}
        for pt in raw[mt]:
            results[mt][pt] = {int(k): v for k, v in raw[mt][pt].items()}
    return results


def _get_best_layer(results: dict, model_type: str = "pretrained", probe_type: str = "linear") -> int:
    layer_metrics = results[model_type][probe_type]
    return min(layer_metrics, key=lambda l: layer_metrics[l]["mae"])


def _run_inference(config, device, model_type, probe_type, layer, split="test"):
    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states"
    labels_dir = Path(config["dataset"]["processed_dir"]) / "depth_labels"
    checkpoints_dir = Path(config["results"]["checkpoints_dir"])

    run_name = f"{model_type}_{probe_type}_layer{layer:02d}"
    probe_kwargs = {}
    if probe_type == "mlp":
        probe_kwargs = {
            "hidden_dim": config["mlp_probe"]["hidden_dim"],
            "dropout": config["mlp_probe"]["dropout"],
        }

    probe = get_probe(probe_type, input_dim=config["model"]["hidden_dim"], **probe_kwargs)
    state_dict = torch.load(checkpoints_dir / f"{run_name}.pt", map_location="cpu", weights_only=True)
    probe.load_state_dict(state_dict)

    dm = DepthDataModule(
        cached_dir=str(cached_dir),
        labels_dir=str(labels_dir),
        model_type=model_type,
        layer=layer,
        batch_size=config["training"]["batch_size"],
        num_workers=config["num_workers"],
    )
    loader = dm.test_dataloader() if split == "test" else dm.val_dataloader()

    trainer = DepthProbeTrainer(model=probe, device=device)
    trainer.model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for feats, lbls in loader:
            preds = trainer.model(feats.to(device)).cpu()
            all_preds.append(preds)
            all_labels.append(lbls)

    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def _resize_patch_grid(arr_14x14):
    """Resize a 14x14 float32 array to 224x224 using PIL BILINEAR."""
    from PIL import Image as PILImage
    import numpy as np
    arr = arr_14x14.astype(np.float32)
    # PIL MODE 'F' handles float32
    pil_img = PILImage.fromarray(arr, mode='F')
    return np.array(pil_img.resize((224, 224), PILImage.BILINEAR))


def plot_layerwise_mae(results: dict, figures_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {
        ("pretrained", "linear"): {"color": "blue", "marker": "o", "linestyle": "-"},
        ("pretrained", "mlp"):    {"color": "blue", "marker": "s", "linestyle": "--"},
        ("random",     "linear"): {"color": "red",  "marker": "o", "linestyle": "-"},
        ("random",     "mlp"):    {"color": "red",  "marker": "s", "linestyle": "--"},
    }
    for model_type in results:
        for probe_type in results[model_type]:
            layer_metrics = results[model_type][probe_type]
            layers = sorted(layer_metrics.keys())
            maes = [layer_metrics[l]["mae"] for l in layers]
            style = styles.get((model_type, probe_type), {})
            ax.plot(layers, maes, label=f"{model_type} {probe_type}", linewidth=2, markersize=6, **style)
    ax.set_xlabel("ViT Layer", fontsize=12)
    ax.set_ylabel("MAE (normalized depth [0,1])", fontsize=12)
    ax.set_title("Depth Probe: MAE by Layer", fontsize=14)
    ax.set_xticks(range(13))
    ax.set_xticklabels([f"L{i}" for i in range(13)])
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "layerwise_mae.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: layerwise_mae.png")


def plot_pretrained_vs_random(results: dict, figures_dir: Path):
    if "pretrained" not in results or "random" not in results:
        print("  Skipping pretrained_vs_random_depth.png (missing model types)")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_type, color in [("pretrained", "blue"), ("random", "red")]:
        if "linear" in results[model_type]:
            lm = results[model_type]["linear"]
            layers = sorted(lm.keys())
            ax.plot(layers, [lm[l]["mae"] for l in layers], color=color,
                    marker="o", linewidth=2, label=f"{model_type} (linear)")
    ax.set_xlabel("ViT Layer", fontsize=12)
    ax.set_ylabel("MAE (normalized depth [0,1])", fontsize=12)
    ax.set_title("Depth Probe: Pretrained vs Random ViT", fontsize=14)
    ax.set_xticks(range(13))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "pretrained_vs_random_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: pretrained_vs_random_depth.png")


def plot_cross_task(depth_results: dict, figures_dir: Path):
    boundary_path = Path("results/metrics/test_results.json")
    if not boundary_path.exists():
        print("  Skipping cross_task_layerwise.png (boundary results not found)")
        return
    with open(boundary_path) as f:
        boundary_raw = json.load(f)
    boundary_results = {}
    for mt in boundary_raw:
        boundary_results[mt] = {}
        for pt in boundary_raw[mt]:
            boundary_results[mt][pt] = {int(k): v for k, v in boundary_raw[mt][pt].items()}

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    if "pretrained" in boundary_results and "linear" in boundary_results["pretrained"]:
        lm = boundary_results["pretrained"]["linear"]
        layers = sorted(lm.keys())
        ax1.plot(layers, [lm[l]["f1"] for l in layers],
                 color="green", marker="o", linewidth=2, label="Boundary F1 (pretrained linear)")
    if "pretrained" in depth_results and "linear" in depth_results["pretrained"]:
        lm = depth_results["pretrained"]["linear"]
        layers = sorted(lm.keys())
        ax2.plot(layers, [lm[l]["mae"] for l in layers],
                 color="purple", marker="s", linewidth=2, linestyle="--", label="Depth MAE (pretrained linear)")
    ax1.set_xlabel("ViT Layer", fontsize=12)
    ax1.set_ylabel("Boundary F1 (higher = better)", fontsize=12, color="green")
    ax2.set_ylabel("Depth MAE (lower = better)", fontsize=12, color="purple")
    ax1.set_xticks(range(13))
    ax1.set_xticklabels([f"L{i}" for i in range(13)])
    ax1.tick_params(axis="y", labelcolor="green")
    ax2.tick_params(axis="y", labelcolor="purple")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)
    ax1.set_title("Boundary vs Depth Encoding Across ViT Layers", fontsize=14)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "cross_task_layerwise.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: cross_task_layerwise.png")


def plot_qualitative(config: dict, device: torch.device, results: dict, figures_dir: Path):
    best_layer = _get_best_layer(results)
    raw_dir = Path(config["dataset"]["raw_dir"])
    labels_dir = Path(config["dataset"]["processed_dir"]) / "depth_labels" / "test"
    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states" / "pretrained" / "test"
    image_ids = sorted([p.stem for p in (raw_dir / "images" / "test").glob("*.png")])[:5]
    run_name = f"pretrained_linear_layer{best_layer:02d}"
    probe = get_probe("linear", input_dim=config["model"]["hidden_dim"])
    state = torch.load(Path(config["results"]["checkpoints_dir"]) / f"{run_name}.pt",
                       map_location="cpu", weights_only=True)
    probe.load_state_dict(state)
    probe.eval()
    fig, axes = plt.subplots(5, 3, figsize=(12, 20))
    for row, image_id in enumerate(image_ids):
        image = Image.open(raw_dir / "images" / "test" / f"{image_id}.png").convert("RGB")
        gt_labels = np.load(labels_dir / f"{image_id}.npy").reshape(14, 14)
        hidden = torch.load(cached_dir / f"{image_id}.pt", weights_only=True)
        feats = hidden[best_layer].float()
        with torch.no_grad():
            pred = probe(feats).numpy().reshape(14, 14)
        gt_up = _resize_patch_grid(gt_labels)
        pred_up = _resize_patch_grid(pred)
        axes[row, 0].imshow(image)
        axes[row, 0].set_title("Original" if row == 0 else "")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(gt_up, cmap="plasma", vmin=0, vmax=1)
        axes[row, 1].set_title("GT Depth (norm.)" if row == 0 else "")
        axes[row, 1].axis("off")
        axes[row, 2].imshow(pred_up, cmap="plasma", vmin=0, vmax=1)
        axes[row, 2].set_title(f"Predicted (L{best_layer})" if row == 0 else "")
        axes[row, 2].axis("off")
    plt.suptitle(f"Depth Probe Qualitative — Best Layer L{best_layer}", fontsize=14)
    plt.tight_layout()
    fig.savefig(figures_dir / f"qualitative_depth_layer{best_layer:02d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: qualitative_depth_layer{best_layer:02d}.png")


def plot_heatmap_grid(config: dict, device: torch.device, figures_dir: Path):
    target_layers = [0, 3, 6, 9, 12]
    raw_dir = Path(config["dataset"]["raw_dir"])
    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states" / "pretrained" / "test"
    checkpoints_dir = Path(config["results"]["checkpoints_dir"])
    image_ids = sorted([p.stem for p in (raw_dir / "images" / "test").glob("*.png")])[:5]
    probes = {}
    for layer in target_layers:
        run_name = f"pretrained_linear_layer{layer:02d}"
        p = get_probe("linear", input_dim=config["model"]["hidden_dim"])
        state = torch.load(checkpoints_dir / f"{run_name}.pt", map_location="cpu", weights_only=True)
        p.load_state_dict(state)
        p.eval()
        probes[layer] = p
    n_rows, n_cols = len(image_ids), len(target_layers)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    im = None
    for row, image_id in enumerate(image_ids):
        hidden = torch.load(cached_dir / f"{image_id}.pt", weights_only=True)
        image = Image.open(raw_dir / "images" / "test" / f"{image_id}.png").convert("RGB")
        img_arr = np.array(image.resize((224, 224)))
        for col, layer in enumerate(target_layers):
            feats = hidden[layer].float()
            with torch.no_grad():
                pred = probes[layer](feats).numpy().reshape(14, 14)
            pred_up = _resize_patch_grid(pred)
            ax = axes[row, col]
            ax.imshow(img_arr, alpha=0.5)
            im = ax.imshow(pred_up, cmap="plasma", alpha=0.6, vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"Layer {layer}", fontsize=10)
    if im is not None:
        plt.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.02,
                     label="Predicted depth (normalized)")
    plt.suptitle("Depth Probe Heatmap Grid — Pretrained Linear", fontsize=13)
    plt.tight_layout()
    fig.savefig(figures_dir / "depth_heatmap_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: depth_heatmap_grid.png")


def plot_scatter(config: dict, device: torch.device, results: dict, figures_dir: Path):
    best_layer = _get_best_layer(results)
    preds, labels = _run_inference(config, device, "pretrained", "linear", best_layer)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(labels, preds, alpha=0.03, s=1, color="steelblue")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual depth (normalized)", fontsize=12)
    ax.set_ylabel("Predicted depth (normalized)", fontsize=12)
    ax.set_title(f"Depth Probe Calibration — Layer {best_layer} (linear, pretrained)", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    mae = float(np.abs(preds - labels).mean())
    ax.text(0.05, 0.92, f"MAE = {mae:.4f} ({mae*10:.3f}m)", transform=ax.transAxes, fontsize=11)
    plt.tight_layout()
    fig.savefig(figures_dir / "depth_scatter_best_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: depth_scatter_best_layer.png")


def plot_mae_by_bin(config: dict, device: torch.device, results: dict, figures_dir: Path):
    best_layer = _get_best_layer(results)
    preds, labels = _run_inference(config, device, "pretrained", "linear", best_layer)
    bins = [(0.0, 0.3, "Near\n(0–3m)"), (0.3, 0.6, "Mid\n(3–6m)"), (0.6, 1.0, "Far\n(6–10m)")]
    bin_maes, bin_labels, bin_counts = [], [], []
    for lo, hi, label in bins:
        mask = (labels >= lo) & (labels < hi)
        if mask.sum() > 0:
            bin_maes.append(float(np.abs(preds[mask] - labels[mask]).mean()))
            bin_counts.append(int(mask.sum()))
        else:
            bin_maes.append(0.0)
            bin_counts.append(0)
        bin_labels.append(label)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(bin_labels, bin_maes, color=["#4c72b0", "#dd8452", "#55a868"], edgecolor="black", linewidth=0.8)
    for bar, count in zip(bars, bin_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"n={count:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("MAE (normalized depth [0,1])", fontsize=12)
    ax.set_title(f"Depth Probe MAE by Depth Range — Layer {best_layer} (linear, pretrained)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "mae_by_depth_bin.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: mae_by_depth_bin.png")


def plot_patch_error_map(config: dict, device: torch.device, results: dict, figures_dir: Path):
    best_layer = _get_best_layer(results)
    raw_dir = Path(config["dataset"]["raw_dir"])
    labels_dir = Path(config["dataset"]["processed_dir"]) / "depth_labels" / "test"
    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states" / "pretrained" / "test"
    checkpoints_dir = Path(config["results"]["checkpoints_dir"])
    image_ids = sorted([p.stem for p in (raw_dir / "images" / "test").glob("*.png")])[:5]
    run_name = f"pretrained_linear_layer{best_layer:02d}"
    probe = get_probe("linear", input_dim=config["model"]["hidden_dim"])
    state = torch.load(checkpoints_dir / f"{run_name}.pt", map_location="cpu", weights_only=True)
    probe.load_state_dict(state)
    probe.eval()
    fig, axes = plt.subplots(5, 3, figsize=(12, 20))
    im = None
    for row, image_id in enumerate(image_ids):
        image = Image.open(raw_dir / "images" / "test" / f"{image_id}.png").convert("RGB")
        gt = np.load(labels_dir / f"{image_id}.npy").reshape(14, 14)
        hidden = torch.load(cached_dir / f"{image_id}.pt", weights_only=True)
        with torch.no_grad():
            pred = probe(hidden[best_layer].float()).numpy().reshape(14, 14)
        error = np.abs(pred - gt)
        error_up = _resize_patch_grid(error)
        img_224 = np.array(image.resize((224, 224)))
        axes[row, 0].imshow(img_224)
        axes[row, 0].set_title("Original" if row == 0 else "")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(_resize_patch_grid(gt), cmap="plasma", vmin=0, vmax=1)
        axes[row, 1].set_title("GT Depth" if row == 0 else "")
        axes[row, 1].axis("off")
        im = axes[row, 2].imshow(error_up, cmap="hot", vmin=0, vmax=0.3)
        axes[row, 2].set_title(f"Abs Error (L{best_layer})" if row == 0 else "")
        axes[row, 2].axis("off")
    if im is not None:
        plt.colorbar(im, ax=axes[:, 2], orientation="vertical", fraction=0.05, pad=0.02,
                     label="|pred - GT| (normalized)")
    plt.suptitle(f"Per-Patch Depth Error Map — Layer {best_layer}", fontsize=14)
    plt.tight_layout()
    fig.savefig(figures_dir / "patch_error_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: patch_error_map.png")


def main():
    config = load_config("configs/depth.yaml")
    device = get_device()
    figures_dir = Path(config["results"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    results = _load_results(config)

    print("Figure 1: Layerwise MAE...")
    plot_layerwise_mae(results, figures_dir)

    print("Figure 2: Pretrained vs Random...")
    plot_pretrained_vs_random(results, figures_dir)

    print("Figure 3: Cross-task comparison...")
    plot_cross_task(results, figures_dir)

    print("Figure 4: Qualitative...")
    plot_qualitative(config, device, results, figures_dir)

    print("Figure 5: Heatmap grid...")
    plot_heatmap_grid(config, device, figures_dir)

    print("Figure 6: Scatter plot...")
    plot_scatter(config, device, results, figures_dir)

    print("Figure 7: MAE by depth bin...")
    plot_mae_by_bin(config, device, results, figures_dir)

    print("Figure 8: Patch error map...")
    plot_patch_error_map(config, device, results, figures_dir)

    print(f"\nAll figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
