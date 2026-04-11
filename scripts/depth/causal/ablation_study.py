"""Probe direction ablation study.

Tests whether depth information is concentrated in the linear probe's weight
direction (structured) or diffuse across many directions (unstructured).

Three experiments per layer:
1. Probe direction ablation: project out the probe's weight vector w
2. Random direction ablation: project out 10 random unit vectors (control)
3. Dose-response at best layer (L=8): scale ablation by alpha in {0,0.1,...,1.0}
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.utils.config import load_config
from src.data.depth_dataset import DepthDataModule


def _load_probe_direction(checkpoint_path: str) -> torch.Tensor:
    """Load linear probe checkpoint and return normalized weight vector (768,)."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    w = state["linear.weight"].squeeze(0).float()  # (768,)
    return w / w.norm()


def _load_test_features_labels(config: dict, layer: int):
    """Load all test features and labels for a given layer."""
    dm = DepthDataModule(
        cached_dir=str(Path(config["dataset"]["cached_dir"]) / "hidden_states"),
        labels_dir=str(Path(config["dataset"]["processed_dir"]) / "depth_labels"),
        model_type="pretrained",
        layer=layer,
        batch_size=512,
        num_workers=0,
    )
    dataset = dm._get_dataset("test")
    return dataset.features.float(), dataset.labels


def _compute_mae(features, labels, probe_w, probe_b):
    """MAE using probe prediction = features @ w + b."""
    with torch.no_grad():
        preds = features @ probe_w + probe_b
    return float((preds - labels).abs().mean())


def _ablate(features, direction, alpha=1.0):
    """Project out alpha fraction of the component along unit vector direction."""
    proj = (features @ direction).unsqueeze(1) * direction.unsqueeze(0)
    return features - alpha * proj


def run_ablation(config: dict) -> dict:
    checkpoints_dir = Path(config["results"]["checkpoints_dir"])
    layers = config["training"]["layers"]
    torch.manual_seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))

    with open(Path(config["results"]["metrics_dir"]) / "depth_test_results.json") as f:
        existing = json.load(f)

    results = {
        "probe_direction_ablation": {},
        "random_direction_ablation": {},
        "dose_response": {},
    }

    for layer in layers:
        print(f"\nLayer {layer}...")
        ckpt = checkpoints_dir / f"pretrained_linear_layer{layer:02d}.pt"
        if not ckpt.exists():
            continue

        w_hat = _load_probe_direction(str(ckpt))
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        probe_w = state["linear.weight"].squeeze(0).float()
        probe_b = float(state["linear.bias"])

        features, labels = _load_test_features_labels(config, layer)
        original_mae = existing["pretrained"]["linear"][str(layer)]["mae"]

        # 1. Probe direction ablation
        ablated_mae = _compute_mae(_ablate(features, w_hat), labels, probe_w, probe_b)
        gap = ablated_mae - original_mae
        gap_pct = 100.0 * gap / original_mae
        results["probe_direction_ablation"][str(layer)] = {
            "original_mae": original_mae,
            "ablated_mae": ablated_mae,
            "gap": gap,
            "gap_pct": gap_pct,
        }
        print(f"  Probe dir: {original_mae:.4f} -> {ablated_mae:.4f} (+{gap_pct:.1f}%)")

        # 2. Random direction ablation (10 trials)
        random_maes = []
        for _ in range(10):
            d = torch.randn(768)
            d = d / d.norm()
            random_maes.append(_compute_mae(_ablate(features, d), labels, probe_w, probe_b))
        results["random_direction_ablation"][str(layer)] = {
            "original_mae": original_mae,
            "mean_ablated_mae": float(np.mean(random_maes)),
            "std_ablated_mae": float(np.std(random_maes)),
            "n_random": 10,
        }
        print(f"  Random dir: mean={np.mean(random_maes):.4f} +/- {np.std(random_maes):.4f}")

        # 3. Dose-response (best layer = 8)
        if layer == 8:
            dose = {}
            for a in range(11):
                alpha = round(a * 0.1, 1)
                mae_a = _compute_mae(_ablate(features, w_hat, alpha=alpha), labels, probe_w, probe_b)
                dose[str(alpha)] = mae_a
                print(f"  Dose alpha={alpha:.1f}: MAE={mae_a:.4f}")
            results["dose_response"]["8"] = dose

    return results


def main():
    config = load_config("configs/depth.yaml")
    results = run_ablation(config)

    out_path = Path(config["results"]["metrics_dir"]) / "ablation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'Layer':<6} {'Orig MAE':>10} {'Ablated':>10} {'Gap%':>8} {'Rand mean':>10}")
    print("-" * 50)
    for layer in range(13):
        k = str(layer)
        if k not in results["probe_direction_ablation"]:
            continue
        p = results["probe_direction_ablation"][k]
        r = results["random_direction_ablation"][k]
        print(f"  L{layer:<4} {p['original_mae']:>10.4f} {p['ablated_mae']:>10.4f} "
              f"{p['gap_pct']:>7.1f}% {r['mean_ablated_mae']:>10.4f}")


if __name__ == "__main__":
    main()
