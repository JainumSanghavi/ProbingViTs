"""Targeted activation patching: swap only the depth-direction component.

Instead of replacing all 196 patch tokens (which trivially shifts predictions),
we replace ONLY the 1D depth-direction component at layer L:

    h_patched = h_source + (proj_w(h_dest) - proj_w(h_source))

where proj_w(h) = (h . w_hat) * w_hat is the projection onto the probe's
weight direction.  This keeps the other 767 dimensions from the source intact.

If depth predictions shift -> the depth direction at layer L causally drives
downstream depth predictions.  If they don't -> the model re-derives depth
from other features at later layers.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.utils.config import load_config
from src.utils.device import get_device
from src.models.vit_extractor import ViTExtractor
from src.data.transforms import get_vit_transform
from src.probes.linear_probe import get_probe


def _select_pairs(config: dict, n_near: int = 10, n_far: int = 10):
    """Select image pairs with maximum depth contrast from test set."""
    labels_dir = Path(config["dataset"]["processed_dir"]) / "depth_labels" / "test"
    ids_and_means = []
    for p in sorted(labels_dir.glob("*.npy")):
        labels = np.load(p)
        ids_and_means.append((p.stem, float(labels.mean())))

    ids_and_means.sort(key=lambda x: x[1])
    near_ids = [x[0] for x in ids_and_means[:n_near]]
    far_ids = [x[0] for x in ids_and_means[-n_far:]]

    pairs = []
    for n_id in near_ids:
        for f_id in far_ids:
            pairs.append((n_id, f_id))
            if len(pairs) >= 20:
                break
        if len(pairs) >= 20:
            break
    return pairs


def _load_probes_and_directions(config: dict):
    """Load probes and extract normalised weight directions for each layer."""
    checkpoints_dir = Path(config["results"]["checkpoints_dir"])
    probes = {}
    directions = {}
    for layer in config["training"]["layers"]:
        ckpt = checkpoints_dir / f"pretrained_linear_layer{layer:02d}.pt"
        if ckpt.exists():
            probe = get_probe("linear", input_dim=config["model"]["hidden_dim"])
            state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
            probe.load_state_dict(state)
            probe.eval()
            probes[layer] = probe
            w = state["linear.weight"].squeeze(0).float()
            directions[layer] = w / w.norm()
    return probes, directions


def _predict_depth(probe, hidden_states_single_layer):
    """Run probe on (196, 768) -> (196,) predictions."""
    with torch.no_grad():
        return probe(hidden_states_single_layer.float()).numpy()


def run_patching(config: dict):
    device = get_device()
    print(f"Using device: {device}")

    raw_dir = Path(config["dataset"]["raw_dir"])
    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states" / "pretrained" / "test"
    transform = get_vit_transform(config["model"]["image_size"])

    pairs = _select_pairs(config)
    print(f"Selected {len(pairs)} (near, far) image pairs")

    probes, directions = _load_probes_and_directions(config)
    print(f"Loaded {len(probes)} probe checkpoints + depth directions")

    extractor = ViTExtractor(
        model_name=config["model"]["name"],
        pretrained=True,
        device=device,
    )

    layers = config["training"]["layers"]  # [0, 1, ..., 12]
    intervention_layers = [l for l in layers if l >= 1]  # can't hook layer 0

    # patch_effects[L][T] = list of effects across pairs
    patch_effects = {L: {T: [] for T in layers if T >= L} for L in intervention_layers}

    for pair_idx, (source_id, dest_id) in enumerate(pairs):
        print(f"\nPair {pair_idx+1}/{len(pairs)}: source={source_id}, dest={dest_id}")

        # Load source image
        source_img = Image.open(raw_dir / "images" / "test" / f"{source_id}.png").convert("RGB")
        source_pixels = transform(source_img).to(device)

        # Load cached destination hidden states (13, 196, 768)
        dest_hidden = torch.load(cached_dir / f"{dest_id}.pt", map_location="cpu", weights_only=True)

        # Get source baseline hidden states (no patching)
        source_hidden = extractor.extract_single(source_pixels).cpu()  # (13, 196, 768)

        # Baseline predictions at each layer
        dest_preds = {}
        source_preds = {}
        for T in layers:
            if T in probes:
                dest_preds[T] = _predict_depth(probes[T], dest_hidden[T])
                source_preds[T] = _predict_depth(probes[T], source_hidden[T])

        # For each intervention layer L: swap only the depth direction
        for L in intervention_layers:
            if L not in directions:
                continue

            w_hat = directions[L].to(device)  # (768,) unit vector
            dest_patches_L = dest_hidden[L].float().to(device)  # (196, 768)

            def make_hook(w_hat_local, dest_local):
                def hook_fn(module, input, output):
                    modified = output[0].clone()  # (B, 197, 768)
                    h_source = modified[:, 1:, :]  # (B, 196, 768) — source patch tokens

                    # Project source and dest onto depth direction
                    src_proj = (h_source @ w_hat_local).unsqueeze(-1) * w_hat_local  # (B, 196, 768)
                    dst_proj = (dest_local.unsqueeze(0) @ w_hat_local).unsqueeze(-1) * w_hat_local  # (1, 196, 768)

                    # Swap only the depth component: keep source's other 767 dims
                    modified[:, 1:, :] = h_source - src_proj + dst_proj
                    return (modified,) + output[1:]
                return hook_fn

            hook = make_hook(w_hat, dest_patches_L)
            patched_hidden = extractor.extract_with_hook(source_pixels, hook, layer_idx=L-1).cpu()

            # Measure effect at each downstream layer T >= L
            for T in layers:
                if T < L or T not in probes:
                    continue

                patched_pred = _predict_depth(probes[T], patched_hidden[T])
                source_pred = source_preds[T]
                dest_pred = dest_preds[T]

                shift = float(np.abs(patched_pred - source_pred).mean())
                total_gap = float(np.abs(dest_pred - source_pred).mean())
                effect = shift / max(total_gap, 1e-8)

                patch_effects[L][T].append(effect)

    # Aggregate
    results = {
        "pairs": [{"source": s, "dest": d} for s, d in pairs],
        "n_pairs": len(pairs),
        "intervention_type": "depth_direction_only",
        "patch_effects": {},
    }

    print("\n=== TARGETED PATCH EFFECT MATRIX (mean across pairs) ===")
    print(f"{'L\\T':<6}", end="")
    for T in layers:
        print(f"T={T:<5}", end="")
    print()

    for L in intervention_layers:
        print(f"L={L:<4}", end="")
        for T in layers:
            if T < L or T not in patch_effects[L]:
                print(f"{'---':<7}", end="")
            else:
                effects = patch_effects[L][T]
                if not effects:
                    print(f"{'---':<7}", end="")
                    continue
                mean_eff = float(np.mean(effects))
                std_eff = float(np.std(effects))
                key = f"L{L:02d}_T{T:02d}"
                results["patch_effects"][key] = {
                    "mean": mean_eff,
                    "std": std_eff,
                    "n": len(effects),
                }
                print(f"{mean_eff:<7.3f}", end="")
        print()

    return results


def main():
    config = load_config("configs/depth.yaml")
    results = run_patching(config)

    out_path = Path(config["results"]["metrics_dir"]) / "patching_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
