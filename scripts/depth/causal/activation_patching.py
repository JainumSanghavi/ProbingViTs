"""Activation patching study for depth probing.

Tests whether depth information at layer L causally propagates to downstream
layers by replacing activations from a 'source' image with those from a
'destination' image at layer L and measuring the downstream depth prediction shift.

Produces a 12x12 lower-triangular matrix of patch effects: rows = intervention
layer L (1-12), columns = target layer T (L..12).
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
    """Select image pairs with maximum depth contrast from test set.
    Returns list of (source_id, dest_id) tuples.
    """
    labels_dir = Path(config["dataset"]["processed_dir"]) / "depth_labels" / "test"
    ids_and_means = []
    for p in sorted(labels_dir.glob("*.npy")):
        labels = np.load(p)
        ids_and_means.append((p.stem, float(labels.mean())))

    ids_and_means.sort(key=lambda x: x[1])
    near_ids = [x[0] for x in ids_and_means[:n_near]]
    far_ids = [x[0] for x in ids_and_means[-n_far:]]

    # Pair each near with each far (take first 20 pairs)
    pairs = []
    for n_id in near_ids:
        for f_id in far_ids:
            pairs.append((n_id, f_id))
            if len(pairs) >= 20:
                break
        if len(pairs) >= 20:
            break
    return pairs


def _load_probes(config: dict) -> dict:
    """Load pretrained linear probe checkpoints for all layers."""
    checkpoints_dir = Path(config["results"]["checkpoints_dir"])
    probes = {}
    for layer in config["training"]["layers"]:
        ckpt = checkpoints_dir / f"pretrained_linear_layer{layer:02d}.pt"
        if ckpt.exists():
            probe = get_probe("linear", input_dim=config["model"]["hidden_dim"])
            state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
            probe.load_state_dict(state)
            probe.eval()
            probes[layer] = probe
    return probes


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

    # Select pairs
    pairs = _select_pairs(config)
    print(f"Selected {len(pairs)} (near, far) image pairs")

    # Load probes
    probes = _load_probes(config)
    print(f"Loaded {len(probes)} probe checkpoints")

    # Create extractor
    extractor = ViTExtractor(
        model_name=config["model"]["name"],
        pretrained=True,
        device=device,
    )

    layers = config["training"]["layers"]  # [0, 1, ..., 12]
    intervention_layers = [l for l in layers if l >= 1]  # can't hook embedding (layer 0)

    # Results: patch_effects[L][T] = list of effects across all pairs
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

        # Get destination baseline predictions at each layer
        dest_preds = {}
        source_preds = {}
        for T in layers:
            if T in probes:
                dest_preds[T] = _predict_depth(probes[T], dest_hidden[T])
                source_preds[T] = _predict_depth(probes[T], source_hidden[T])

        # For each intervention layer L
        for L in intervention_layers:
            # Build hook that replaces patch tokens at layer L
            # encoder.layer[L-1] produces what we call "layer L"
            dest_patches_L = dest_hidden[L].float().to(device)  # (196, 768)

            def make_hook(dest_patches):
                def hook_fn(module, input, output):
                    # output is a tuple: (hidden_states, ...) where hidden_states is (B, 197, 768)
                    modified = output[0].clone()
                    modified[:, 1:, :] = dest_patches.unsqueeze(0)  # replace 196 patch tokens
                    return (modified,) + output[1:]
                return hook_fn

            hook = make_hook(dest_patches_L)
            patched_hidden = extractor.extract_with_hook(source_pixels, hook, layer_idx=L-1).cpu()

            # Measure patch effect at each target layer T >= L
            for T in layers:
                if T < L or T not in probes:
                    continue

                patched_pred = _predict_depth(probes[T], patched_hidden[T])
                source_pred = source_preds[T]
                dest_pred = dest_preds[T]

                # Patch effect: how much did prediction shift toward destination?
                shift = float(np.abs(patched_pred - source_pred).mean())
                total_gap = float(np.abs(dest_pred - source_pred).mean())
                effect = shift / max(total_gap, 1e-8)

                patch_effects[L][T].append(effect)

    # Aggregate across pairs
    results = {
        "pairs": [{"source": s, "dest": d} for s, d in pairs],
        "n_pairs": len(pairs),
        "patch_effects": {},
    }

    print("\n=== PATCH EFFECT MATRIX (mean across pairs) ===")
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
