"""Train probes across all layers, probe types, and model types."""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import load_config
from src.utils.device import get_device
from src.data.hidden_state_dataset import PatchLevelDataModule
from src.probes.linear_probe import get_probe
from src.training.trainer import ProbeTrainer


def train_all_probes(config: dict):
    """Train probes for all layers, probe types, and model types."""
    device = get_device()
    print(f"Using device: {device}")

    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states"
    labels_dir = Path(config["dataset"]["processed_dir"]) / "patch_labels"
    metrics_dir = Path(config["results"]["metrics_dir"])
    checkpoints_dir = Path(config["results"]["checkpoints_dir"])

    layers = config["training"]["layers"]
    probe_types = config["training"]["probe_types"]
    model_types = config["extraction"]["models"]

    all_results = {}

    total_runs = len(layers) * len(probe_types) * len(model_types)
    run_idx = 0

    for model_type in model_types:
        for probe_type in probe_types:
            for layer in layers:
                run_idx += 1
                run_name = f"{model_type}_{probe_type}_layer{layer:02d}"
                print(f"\n{'='*60}")
                print(f"[{run_idx}/{total_runs}] {run_name}")
                print(f"{'='*60}")

                start_time = time.time()

                # Build data module
                data_module = PatchLevelDataModule(
                    cached_dir=str(cached_dir),
                    labels_dir=str(labels_dir),
                    model_type=model_type,
                    layer=layer,
                    batch_size=config["training"]["batch_size"],
                    num_workers=config["num_workers"],
                )

                # Get pos_weight for loss function
                pos_weight = data_module.get_pos_weight()
                print(f"  pos_weight: {pos_weight:.2f}")

                # Create probe
                probe_kwargs = {}
                if probe_type == "mlp":
                    probe_kwargs = {
                        "hidden_dim": config["mlp_probe"]["hidden_dim"],
                        "dropout": config["mlp_probe"]["dropout"],
                    }
                elif probe_type == "conv":
                    probe_kwargs = {
                        "hidden_dim": config["conv_probe"]["hidden_dim"],
                    }

                probe = get_probe(
                    probe_type,
                    input_dim=config["model"]["hidden_dim"],
                    **probe_kwargs,
                )

                num_params = sum(p.numel() for p in probe.parameters())
                print(f"  Probe params: {num_params:,}")

                # Train
                checkpoint_path = str(checkpoints_dir / f"{run_name}.pt")
                trainer = ProbeTrainer(
                    model=probe,
                    device=device,
                    lr=config["training"]["optimizer"]["lr"],
                    weight_decay=config["training"]["optimizer"]["weight_decay"],
                    pos_weight=pos_weight,
                    patience=config["training"]["early_stopping"]["patience"],
                    max_epochs=config["training"]["max_epochs"],
                )

                train_loader = data_module.train_dataloader()
                val_loader = data_module.val_dataloader()

                results = trainer.train(
                    train_loader, val_loader,
                    checkpoint_path=checkpoint_path,
                )

                elapsed = time.time() - start_time
                print(f"  Training time: {elapsed:.1f}s")
                print(f"  Best val F1: {results['best_val_f1']:.4f}")

                # Save results
                all_results[run_name] = {
                    "model_type": model_type,
                    "probe_type": probe_type,
                    "layer": layer,
                    "num_params": num_params,
                    "best_val_f1": results["best_val_f1"],
                    "epochs_trained": results["epochs_trained"],
                    "training_time": elapsed,
                }

    # Save all results
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results_path = metrics_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {results_path}")
    _print_summary(all_results)


def _print_summary(results: dict):
    """Print a summary table of results."""
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"{'Run':<40} {'Val F1':>8} {'Epochs':>8} {'Time':>8}")
    print("-" * 80)

    for name, r in sorted(results.items()):
        print(
            f"{name:<40} "
            f"{r['best_val_f1']:>8.4f} "
            f"{r['epochs_trained']:>8d} "
            f"{r['training_time']:>7.1f}s"
        )


if __name__ == "__main__":
    # Set seeds for reproducibility
    config = load_config()
    torch.manual_seed(config["seed"])

    train_all_probes(config)
