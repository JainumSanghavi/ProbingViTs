"""Train depth regression probes across all layers, probe types, and model types."""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import load_config
from src.utils.device import get_device
from src.data.depth_dataset import DepthDataModule
from src.probes.linear_probe import get_probe
from src.training.depth_trainer import DepthProbeTrainer


def train_all_depth_probes(config: dict):
    device = get_device()
    print(f"Using device: {device}")

    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states"
    labels_dir = Path(config["dataset"]["processed_dir"]) / "depth_labels"
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

                data_module = DepthDataModule(
                    cached_dir=str(cached_dir),
                    labels_dir=str(labels_dir),
                    model_type=model_type,
                    layer=layer,
                    batch_size=config["training"]["batch_size"],
                    num_workers=config["num_workers"],
                )

                probe_kwargs = {}
                if probe_type == "mlp":
                    probe_kwargs = {
                        "hidden_dim": config["mlp_probe"]["hidden_dim"],
                        "dropout": config["mlp_probe"]["dropout"],
                    }

                probe = get_probe(
                    probe_type,
                    input_dim=config["model"]["hidden_dim"],
                    **probe_kwargs,
                )

                num_params = sum(p.numel() for p in probe.parameters())
                print(f"  Probe params: {num_params:,}")

                checkpoint_path = str(checkpoints_dir / f"{run_name}.pt")
                trainer = DepthProbeTrainer(
                    model=probe,
                    device=device,
                    lr=config["training"]["optimizer"]["lr"],
                    weight_decay=config["training"]["optimizer"]["weight_decay"],
                    patience=config["training"]["early_stopping"]["patience"],
                    max_epochs=config["training"]["max_epochs"],
                )

                results = trainer.train(
                    data_module.train_dataloader(),
                    data_module.val_dataloader(),
                    checkpoint_path=checkpoint_path,
                )

                elapsed = time.time() - start_time
                print(f"  Training time: {elapsed:.1f}s")
                print(f"  Best val MAE: {results['best_val_mae']:.4f}")

                all_results[run_name] = {
                    "model_type": model_type,
                    "probe_type": probe_type,
                    "layer": layer,
                    "num_params": num_params,
                    "best_val_mae": results["best_val_mae"],
                    "epochs_trained": results["epochs_trained"],
                    "training_time": elapsed,
                }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    results_path = metrics_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {results_path}")
    _print_summary(all_results)


def _print_summary(results: dict):
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"{'Run':<40} {'Val MAE':>8} {'Epochs':>8} {'Time':>8}")
    print("-" * 80)
    for name, r in sorted(results.items()):
        print(
            f"{name:<40} "
            f"{r['best_val_mae']:>8.4f} "
            f"{r['epochs_trained']:>8d} "
            f"{r['training_time']:>7.1f}s"
        )


if __name__ == "__main__":
    config = load_config("configs/depth.yaml")
    torch.manual_seed(config["seed"])
    train_all_depth_probes(config)
