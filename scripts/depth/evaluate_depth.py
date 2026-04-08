"""Evaluate all trained depth probes on the test set."""

import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import load_config
from src.utils.device import get_device
from src.data.depth_dataset import DepthDataModule
from src.probes.linear_probe import get_probe
from src.training.depth_trainer import DepthProbeTrainer


def evaluate_all_depth_probes(config: dict, device: torch.device) -> dict:
    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states"
    labels_dir = Path(config["dataset"]["processed_dir"]) / "depth_labels"
    checkpoints_dir = Path(config["results"]["checkpoints_dir"])

    layers = config["training"]["layers"]
    probe_types = config["training"]["probe_types"]
    model_types = config["extraction"]["models"]

    results = {}

    for model_type in model_types:
        results[model_type] = {}
        for probe_type in probe_types:
            results[model_type][probe_type] = {}

            for layer in tqdm(layers, desc=f"{model_type}/{probe_type}"):
                run_name = f"{model_type}_{probe_type}_layer{layer:02d}"
                checkpoint_path = checkpoints_dir / f"{run_name}.pt"

                if not checkpoint_path.exists():
                    print(f"  Warning: checkpoint not found for {run_name}")
                    continue

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

                probe = get_probe(probe_type, input_dim=config["model"]["hidden_dim"], **probe_kwargs)
                state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                probe.load_state_dict(state_dict)

                trainer = DepthProbeTrainer(model=probe, device=device)
                metrics = trainer.evaluate_test(data_module.test_dataloader())

                results[model_type][probe_type][layer] = metrics

    return results


def main():
    config = load_config("configs/depth.yaml")
    device = get_device()
    print(f"Using device: {device}")

    print("\nEvaluating all depth probes on test set...")
    results = evaluate_all_depth_probes(config, device)

    metrics_dir = Path(config["results"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for mt in results:
        serializable[mt] = {}
        for pt in results[mt]:
            serializable[mt][pt] = {str(k): v for k, v in results[mt][pt].items()}

    with open(metrics_dir / "depth_test_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {metrics_dir / 'depth_test_results.json'}")

    print("\nBest test MAE per model/probe type (pretrained ViT):")
    for pt in results.get("pretrained", {}):
        if results["pretrained"][pt]:
            best_layer = min(results["pretrained"][pt], key=lambda l: results["pretrained"][pt][l]["mae"])
            best_mae = results["pretrained"][pt][best_layer]["mae"]
            max_depth = config["depth_labels"]["max_depth"]
            print(f"  {pt}: layer {best_layer}, MAE={best_mae:.4f} ({best_mae*max_depth:.3f}m)")


if __name__ == "__main__":
    main()
