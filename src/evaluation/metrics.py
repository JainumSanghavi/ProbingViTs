"""Evaluation metrics for probe performance."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    average_precision_score, precision_recall_curve,
)


def evaluate_probe(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict:
    """Evaluate a trained probe on a DataLoader.

    Returns dict with accuracy, f1, precision, recall, average_precision,
    plus raw logits and labels for PR curve plotting.
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            logits = model(features)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Sigmoid for probability scores
    probs = 1 / (1 + np.exp(-all_logits))
    predictions = (probs >= threshold).astype(np.float32)

    metrics = {
        "accuracy": float(accuracy_score(all_labels, predictions)),
        "f1": float(f1_score(all_labels, predictions, zero_division=0)),
        "precision": float(precision_score(all_labels, predictions, zero_division=0)),
        "recall": float(recall_score(all_labels, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(all_labels, probs)),
    }

    return metrics, probs, all_labels


def evaluate_all_probes(
    config: dict,
    device: torch.device,
) -> Dict:
    """Evaluate all trained probes on the test set.

    Loads checkpoints and evaluates on test data.
    Returns nested dict: results[model_type][probe_type][layer] = metrics
    """
    from src.data.hidden_state_dataset import PatchLevelDataModule
    from src.probes.linear_probe import get_probe

    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states"
    labels_dir = Path(config["dataset"]["processed_dir"]) / "patch_labels"
    checkpoints_dir = Path(config["results"]["checkpoints_dir"])

    layers = config["training"]["layers"]
    probe_types = config["training"]["probe_types"]
    model_types = config["extraction"]["models"]

    results = {}

    for model_type in model_types:
        results[model_type] = {}
        for probe_type in probe_types:
            results[model_type][probe_type] = {}
            for layer in layers:
                run_name = f"{model_type}_{probe_type}_layer{layer:02d}"
                checkpoint_path = checkpoints_dir / f"{run_name}.pt"

                if not checkpoint_path.exists():
                    print(f"  Skipping {run_name} (no checkpoint)")
                    continue

                # Build data module for this layer
                data_module = PatchLevelDataModule(
                    cached_dir=str(cached_dir),
                    labels_dir=str(labels_dir),
                    model_type=model_type,
                    layer=layer,
                    batch_size=config["training"]["batch_size"],
                    num_workers=config["num_workers"],
                )

                # Create probe and load checkpoint
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
                state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                probe.load_state_dict(state_dict)
                probe.to(device)

                # Evaluate
                test_loader = data_module.test_dataloader()
                metrics, probs, labels = evaluate_probe(
                    probe, test_loader, device,
                    threshold=config["evaluation"]["threshold"],
                )

                results[model_type][probe_type][layer] = metrics
                print(f"  {run_name}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")

    return results


def compute_baseline_metrics(config: dict) -> Dict:
    """Compute baseline metrics on test set."""
    from src.data.hidden_state_dataset import HiddenStateDataset
    from src.probes.baselines import MajorityClassBaseline, RandomBaseline

    labels_dir = Path(config["dataset"]["processed_dir"]) / "patch_labels"
    cached_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states"

    # Load train and test labels
    train_dataset = HiddenStateDataset(
        hidden_dir=str(cached_dir / "pretrained" / "train"),
        labels_dir=str(labels_dir / "train"),
        layer=0,
    )
    test_dataset = HiddenStateDataset(
        hidden_dir=str(cached_dir / "pretrained" / "test"),
        labels_dir=str(labels_dir / "test"),
        layer=0,
    )

    train_labels = train_dataset.get_all_labels()
    test_labels = test_dataset.get_all_labels()

    # Majority class baseline
    majority = MajorityClassBaseline().fit(train_labels)
    majority_metrics = majority.evaluate(test_labels)

    # Random baseline
    random = RandomBaseline(seed=config["seed"]).fit(train_labels)
    random_metrics = random.evaluate(test_labels)

    return {
        "majority_class": majority_metrics,
        "random": random_metrics,
    }
