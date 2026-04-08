"""Evaluate all trained probes on the test set."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import load_config
from src.utils.device import get_device
from src.evaluation.metrics import evaluate_all_probes, compute_baseline_metrics


def main():
    config = load_config()
    device = get_device()
    print(f"Using device: {device}")

    print("\nEvaluating baselines...")
    baseline_metrics = compute_baseline_metrics(config)
    for name, metrics in baseline_metrics.items():
        print(f"  {name}: {metrics}")

    print("\nEvaluating trained probes on test set...")
    results = evaluate_all_probes(config, device)

    # Save results
    metrics_dir = Path(config["results"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Convert layer keys to strings for JSON serialization
    serializable = {}
    for mt in results:
        serializable[mt] = {}
        for pt in results[mt]:
            serializable[mt][pt] = {str(k): v for k, v in results[mt][pt].items()}

    with open(metrics_dir / "test_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    with open(metrics_dir / "baseline_results.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2)

    print(f"\nResults saved to {metrics_dir}")


if __name__ == "__main__":
    main()
