"""Generate all visualization figures."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import load_config
from src.utils.device import get_device
from src.visualization.layerwise_plots import plot_layerwise_metrics, plot_layerwise_comparison
from src.visualization.pr_curves import plot_pr_curves
from src.visualization.qualitative import visualize_predictions, visualize_heatmap_grid


def main():
    config = load_config()
    device = get_device()

    metrics_dir = Path(config["results"]["metrics_dir"])
    figures_dir = Path(config["results"]["figures_dir"])

    # Load test results
    test_results_path = metrics_dir / "test_results.json"
    baseline_results_path = metrics_dir / "baseline_results.json"

    if not test_results_path.exists():
        print("No test results found. Run scripts/evaluate.py first.")
        return

    with open(test_results_path) as f:
        results_raw = json.load(f)

    # Convert string layer keys back to ints
    results = {}
    for mt in results_raw:
        results[mt] = {}
        for pt in results_raw[mt]:
            results[mt][pt] = {int(k): v for k, v in results_raw[mt][pt].items()}

    baseline_metrics = {}
    if baseline_results_path.exists():
        with open(baseline_results_path) as f:
            baseline_metrics = json.load(f)

    # Generate plots
    print("Generating layerwise plots...")
    plot_layerwise_metrics(results, baseline_metrics, str(figures_dir))

    print("Generating comparison plots...")
    plot_layerwise_comparison(results, str(figures_dir))

    print("Generating PR curves...")
    plot_pr_curves(config, device, save_dir=str(figures_dir))

    print("Generating qualitative visualizations...")
    visualize_predictions(config, device, save_dir=str(figures_dir))

    print("Generating heatmap grid...")
    visualize_heatmap_grid(config, device, save_dir=str(figures_dir))

    print(f"\nAll figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
