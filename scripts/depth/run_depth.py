"""End-to-end depth probing pipeline: download → preprocess → extract → train → evaluate → visualize."""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run full depth probing pipeline")
    parser.add_argument("--config", default="configs/depth.yaml", help="Config file path")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip label preprocessing")
    parser.add_argument("--skip-extract", action="store_true", help="Skip feature extraction")
    parser.add_argument("--skip-train", action="store_true", help="Skip probe training")
    args = parser.parse_args()

    config = load_config(args.config)
    total_start = time.time()

    if not args.skip_download:
        print("\n" + "=" * 60)
        print("STEP 1: Downloading NYU Depth V2")
        print("=" * 60)
        from scripts.depth.download_nyu import download_nyu
        download_nyu(config)

    if not args.skip_preprocess:
        print("\n" + "=" * 60)
        print("STEP 2: Preprocessing depth labels")
        print("=" * 60)
        from scripts.depth.preprocess_depth_labels import preprocess_all_depth_labels
        preprocess_all_depth_labels(config)

    if not args.skip_extract:
        print("\n" + "=" * 60)
        print("STEP 3: Extracting ViT hidden states")
        print("=" * 60)
        from scripts.depth.extract_hidden_states_depth import extract_hidden_states
        extract_hidden_states(config)

    if not args.skip_train:
        print("\n" + "=" * 60)
        print("STEP 4: Training depth probes")
        print("=" * 60)
        torch.manual_seed(config["seed"])
        from scripts.depth.train_probes_depth import train_all_depth_probes
        train_all_depth_probes(config)

    print("\n" + "=" * 60)
    print("STEP 5: Evaluating probes")
    print("=" * 60)
    from scripts.depth.evaluate_depth import main as evaluate_main
    evaluate_main()

    print("\n" + "=" * 60)
    print("STEP 6: Generating visualizations")
    print("=" * 60)
    from scripts.depth.visualize_depth import main as visualize_main
    visualize_main()

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"DEPTH PIPELINE COMPLETE in {elapsed / 60:.1f} minutes")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
