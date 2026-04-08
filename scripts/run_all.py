"""End-to-end orchestrator: download → preprocess → extract → train → evaluate → visualize."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Run full probing pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument("--skip-extract", action="store_true", help="Skip feature extraction")
    parser.add_argument("--skip-train", action="store_true", help="Skip probe training")
    args = parser.parse_args()

    config = load_config(args.config)
    total_start = time.time()

    # Step 1: Download BSDS500
    if not args.skip_download:
        print("\n" + "=" * 60)
        print("STEP 1: Downloading BSDS500")
        print("=" * 60)
        from scripts.download_bsds500 import download_bsds500
        download_bsds500(config["dataset"]["raw_dir"])

    # Step 2: Preprocess labels
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing patch labels")
    print("=" * 60)
    from scripts.preprocess_labels import preprocess_all_labels
    preprocess_all_labels(config)

    # Step 3: Extract hidden states
    if not args.skip_extract:
        print("\n" + "=" * 60)
        print("STEP 3: Extracting ViT hidden states")
        print("=" * 60)
        from scripts.extract_hidden_states import extract_hidden_states
        extract_hidden_states(config)

    # Step 4: Train probes
    if not args.skip_train:
        print("\n" + "=" * 60)
        print("STEP 4: Training probes")
        print("=" * 60)
        import torch
        torch.manual_seed(config["seed"])
        from scripts.train_probes import train_all_probes
        train_all_probes(config)

    # Step 5: Evaluate
    print("\n" + "=" * 60)
    print("STEP 5: Evaluating probes")
    print("=" * 60)
    from scripts.evaluate import main as evaluate_main
    evaluate_main()

    # Step 6: Visualize
    print("\n" + "=" * 60)
    print("STEP 6: Generating visualizations")
    print("=" * 60)
    from scripts.visualize import main as visualize_main
    visualize_main()

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE in {elapsed / 60:.1f} minutes")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
