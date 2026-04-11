"""Run the full causal analysis pipeline."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run causal depth probing analysis")
    parser.add_argument("--config", default="configs/depth.yaml")
    parser.add_argument("--skip-patching", action="store_true", help="Skip activation patching (Approach B)")
    args = parser.parse_args()

    total_start = time.time()

    print("\n" + "=" * 60)
    print("STEP 1: Probe direction ablation")
    print("=" * 60)
    from scripts.depth.causal.ablation_study import main as ablation_main
    ablation_main()

    if not args.skip_patching:
        print("\n" + "=" * 60)
        print("STEP 2: Activation patching")
        print("=" * 60)
        from scripts.depth.causal.activation_patching import main as patching_main
        patching_main()

    print("\n" + "=" * 60)
    print("STEP 3: Generating causal figures")
    print("=" * 60)
    from scripts.depth.causal.visualize_causal import main as viz_main
    viz_main()

    elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"CAUSAL ANALYSIS COMPLETE in {elapsed / 60:.1f} minutes")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
