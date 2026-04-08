"""Download NYU Depth V2 via HuggingFace Hub and save images/depth maps to disk."""

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import load_config


def _stream_and_save(split_name, hf_split, max_items, raw_dir):
    """Stream items from a HuggingFace split and save to disk."""
    img_dir = raw_dir / "images" / split_name
    depth_dir = raw_dir / "depth" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving {split_name}: up to {max_items} images...")

    saved = 0
    for idx, item in enumerate(tqdm(hf_split, desc=split_name, total=max_items)):
        if idx >= max_items:
            break

        image_id = f"{idx:05d}"
        img_path = img_dir / f"{image_id}.png"
        depth_path = depth_dir / f"{image_id}.npy"

        if not (img_path.exists() and depth_path.exists()):
            item["image"].save(img_path)
            depth_array = np.array(item["depth_map"], dtype=np.float32)
            np.save(depth_path, depth_array)

        saved += 1

    return saved


def download_nyu(config: dict):
    """Download NYU Depth V2 and save RGB images + depth maps to disk."""
    from datasets import load_dataset

    raw_dir = Path(config["dataset"]["raw_dir"])
    val_size = config["depth_labels"]["val_split_size"]
    train_size = config["dataset"]["splits"]["train"]
    test_size = config["dataset"]["splits"]["test"]

    print("Loading NYU Depth V2 from HuggingFace Hub (streaming)...")
    print("(Downloads images on-the-fly; cached by HuggingFace locally after that)")

    # Stream HF train split: first val_size go to val, next train_size go to train
    hf_train = load_dataset(
        "sayakpaul/nyu_depth_v2", split="train", streaming=True
    )

    # --- val split: indices 0 .. val_size-1 ---
    val_iter = hf_train.take(val_size)
    val_count = _stream_and_save("val", val_iter, val_size, raw_dir)

    # --- train split: indices val_size .. val_size+train_size-1 ---
    train_iter = hf_train.skip(val_size).take(train_size)
    train_count = _stream_and_save("train", train_iter, train_size, raw_dir)

    # --- test split: HF validation ---
    hf_test = load_dataset(
        "sayakpaul/nyu_depth_v2", split="validation", streaming=True
    )
    test_iter = hf_test.take(test_size)
    test_count = _stream_and_save("test", test_iter, test_size, raw_dir)

    print("\nDownload complete!")
    print(f"  val: {val_count} images")
    print(f"  train: {train_count} images")
    print(f"  test: {test_count} images")


if __name__ == "__main__":
    config = load_config("configs/depth.yaml")
    download_nyu(config)
