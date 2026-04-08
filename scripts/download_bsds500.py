"""Download and extract BSDS500 dataset."""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import load_config


def download_bsds500(data_dir: str):
    """Download BSDS500 from Berkeley mirror."""
    data_dir = Path(data_dir)

    # Check if already downloaded
    images_dir = data_dir / "data" / "images" / "train"
    if images_dir.exists() and len(list(images_dir.glob("*.jpg"))) >= 200:
        print("BSDS500 already downloaded and verified.")
        return

    data_dir.mkdir(parents=True, exist_ok=True)

    # Download from Berkeley BSDS500 mirror
    url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
    archive_path = data_dir.parent / "BSR_bsds500.tgz"

    if not archive_path.exists():
        print(f"Downloading BSDS500 from {url}...")
        print("(This may take a few minutes)")
        urllib.request.urlretrieve(url, archive_path, _progress_hook)
        print("\nDownload complete.")
    else:
        print(f"Archive already exists at {archive_path}")

    # Extract
    print("Extracting archive...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(data_dir.parent)

    # The archive extracts to BSR/BSDS500/
    # Move contents to expected location
    extracted_dir = data_dir.parent / "BSR" / "BSDS500"
    if extracted_dir.exists():
        # Move data directory contents
        import shutil
        if data_dir.exists():
            shutil.rmtree(data_dir)
        shutil.move(str(extracted_dir), str(data_dir))
        # Clean up BSR directory
        bsr_dir = data_dir.parent / "BSR"
        if bsr_dir.exists():
            shutil.rmtree(bsr_dir)

    # Clean up archive
    if archive_path.exists():
        archive_path.unlink()

    # Verify
    verify_bsds500(data_dir)


def verify_bsds500(data_dir: str):
    """Verify BSDS500 dataset integrity."""
    data_dir = Path(data_dir)

    expected = {"train": 200, "val": 100, "test": 200}

    for split, count in expected.items():
        img_dir = data_dir / "data" / "images" / split
        gt_dir = data_dir / "data" / "groundTruth" / split

        images = list(img_dir.glob("*.jpg"))
        gts = list(gt_dir.glob("*.mat"))

        print(f"  {split}: {len(images)} images, {len(gts)} ground truth files")

        if len(images) < count:
            print(f"  WARNING: Expected {count} images in {split}, found {len(images)}")
        if len(gts) < count:
            print(f"  WARNING: Expected {count} .mat files in {split}, found {len(gts)}")

    print("Verification complete.")


def _progress_hook(count, block_size, total_size):
    """Progress bar for urllib download."""
    percent = int(count * block_size * 100 / total_size)
    percent = min(percent, 100)
    sys.stdout.write(f"\r  Progress: {percent}%")
    sys.stdout.flush()


if __name__ == "__main__":
    config = load_config()
    download_bsds500(config["dataset"]["raw_dir"])
