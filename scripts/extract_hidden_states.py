"""Extract and cache ViT hidden states for all BSDS500 images."""

import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.config import load_config
from src.utils.device import get_device
from src.models.vit_extractor import ViTExtractor
from src.data.transforms import get_vit_transform


def extract_hidden_states(config: dict):
    """Extract and cache hidden states for all images."""
    device = get_device()
    print(f"Using device: {device}")

    raw_dir = Path(config["dataset"]["raw_dir"])
    cache_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states"
    model_name = config["model"]["name"]
    use_fp16 = config["extraction"]["dtype"] == "float16"

    for model_type in config["extraction"]["models"]:
        pretrained = (model_type == "pretrained")
        print(f"\n{'='*60}")
        print(f"Extracting with {model_type} ViT...")
        print(f"{'='*60}")

        extractor = ViTExtractor(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
        )

        transform = get_vit_transform(config["model"]["image_size"])

        for split in ["train", "val", "test"]:
            img_dir = raw_dir / "data" / "images" / split
            save_dir = cache_dir / model_type / split
            save_dir.mkdir(parents=True, exist_ok=True)

            image_paths = sorted(img_dir.glob("*.jpg"))
            print(f"\n{split}: {len(image_paths)} images")

            for img_path in tqdm(image_paths, desc=f"{model_type}/{split}"):
                image_id = img_path.stem
                save_path = save_dir / f"{image_id}.pt"

                # Resume-safe: skip already cached
                if save_path.exists():
                    continue

                # Load and preprocess image
                image = Image.open(img_path).convert("RGB")
                pixel_values = transform(image)  # (3, 224, 224)

                # Extract hidden states
                hidden_states = extractor.extract_single(pixel_values)  # (13, 196, 768)

                # Convert to float16 for storage efficiency
                if use_fp16:
                    hidden_states = hidden_states.half()

                # Save to disk (move to CPU first)
                torch.save(hidden_states.cpu(), save_path)

        # Free GPU memory before loading next model
        del extractor
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    print("\nExtraction complete!")
    _print_cache_stats(cache_dir)


def _print_cache_stats(cache_dir: Path):
    """Print cache size statistics."""
    total_size = 0
    total_files = 0
    for pt_file in cache_dir.rglob("*.pt"):
        total_size += pt_file.stat().st_size
        total_files += 1

    print(f"\nCache statistics:")
    print(f"  Total files: {total_files}")
    print(f"  Total size: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    config = load_config()
    extract_hidden_states(config)
