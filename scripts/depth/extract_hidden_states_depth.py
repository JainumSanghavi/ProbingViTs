"""Extract and cache ViT hidden states for all NYU Depth V2 images."""

import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import load_config
from src.utils.device import get_device
from src.models.vit_extractor import ViTExtractor
from src.data.transforms import get_vit_transform
from src.data.nyu_depth import NYUDepthDataset


def extract_hidden_states(config: dict):
    device = get_device()
    print(f"Using device: {device}")

    raw_dir = config["dataset"]["raw_dir"]
    cache_dir = Path(config["dataset"]["cached_dir"]) / "hidden_states"
    model_name = config["model"]["name"]
    use_fp16 = config["extraction"]["dtype"] == "float16"
    transform = get_vit_transform(config["model"]["image_size"])

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

        for split in ["train", "val", "test"]:
            dataset = NYUDepthDataset(raw_dir, split=split)
            save_dir = cache_dir / model_type / split
            save_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{split}: {len(dataset)} images")

            for idx in tqdm(range(len(dataset)), desc=f"{model_type}/{split}"):
                image, _, image_id = dataset[idx]
                save_path = save_dir / f"{image_id}.pt"

                if save_path.exists():
                    continue

                pixel_values = transform(image)
                hidden_states = extractor.extract_single(pixel_values)  # (13, 196, 768)

                if use_fp16:
                    hidden_states = hidden_states.half()

                torch.save(hidden_states.cpu(), save_path)

        del extractor
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nExtraction complete!")
    _print_cache_stats(cache_dir)


def _print_cache_stats(cache_dir: Path):
    total_size = 0
    total_files = 0
    for pt_file in cache_dir.rglob("*.pt"):
        total_size += pt_file.stat().st_size
        total_files += 1
    print(f"\nCache: {total_files} files, {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    config = load_config("configs/depth.yaml")
    extract_hidden_states(config)
