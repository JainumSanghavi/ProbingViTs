"""YAML config loader."""

import yaml
from pathlib import Path


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load YAML configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent.parent
