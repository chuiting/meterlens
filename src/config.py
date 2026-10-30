"""Loads config.yaml and exposes it as a dict. All other modules import from here."""

from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from config.yaml relative to the project root."""
    return PROJECT_ROOT / relative_path


CONFIG = load_config()

if __name__ == "__main__":
    import json
    print(json.dumps(CONFIG, indent=2))
