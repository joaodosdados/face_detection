import json
import os
from pathlib import Path
from typing import Any


CONFIG_PATH = Path("config.json")


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid JSON in {config_path}: {error}") from error


def setup_cuda_paths(config: dict[str, Any]) -> None:
    paths_config = config.get("paths", {})

    for dll_path in (paths_config.get("cuda_path"), paths_config.get("cudnn_path")):
        if not dll_path:
            continue

        if os.path.exists(dll_path):
            os.add_dll_directory(dll_path)
            os.environ["PATH"] = dll_path + os.pathsep + os.environ["PATH"]
        else:
            print(f"[WARN] CUDA/cuDNN path not found: {dll_path}")


def ensure_runtime_dirs(config: dict[str, Any]) -> None:
    directories = [
        Path(config.get("paths", {}).get("reference_dir", "img/references")),
        Path(config.get("logging", {}).get("directory", "logs")),
        Path(config.get("cache", {}).get("path", "data/embeddings_cache.pkl")).parent,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
