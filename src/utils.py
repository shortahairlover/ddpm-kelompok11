"""Utilitas umum: seed, device, config YAML, direktori run, simpan grid gambar, logger."""

from __future__ import annotations

import copy
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torchvision.utils import save_image


class Config(dict):
    """Dict bersarang yang juga bisa diakses sebagai atribut (cfg.train.lr)."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = Config(value) if isinstance(value, dict) else value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(f"config tidak punya kunci '{name}'") from err

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = Config(value) if isinstance(value, dict) else value

    def __deepcopy__(self, memo: dict) -> Config:
        return Config(copy.deepcopy(self.to_dict(), memo))

    def to_dict(self) -> dict:
        """Kembalikan dict biasa (rekursif), misal untuk disimpan ke YAML."""
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self.items()}


def _deep_merge(base: dict, override: dict) -> dict:
    """Gabungkan override ke base secara rekursif (override menang)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_override(cfg: dict, item: str) -> None:
    """Terapkan override bentuk 'a.b.c=nilai'; nilai di-parse sebagai YAML (angka/bool/list)."""
    if "=" not in item:
        raise ValueError(f"override harus berbentuk kunci=nilai, dapat: '{item}'")
    dotted, raw = item.split("=", 1)
    keys = dotted.strip().split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            raise KeyError(f"kunci override tidak dikenal: '{dotted}'")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"kunci override tidak dikenal: '{dotted}'")
    node[keys[-1]] = _parse_value(raw)


def _parse_value(raw: str) -> Any:
    """Parse nilai override via YAML; '1e-4' (dibaca string oleh PyYAML) diubah ke float."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Config:
    """Baca config YAML.

    Kalau file punya kunci `base`, file base dibaca dulu (path relatif terhadap working
    directory, atau terhadap folder file ini) lalu ditimpa oleh isi file ini.
    `overrides` berbentuk ["train.lr=1e-4", ...] diterapkan paling akhir.
    """
    path = Path(path)
    with open(path, encoding="utf8") as f:
        data = yaml.safe_load(f) or {}

    base = data.pop("base", None)
    if base is not None:
        base_path = Path(base)
        if not base_path.is_absolute() and not base_path.exists():
            base_path = path.parent / base
        data = _deep_merge(load_config(base_path).to_dict(), data)

    for item in overrides or []:
        _apply_override(data, item)
    return Config(data)


def save_config(cfg: Config | dict, path: str | Path) -> None:
    """Simpan config ke YAML."""
    data = cfg.to_dict() if isinstance(cfg, Config) else cfg
    with open(path, "w", encoding="utf8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set seed Python, NumPy, dan PyTorch (CPU & CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    """'auto' -> cuda kalau tersedia, selain itu cpu. 'cuda'/'cpu' dipaksa."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device cuda diminta tetapi CUDA tidak tersedia")
    return torch.device(preference)


def make_run_dir(cfg: Config, root: str | Path = "runs", name: str | None = None) -> Path:
    """Buat runs/<nama>_<YYYYmmdd-HHMMSS>/ berisi checkpoints/, samples/, dan salinan config."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"{name or cfg.name}_{stamp}"
    run_dir = Path(root) / base_name
    suffix = 1
    while run_dir.exists():  # hindari tabrakan kalau dua run dibuat di detik yang sama
        run_dir = Path(root) / f"{base_name}-{suffix}"
        suffix += 1
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "samples").mkdir()
    save_config(cfg, run_dir / "config.yaml")
    return run_dir


def save_image_grid(images: torch.Tensor, path: str | Path, nrow: int = 8) -> None:
    """Simpan batch gambar sebagai grid PNG.

    images: (B, C, H, W) dalam rentang [-1, 1] (keluaran model / data ternormalisasi).
    """
    images = (images.detach().float().cpu().clamp(-1, 1) + 1) / 2  # -> [0, 1]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_image(images, path, nrow=nrow, padding=2)


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Logger ke konsol (dan ke file kalau diberikan) dengan format waktu."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    has_console = any(
        type(h) is logging.StreamHandler and h.stream is sys.stdout for h in logger.handlers
    )
    if not has_console:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)
    if log_file is not None:
        log_file = str(Path(log_file).resolve())
        if not any(getattr(h, "baseFilename", None) == log_file for h in logger.handlers):
            file_handler = logging.FileHandler(log_file, encoding="utf8")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
    return logger
