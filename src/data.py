"""Data pipeline MNIST & CIFAR-10: transform, dataset, dataloader, denormalisasi.

Konvensi (PLANNING.md Bagian D): gambar 32x32, nilai di [-1, 1], label 0-9.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from src.utils import Config

ROOT = Path(__file__).resolve().parents[1]

MNIST_CLASSES = [str(i) for i in range(10)]
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]  # fmt: skip


def class_names(dataset: str) -> list[str]:
    """Nama kelas sesuai urutan label torchvision."""
    return {"mnist": MNIST_CLASSES, "cifar10": CIFAR10_CLASSES}[dataset]


def build_transform(data_cfg: Config, train: bool = True) -> transforms.Compose:
    """Transform PIL -> tensor (C, 32, 32) di [-1, 1].

    Normalize(mean=0.5, std=0.5) setara x * 2 - 1, dipakai (bukan Lambda) agar transform
    bisa di-pickle oleh worker DataLoader di Windows.
    """
    steps: list = []
    if data_cfg.dataset == "mnist":
        steps.append(transforms.Pad(2))  # 28x28 -> 32x32 (latar hitam = 0)
    if train and data_cfg.hflip:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    steps.append(transforms.ToTensor())  # (C, H, W) di [0, 1]
    channels = data_cfg.channels
    steps.append(transforms.Normalize(mean=[0.5] * channels, std=[0.5] * channels))
    return transforms.Compose(steps)


def _data_root(data_cfg: Config) -> Path:
    """Folder data; path relatif dihitung dari root repo, bukan working directory."""
    root = Path(data_cfg.root)
    return root if root.is_absolute() else ROOT / root


def get_dataset(cfg: Config, train: bool = True, debug: bool = False) -> Dataset:
    """Dataset torchvision (diunduh otomatis saat pertama kali).

    debug=True -> subset acak tetap (seed dari config) sebanyak cfg.data.debug_subset gambar.
    Setiap item: (image (C, 32, 32) float di [-1, 1], label int 0-9).
    """
    data_cfg = cfg.data
    root = _data_root(data_cfg)
    transform = build_transform(data_cfg, train=train)
    if data_cfg.dataset == "mnist":
        dataset = datasets.MNIST(root, train=train, download=True, transform=transform)
    elif data_cfg.dataset == "cifar10":
        dataset = datasets.CIFAR10(root, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"dataset tidak dikenal: {data_cfg.dataset!r} (mnist | cifar10)")

    if debug:
        n = min(data_cfg.debug_subset, len(dataset))
        generator = torch.Generator().manual_seed(cfg.seed)
        indices = torch.randperm(len(dataset), generator=generator)[:n].tolist()
        dataset = Subset(dataset, indices)
    return dataset


def get_dataloader(
    cfg: Config,
    train: bool = True,
    debug: bool = False,
    num_workers: int | None = None,
) -> DataLoader:
    """DataLoader dengan pengaturan dari config.

    Di Windows, pemanggil yang memakai num_workers > 0 wajib berada di dalam
    `if __name__ == "__main__":` (worker dibuat dengan metode spawn).
    """
    data_cfg = cfg.data
    workers = data_cfg.num_workers if num_workers is None else num_workers
    dataset = get_dataset(cfg, train=train, debug=debug)
    return DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train and len(dataset) >= data_cfg.batch_size,
        persistent_workers=workers > 0,
    )


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] (dipotong ke rentang valid). x: (..., C, H, W)."""
    return ((x + 1) / 2).clamp(0, 1)
