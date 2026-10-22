"""Tampilkan & periksa satu batch data (T2.6).

Jalankan dari root repo:
    python -m scripts.show_batch                 # MNIST dan CIFAR-10
    python -m scripts.show_batch --dataset mnist

Menyimpan grid ke reports/figures/batch_<dataset>.png dan mencetak shape, dtype,
rentang nilai, serta distribusi label batch.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.data import class_names, get_dataloader
from src.utils import load_config, save_image_grid, set_seed

ROOT = Path(__file__).resolve().parents[1]


def inspect(dataset: str, out_dir: Path) -> Path:
    """Ambil 1 batch, cetak statistik, simpan grid 8x8. Return path gambar."""
    cfg = load_config(ROOT / "configs" / f"{dataset}.yaml")
    set_seed(cfg.seed)
    loader = get_dataloader(cfg, train=True)
    images, labels = next(iter(loader))  # images: (B, C, 32, 32), labels: (B,)

    names = class_names(dataset)
    counts = Counter(labels.tolist())
    print(f"== {dataset} ==")
    print(f"jumlah data train : {len(loader.dataset)}")
    print(f"images            : shape={tuple(images.shape)} dtype={images.dtype}")
    print(f"rentang nilai     : min={images.min().item():.3f} max={images.max().item():.3f}")
    print(f"labels            : shape={tuple(labels.shape)} dtype={labels.dtype}")
    print("distribusi label  : " + ", ".join(f"{names[k]}={counts[k]}" for k in sorted(counts)))

    out_path = out_dir / f"batch_{dataset}.png"
    save_image_grid(images[:64], out_path, nrow=8)
    print(f"grid disimpan     : {out_path.relative_to(ROOT)}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "all"], default="all")
    args = parser.parse_args()
    out_dir = ROOT / "reports" / "figures"
    targets = ["mnist", "cifar10"] if args.dataset == "all" else [args.dataset]
    for dataset in targets:
        inspect(dataset, out_dir)


if __name__ == "__main__":
    main()
