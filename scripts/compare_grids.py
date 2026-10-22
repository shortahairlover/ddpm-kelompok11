"""Susun beberapa gambar grid berdampingan dengan judul (perbandingan eksperimen).

Contoh (dari root repo):
    python -m scripts.compare_grids a.png b.png --titles "w=0" "w=3" \
        --out reports/figures/cmp.png --suptitle "Efek guidance"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def compare(paths: list[Path], titles: list[str], out: Path, suptitle: str | None = None,
            ncols: int | None = None) -> Path:
    """Tampilkan setiap gambar sebagai panel dengan judulnya; simpan ke `out`."""
    ncols = ncols or len(paths)
    nrows = -(-len(paths) // ncols)  # pembulatan ke atas
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.9 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, path, title in zip(axes.flat, paths, titles, strict=False):
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=10)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Bandingkan beberapa grid gambar")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--titles", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suptitle", type=str)
    parser.add_argument("--ncols", type=int)
    args = parser.parse_args(argv)
    if len(args.titles) != len(args.images):
        parser.error("jumlah --titles harus sama dengan jumlah gambar")
    out = compare(args.images, args.titles, args.out, args.suptitle, args.ncols)
    print(f"perbandingan disimpan: {out}")
    return out


if __name__ == "__main__":
    main()
