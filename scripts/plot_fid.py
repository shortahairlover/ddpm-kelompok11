"""Grafik hasil FID dari reports/fid_results.csv (T6.2).

Jalankan dari root repo:
    python -m scripts.plot_fid

Output (reports/figures/):
    fid_experiments_epoch30.png : FID E1-E5 pada epoch 30 (w = 3, DDIM-100, 5.000 sampel)
    fid_guidance_e1.png         : FID vs guidance scale untuk E1 epoch 100
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP_NAMES = {"E1": "E1\nbaseline", "E2": "E2\nlr 1e-4", "E3": "E3\nlr 4e-4",
             "E4": "E4\nbatch 64", "E5": "E5\ncosine"}


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def select(rows, **conds) -> list[dict]:
    """Baris yang semua kolomnya cocok dengan conds (dibandingkan sebagai string/float)."""
    def ok(row):
        for key, value in conds.items():
            if isinstance(value, float):
                if float(row[key]) != value:
                    return False
            elif row[key] != str(value):
                return False
        return True
    return [r for r in rows if ok(r)]


def plot_experiments(rows: list[dict], out: Path) -> Path:
    data = select(rows, epoch=30, guidance=3.0, sampler="ddim", num_samples=5000)
    data.sort(key=lambda r: r["label"])
    labels = [EXP_NAMES.get(r["label"], r["label"]) for r in data]
    values = [float(r["fid"]) for r in data]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(labels, values, color=["#4C72B0"] * len(values))
    best = values.index(min(values))
    bars[best].set_color("#55A868")
    for bar, v in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("FID (lebih kecil lebih baik)")
    ax.set_title("FID eksperimen pada epoch 30 (w = 3, DDIM-100, 5.000 sampel)")
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_guidance(rows: list[dict], out: Path) -> Path:
    data = select(rows, label="E1", epoch=100, sampler="ddim", num_samples=5000)
    data.sort(key=lambda r: float(r["guidance"]))
    ws = [float(r["guidance"]) for r in data]
    values = [float(r["fid"]) for r in data]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(ws, values, marker="o")
    best = values.index(min(values))
    ax.scatter([ws[best]], [values[best]], s=120, facecolors="none", edgecolors="#C44E52",
               linewidths=2, label=f"terbaik: w = {ws[best]:g} (FID {values[best]:.2f})")
    for w, v in zip(ws, values, strict=True):
        ax.annotate(f"{v:.2f}", (w, v), textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=9)
    ax.set_ylim(min(values) - 1.5, max(values) + 2.5)  # ruang untuk label angka
    ax.set_xlabel("guidance scale w")
    ax.set_ylabel("FID (lebih kecil lebih baik)")
    ax.set_title("Efek guidance pada FID (E1 epoch 100, DDIM-100, 5.000 sampel)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description="Plot hasil FID")
    parser.add_argument("--csv", type=Path, default=ROOT / "reports" / "fid_results.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "figures")
    args = parser.parse_args(argv)
    rows = read_rows(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outs = [plot_experiments(rows, args.out_dir / "fid_experiments_epoch30.png"),
            plot_guidance(rows, args.out_dir / "fid_guidance_e1.png")]
    for o in outs:
        print(f"grafik disimpan: {o}")
    return outs


if __name__ == "__main__":
    main()
