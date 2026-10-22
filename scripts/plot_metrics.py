"""Plot kurva loss dari satu atau beberapa run (T4.6).

Contoh (dari root repo):
    python -m scripts.plot_metrics runs/mnist_2026... --out reports/figures/loss_mnist.png
    python -m scripts.plot_metrics runs/e1 runs/e2 --labels E1 E2 --out reports/figures/cmp.png
    python -m scripts.plot_metrics runs/e1 --source steps --smooth 20 --out ...

--source epochs (default) membaca metrics.csv (loss rata-rata per epoch);
--source steps membaca steps.csv (loss per log_every step), bisa dihaluskan dengan --smooth.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_curve(run_dir: Path, source: str) -> tuple[list[float], list[float]]:
    """Return (x, loss) dari metrics.csv (x = epoch) atau steps.csv (x = step)."""
    fname, xkey, ykey = {
        "epochs": ("metrics.csv", "epoch", "train_loss"),
        "steps": ("steps.csv", "step", "loss"),
    }[source]
    with open(run_dir / fname, newline="", encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{run_dir / fname} kosong")
    return [float(r[xkey]) for r in rows], [float(r[ykey]) for r in rows]


def moving_average(values: list[float], window: int) -> list[float]:
    """Rata-rata bergerak (window sebelumnya, termasuk titik ini)."""
    if window <= 1:
        return values
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def plot_runs(run_dirs: list[Path], labels: list[str], out: Path, source: str = "epochs",
              smooth: int = 1, title: str | None = None, logy: bool = False) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    for run_dir, label in zip(run_dirs, labels, strict=True):
        x, y = read_curve(run_dir, source)
        ax.plot(x, moving_average(y, smooth), label=label, linewidth=1.5)
    ax.set_xlabel("epoch" if source == "epochs" else "step")
    ax.set_ylabel("train loss (MSE noise)")
    if logy:
        ax.set_yscale("log")
    ax.set_title(title or "Kurva loss training")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Plot kurva loss dari folder run")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="*", help="label legend (default: nama folder)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", choices=["epochs", "steps"], default="epochs")
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--title", type=str)
    parser.add_argument("--logy", action="store_true")
    args = parser.parse_args(argv)
    labels = args.labels or [r.name for r in args.runs]
    if len(labels) != len(args.runs):
        parser.error("jumlah --labels harus sama dengan jumlah run")
    out = plot_runs(args.runs, labels, args.out, args.source, args.smooth, args.title, args.logy)
    print(f"grafik disimpan: {out}")
    return out


if __name__ == "__main__":
    main()
