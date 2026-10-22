"""Tes overfit 1 batch (T3.13): bukti model + loss + optimizer bisa belajar.

Dua mode pada satu batch MNIST yang sama:
    fixed  : t dan noise tetap setiap step -> tugas deterministik, loss harus turun mendekati 0
    random : t dan noise diacak setiap step (seperti training sungguhan) -> loss harus turun

Jalankan dari root repo:
    python -m scripts.overfit_one_batch [--steps 400]
Output: reports/figures/overfit_one_batch.png dan ringkasan di konsol.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.data import get_dataloader  # noqa: E402
from src.diffusion import Diffusion  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import get_device, load_config, set_seed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run_overfit(mode: str, steps: int, lr: float, batch_size: int = 32) -> list[float]:
    """Latih model baru pada 1 batch. Return riwayat loss per step."""
    cfg = load_config(ROOT / "configs" / "mnist.yaml", [f"data.batch_size={batch_size}"])
    set_seed(cfg.seed)
    device = get_device()
    model = build_model(cfg).to(device)
    diffusion = Diffusion.from_config(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    x0, y = next(iter(get_dataloader(cfg, train=True, debug=True, num_workers=0)))
    x0, y = x0.to(device), y.to(device)  # x0: (B, 1, 32, 32), y: (B,)
    fixed_t = diffusion.sample_timesteps(x0.shape[0], device)
    fixed_noise = torch.randn_like(x0)

    history = []
    model.train()
    for _ in range(steps):
        if mode == "fixed":
            x_t = diffusion.q_sample(x0, fixed_t, fixed_noise)
            loss = F.mse_loss(model(x_t, fixed_t, y), fixed_noise)
        else:
            loss = diffusion.training_loss(model, x0, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(loss.item())
    return history


def summarize(history: list[float], window: int = 20) -> tuple[float, float]:
    """Rata-rata loss `window` step pertama dan terakhir."""
    first = sum(history[:window]) / window
    last = sum(history[-window:]) / window
    return first, last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    results = {}
    for mode in ["fixed", "random"]:
        history = run_overfit(mode, args.steps, args.lr)
        first, last = summarize(history)
        results[mode] = history
        print(f"{mode:6s}: loss awal {first:.4f} -> akhir {last:.4f} "
              f"(turun {100 * (1 - last / first):.1f}%)")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for mode, history in results.items():
        ax.plot(history, label=f"{mode} t & noise", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE loss (log)")
    ax.set_title("Overfit 1 batch MNIST (32 gambar)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out = ROOT / "reports" / "figures" / "overfit_one_batch.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"grafik disimpan: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
