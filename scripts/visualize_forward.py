"""Visualisasi forward process q(x_t | x_0) dan perbandingan schedule (T3.3).

Jalankan dari root repo:
    python -m scripts.visualize_forward

Output (reports/figures/):
    forward_mnist.png, forward_cifar10.png : baris = gambar, kolom = t (0, 100, 250, 500, 750, 999)
    schedule_alphas_cumprod.png           : alpha_bar_t schedule linear vs cosine
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.data import get_dataset  # noqa: E402
from src.diffusion import Diffusion  # noqa: E402
from src.schedule import build_schedule  # noqa: E402
from src.utils import load_config, save_image_grid, set_seed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "figures"
TIMESTEPS = [0, 100, 250, 500, 750, 999]


def forward_grid(dataset: str, n_images: int = 4) -> Path:
    """Simpan grid n_images x len(TIMESTEPS) memakai noise yang sama per gambar."""
    cfg = load_config(ROOT / "configs" / f"{dataset}.yaml")
    set_seed(cfg.seed)
    diffusion = Diffusion.from_config(cfg)
    ds = get_dataset(cfg, train=False)
    x0 = torch.stack([ds[i][0] for i in range(n_images)])  # (N, C, 32, 32)
    noise = torch.randn_like(x0)

    rows = []
    for i in range(n_images):
        t = torch.tensor(TIMESTEPS)
        xi = x0[i : i + 1].expand(len(TIMESTEPS), -1, -1, -1)
        ni = noise[i : i + 1].expand(len(TIMESTEPS), -1, -1, -1)
        rows.append(diffusion.q_sample(xi, t, ni))  # (len(TIMESTEPS), C, 32, 32)
    out = OUT_DIR / f"forward_{dataset}.png"
    save_image_grid(torch.cat(rows), out, nrow=len(TIMESTEPS))
    a_bar = diffusion.schedule.alphas_cumprod[TIMESTEPS]
    print(f"{dataset}: t={TIMESTEPS} sqrt(a_bar)={[round(v, 4) for v in a_bar.sqrt().tolist()]}")
    print(f"  disimpan: {out.relative_to(ROOT)}")
    return out


def schedule_plot() -> Path:
    """Plot alpha_bar_t untuk schedule linear dan cosine."""
    cfg = load_config(ROOT / "configs" / "cifar10.yaml")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for kind in ["linear", "cosine"]:
        cfg.diffusion.schedule = kind
        a_bar = build_schedule(cfg.diffusion).alphas_cumprod
        ax.plot(a_bar.numpy(), label=kind)
    ax.set_xlabel("timestep t")
    ax.set_ylabel(r"$\bar{\alpha}_t$")
    ax.set_title("Noise schedule: sisa sinyal per timestep")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "schedule_alphas_cumprod.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"schedule plot disimpan: {out.relative_to(ROOT)}")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in ["mnist", "cifar10"]:
        forward_grid(dataset)
    schedule_plot()


if __name__ == "__main__":
    main()
