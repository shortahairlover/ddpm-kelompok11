"""Noise schedule DDPM: beta_t dan semua konstanta turunannya (T3.1).

Notasi (Ho et al., 2020):
    alpha_t      = 1 - beta_t
    alpha_bar_t  = prod_{s<=t} alpha_s                       (alphas_cumprod)
    q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
    posterior q(x_{t-1} | x_t, x_0) = N(coef1 x_0 + coef2 x_t, beta_tilde_t I)
Indeks t di kode berbasis 0: t = 0 .. T-1.
"""

from __future__ import annotations

import math

import torch

from src.utils import Config


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Beta naik linear dari beta_start ke beta_end. Return (T,) float64."""
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Schedule cosine (Nichol & Dhariwal, 2021): alpha_bar(t) = cos^2(((t/T + s)/(1 + s)) * pi/2).

    beta_t = 1 - alpha_bar(t+1) / alpha_bar(t), dipotong ke max_beta agar tidak singular di
    akhir. Return (T,) float64.
    """
    steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
    alpha_bar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(min=0.0, max=max_beta)


class NoiseSchedule:
    """Menyimpan beta_t dan konstanta turunan sebagai tensor (T,) float32.

    Konstanta dihitung di float64 lalu dikonversi, agar alpha_bar mendekati 0 tetap akurat.
    """

    def __init__(self, betas: torch.Tensor):
        betas = betas.to(torch.float64)
        if betas.ndim != 1 or not bool(((betas > 0) & (betas < 1)).all()):
            raise ValueError("betas harus tensor 1D dengan nilai di (0, 1)")
        self.timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        consts = {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "alphas_cumprod_prev": alphas_cumprod_prev,
            # forward process q(x_t | x_0)
            "sqrt_alphas_cumprod": alphas_cumprod.sqrt(),
            "sqrt_one_minus_alphas_cumprod": (1.0 - alphas_cumprod).sqrt(),
            # rekonstruksi x_0 dari x_t dan eps: x_0 = x_t / sqrt(a_bar) - sqrt(1/a_bar - 1) eps
            "sqrt_recip_alphas_cumprod": (1.0 / alphas_cumprod).sqrt(),
            "sqrt_recipm1_alphas_cumprod": (1.0 / alphas_cumprod - 1.0).sqrt(),
            # posterior q(x_{t-1} | x_t, x_0)
            "posterior_variance": posterior_variance,
            # variance di t=0 bernilai 0 -> log diganti nilai t=1 agar tidak -inf
            "posterior_log_variance_clipped": torch.log(
                torch.cat([posterior_variance[1:2], posterior_variance[1:]])
            ),
            "posterior_mean_coef1": betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod),
            "posterior_mean_coef2": (1.0 - alphas_cumprod_prev) * alphas.sqrt()
            / (1.0 - alphas_cumprod),
        }
        self._names = list(consts)
        for name, value in consts.items():
            setattr(self, name, value.to(torch.float32))

    def to(self, device: torch.device | str) -> NoiseSchedule:
        """Pindahkan semua konstanta ke device (in-place), return self."""
        for name in self._names:
            setattr(self, name, getattr(self, name).to(device))
        return self


def build_schedule(diffusion_cfg: Config) -> NoiseSchedule:
    """Buat NoiseSchedule dari bagian `diffusion` config."""
    kind = diffusion_cfg.schedule
    if kind == "linear":
        betas = linear_beta_schedule(
            diffusion_cfg.timesteps, diffusion_cfg.beta_start, diffusion_cfg.beta_end
        )
    elif kind == "cosine":
        betas = cosine_beta_schedule(diffusion_cfg.timesteps)
    else:
        raise ValueError(f"schedule tidak dikenal: {kind!r} (linear | cosine)")
    return NoiseSchedule(betas)


def extract(a: torch.Tensor, t: torch.Tensor, shape: torch.Size | tuple) -> torch.Tensor:
    """Ambil a[t] per sampel lalu reshape agar bisa di-broadcast ke tensor gambar.

    a: (T,), t: (B,) long, shape: (B, C, H, W) -> output (B, 1, 1, 1).
    """
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))
