"""Proses diffusion: forward (q_sample), loss training, dan sampling DDPM + CFG.

    forward : x_t = sqrt(a_bar_t) x_0 + sqrt(1 - a_bar_t) eps,     eps ~ N(0, I)
    loss    : || eps - eps_theta(x_t, t, y) ||^2                   (label di-drop p_uncond)
    reverse : x_{t-1} = mu(x_t, x0_hat) + sigma_t z,               z ~ N(0, I), z = 0 di t = 0
    CFG     : eps = eps_uncond + w * (eps_cond - eps_uncond)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.schedule import NoiseSchedule, build_schedule, extract
from src.utils import Config


class Diffusion:
    """Membungkus schedule + operasi diffusion. Model dipanggil sebagai model(x_t, t, y)."""

    def __init__(self, schedule: NoiseSchedule, num_classes: int, p_uncond: float):
        self.schedule = schedule
        self.timesteps = schedule.timesteps
        self.num_classes = num_classes
        self.null_class = num_classes  # indeks token null di class embedding
        self.p_uncond = p_uncond

    @classmethod
    def from_config(cls, cfg: Config) -> Diffusion:
        return cls(build_schedule(cfg.diffusion), cfg.data.num_classes, cfg.diffusion.p_uncond)

    def to(self, device: torch.device | str) -> Diffusion:
        self.schedule.to(device)
        return self

    # ------------------------------------------------------------------ forward process
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sampel x_t ~ q(x_t | x_0). x0, noise: (B, C, H, W); t: (B,) long."""
        s = self.schedule
        return (
            extract(s.sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract(s.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """t ~ Uniform{0, ..., T-1}. Return (B,) long."""
        return torch.randint(0, self.timesteps, (batch_size,), device=device)

    def drop_labels(self, y: torch.Tensor) -> torch.Tensor:
        """Ganti label dengan token null secara acak (peluang p_uncond) untuk melatih CFG."""
        drop = torch.rand(y.shape[0], device=y.device) < self.p_uncond
        return torch.where(drop, torch.full_like(y, self.null_class), y)

    def training_loss(self, model: nn.Module, x0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Loss MSE prediksi noise (rata-rata atas batch & piksel). x0: (B, C, H, W), y: (B,)."""
        t = self.sample_timesteps(x0.shape[0], x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_pred = model(x_t, t, self.drop_labels(y))
        return F.mse_loss(eps_pred.float(), noise)

    # ------------------------------------------------------------------ reverse process
    def predict_eps(
        self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Prediksi noise dengan classifier-free guidance. x: (B, C, H, W) -> (B, C, H, W).

        w = 0: tanpa kondisi, w = 1: kondisional murni, w > 1: kondisi diperkuat.
        Untuk w lain, versi cond & uncond dihitung dalam satu forward (batch 2B).
        """
        null = torch.full_like(y, self.null_class)
        if guidance_scale == 1.0:
            return model(x, t, y).float()
        if guidance_scale == 0.0:
            return model(x, t, null).float()
        eps = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null])).float()
        eps_cond, eps_uncond = eps.chunk(2)
        return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """x_0 = x_t / sqrt(a_bar_t) - sqrt(1/a_bar_t - 1) * eps."""
        s = self.schedule
        return (
            extract(s.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(s.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def p_sample(
        self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor,
        guidance_scale: float, generator: torch.Generator | None = None, clip_x0: bool = True,
    ) -> torch.Tensor:
        """Satu langkah x_t -> x_{t-1}. Semua sampel di batch memakai t yang sama."""
        s = self.schedule
        eps = self.predict_eps(model, x_t, t, y, guidance_scale)
        x0_hat = self.predict_x0(x_t, t, eps)
        if clip_x0:
            x0_hat = x0_hat.clamp(-1.0, 1.0)  # data asli ada di [-1, 1]
        mean = (
            extract(s.posterior_mean_coef1, t, x_t.shape) * x0_hat
            + extract(s.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        log_var = extract(s.posterior_log_variance_clipped, t, x_t.shape)
        noise = torch.randn(x_t.shape, device=x_t.device, generator=generator)
        nonzero = (t != 0).float().reshape(-1, *([1] * (x_t.ndim - 1)))  # tanpa noise di t = 0
        return mean + nonzero * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        y: torch.Tensor,
        shape: Sequence[int],
        guidance_scale: float = 1.0,
        generator: torch.Generator | None = None,
        save_every: int | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Sampling DDPM penuh dari noise murni.

        y: (B,) label target; shape: (B, C, H, W). save_every=k menyimpan x_t (CPU) setiap
        k langkah + hasil akhir (untuk GIF). progress(i) dipanggil tiap langkah (opsional).
        Return (gambar akhir (B, C, H, W) di [-1, 1], daftar intermediate).
        """
        device = y.device
        x = torch.randn(tuple(shape), device=device, generator=generator)
        frames: list[torch.Tensor] = []
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, y, guidance_scale, generator=generator)
            if save_every and (i % save_every == 0):
                frames.append(x.clamp(-1, 1).cpu())
            if progress is not None:
                progress(i)
        return x.clamp(-1.0, 1.0), frames
