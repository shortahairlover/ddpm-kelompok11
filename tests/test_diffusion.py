"""Test komponen diffusion (Fase 3): schedule, forward process, loss, CFG, sampling, EMA."""

from pathlib import Path

import pytest
import torch
from torch import nn

from src.diffusion import Diffusion
from src.ema import EMA
from src.schedule import (
    NoiseSchedule,
    build_schedule,
    cosine_beta_schedule,
    extract,
    linear_beta_schedule,
)
from src.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


def _cfg(name="mnist", *overrides):
    return load_config(ROOT / "configs" / f"{name}.yaml", list(overrides))


# ---------------------------------------------------------------- schedule
def test_linear_schedule_endpoints():
    betas = linear_beta_schedule(1000, 1e-4, 0.02)
    assert betas.shape == (1000,)
    assert betas[0].item() == pytest.approx(1e-4)
    assert betas[-1].item() == pytest.approx(0.02)
    assert torch.all(betas[1:] > betas[:-1])


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_schedule_constants_valid(kind):
    cfg = _cfg("cifar10", f"diffusion.schedule={kind}")
    s = build_schedule(cfg.diffusion)
    a_bar = s.alphas_cumprod
    assert s.timesteps == 1000
    assert torch.all((s.betas > 0) & (s.betas < 1))
    assert torch.all(a_bar[1:] < a_bar[:-1])  # sinyal makin hilang
    assert a_bar[0] > 0.99 and a_bar[-1] < 1e-3  # awal hampir bersih, akhir hampir noise murni
    assert torch.allclose(s.sqrt_alphas_cumprod**2 + s.sqrt_one_minus_alphas_cumprod**2,
                          torch.ones(1000), atol=1e-5)
    assert s.posterior_variance[0].item() == pytest.approx(0.0, abs=1e-12)
    assert torch.all(s.posterior_variance[1:] <= s.betas[1:] + 1e-7)  # beta_tilde <= beta
    assert torch.isfinite(s.posterior_log_variance_clipped).all()


def test_cosine_schedule_clipped():
    betas = cosine_beta_schedule(1000)
    assert betas.max().item() <= 0.999
    assert betas.min().item() > 0


def test_schedule_rejects_invalid_betas():
    with pytest.raises(ValueError):
        NoiseSchedule(torch.tensor([0.1, 1.0]))


def test_unknown_schedule_raises():
    with pytest.raises(ValueError):
        build_schedule(_cfg("mnist", "diffusion.schedule=quadratic").diffusion)


def test_extract_shape_and_values():
    a = torch.arange(10, dtype=torch.float32)
    t = torch.tensor([0, 3, 9])
    out = extract(a, t, (3, 1, 32, 32))
    assert out.shape == (3, 1, 1, 1)
    assert out.flatten().tolist() == [0.0, 3.0, 9.0]


# ---------------------------------------------------------------- forward process
def test_q_sample_matches_formula_and_extremes():
    diffusion = Diffusion.from_config(_cfg())
    x0 = torch.rand(4, 1, 32, 32) * 2 - 1
    noise = torch.randn_like(x0)

    # t=0 ~ x0: selisih hanya berasal dari sqrt(1 - a_bar_0) * noise = 0.01 * noise
    t0 = torch.zeros(4, dtype=torch.long)
    err = (diffusion.q_sample(x0, t0, noise) - x0).abs()
    assert torch.all(err <= 0.0101 * noise.abs() + 1e-4)

    t_last = torch.full((4,), 999, dtype=torch.long)
    assert torch.allclose(diffusion.q_sample(x0, t_last, noise), noise, atol=0.02)  # ~ noise

    t = torch.tensor([10, 200, 500, 900])
    a_bar = diffusion.schedule.alphas_cumprod[t].reshape(-1, 1, 1, 1)
    expected = a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * noise
    assert torch.allclose(diffusion.q_sample(x0, t, noise), expected, atol=1e-6)


def test_q_sample_variance_preserving():
    """Untuk x0 berdistribusi N(0,1), x_t juga ~ N(0,1) di semua t (variance preserving)."""
    diffusion = Diffusion.from_config(_cfg())
    x0 = torch.randn(4096, 1, 4, 4)
    for step in [0, 300, 999]:
        t = torch.full((4096,), step, dtype=torch.long)
        x_t = diffusion.q_sample(x0, t, torch.randn_like(x0))
        assert x_t.std().item() == pytest.approx(1.0, abs=0.03)


def test_predict_x0_inverts_q_sample():
    diffusion = Diffusion.from_config(_cfg())
    x0 = torch.rand(4, 1, 32, 32) * 2 - 1
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 50, 400, 800])
    x_t = diffusion.q_sample(x0, t, noise)
    assert torch.allclose(diffusion.predict_x0(x_t, t, noise), x0, atol=1e-3)


# ---------------------------------------------------------------- loss & CFG
class _ConstModel(nn.Module):
    """Model palsu: eps = (y + 1) * 0.1 per sampel, agar efek label & CFG bisa dicek pasti."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, x, t, y):
        return torch.ones_like(x) * ((y.float() + 1) * 0.1).reshape(-1, 1, 1, 1) + self.dummy


def test_drop_labels_rate_and_values():
    diffusion = Diffusion.from_config(_cfg())
    torch.manual_seed(0)
    y = torch.randint(0, 10, (20000,))
    dropped = diffusion.drop_labels(y)
    is_null = dropped == diffusion.null_class
    assert is_null.float().mean().item() == pytest.approx(0.1, abs=0.01)
    assert torch.equal(dropped[~is_null], y[~is_null])


def test_training_loss_scalar_and_backward():
    cfg = _cfg()
    diffusion = Diffusion.from_config(cfg)
    model = _ConstModel()
    loss = diffusion.training_loss(model, torch.rand(8, 1, 32, 32) * 2 - 1,
                                   torch.randint(0, 10, (8,)))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.dummy.grad is not None


@pytest.mark.parametrize("w", [0.0, 1.0, 3.0])
def test_predict_eps_guidance_formula(w):
    diffusion = Diffusion.from_config(_cfg())
    model = _ConstModel()
    x = torch.randn(2, 1, 8, 8)
    t = torch.tensor([5, 5])
    y = torch.tensor([2, 7])
    eps = diffusion.predict_eps(model, x, t, y, guidance_scale=w)
    cond = ((y.float() + 1) * 0.1).reshape(-1, 1, 1, 1)
    uncond = (diffusion.null_class + 1) * 0.1
    expected = uncond + w * (cond - uncond)
    assert torch.allclose(eps, expected.expand_as(eps), atol=1e-6)


def test_sample_shape_range_and_frames():
    cfg = _cfg("mnist", "diffusion.timesteps=20")
    diffusion = Diffusion.from_config(cfg)
    model = _ConstModel()
    y = torch.arange(4)
    out, frames = diffusion.sample(model, y, (4, 1, 8, 8), guidance_scale=2.0, save_every=5)
    assert out.shape == (4, 1, 8, 8)
    assert out.min() >= -1 and out.max() <= 1
    assert len(frames) == 4  # i = 15, 10, 5, 0


def test_sample_reproducible_with_generator():
    cfg = _cfg("mnist", "diffusion.timesteps=10")
    diffusion = Diffusion.from_config(cfg)
    model = _ConstModel()
    y = torch.arange(3)
    a, _ = diffusion.sample(model, y, (3, 1, 8, 8), generator=torch.Generator().manual_seed(1))
    b, _ = diffusion.sample(model, y, (3, 1, 8, 8), generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


# ---------------------------------------------------------------- EMA
def test_ema_update_and_warmup():
    model = nn.Linear(4, 4)
    ema = EMA(model, decay=0.999)
    assert ema.current_decay() == pytest.approx(0.1)  # (1+0)/(10+0)
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    # shadow_baru = d*shadow_lama + (1-d)*param, dengan d = 0.1 dan param = shadow_lama + 1
    # -> shadow_baru = shadow_lama + 0.9 = param - 0.1
    assert torch.allclose(ema.model.weight, model.weight - 0.1, atol=1e-6)
    assert ema.num_updates == 1
    assert all(not p.requires_grad for p in ema.model.parameters())


def test_ema_state_dict_roundtrip():
    model = nn.Linear(4, 4)
    ema = EMA(model, decay=0.99)
    ema.update(model)
    other = EMA(nn.Linear(4, 4), decay=0.5)
    other.load_state_dict(ema.state_dict())
    assert other.num_updates == 1 and other.decay == 0.99
    assert torch.equal(other.model.weight, ema.model.weight)


def test_ema_rejects_invalid_decay():
    with pytest.raises(ValueError):
        EMA(nn.Linear(2, 2), decay=1.0)


# ---------------------------------------------------------------- DDIM
class _LinearModel(nn.Module):
    """Model palsu yang bergantung pada x, t, dan y agar perbandingan sampler bermakna."""

    def forward(self, x, t, y):
        return 0.3 * x + 0.01 * t.float().reshape(-1, 1, 1, 1) / 20 + 0.05 * y.float().reshape(
            -1, 1, 1, 1
        )


def test_ddim_eta1_full_steps_equals_ddpm():
    """eta = 1 dan steps = T: DDIM harus identik dengan DDPM (mean & varians posterior sama)."""
    cfg = _cfg("mnist", "diffusion.timesteps=20")
    diffusion = Diffusion.from_config(cfg)
    model = _LinearModel()
    y = torch.tensor([0, 3, 7])
    shape = (3, 1, 8, 8)
    ddpm, _ = diffusion.sample(model, y, shape, guidance_scale=2.0,
                               generator=torch.Generator().manual_seed(5))
    ddim = diffusion.ddim_sample(model, y, shape, steps=20, eta=1.0, guidance_scale=2.0,
                                 generator=torch.Generator().manual_seed(5))
    assert torch.allclose(ddpm, ddim, atol=1e-4)


def test_ddim_eta0_deterministic_and_shape():
    cfg = _cfg("mnist", "diffusion.timesteps=50")
    diffusion = Diffusion.from_config(cfg)
    model = _LinearModel()
    y = torch.arange(4)
    a = diffusion.ddim_sample(model, y, (4, 1, 8, 8), steps=10,
                              generator=torch.Generator().manual_seed(1))
    b = diffusion.ddim_sample(model, y, (4, 1, 8, 8), steps=10,
                              generator=torch.Generator().manual_seed(1))
    assert a.shape == (4, 1, 8, 8) and torch.equal(a, b)
    assert a.min() >= -1 and a.max() <= 1


def test_ddim_invalid_steps():
    diffusion = Diffusion.from_config(_cfg("mnist", "diffusion.timesteps=10"))
    with pytest.raises(ValueError):
        diffusion.ddim_sample(_LinearModel(), torch.arange(2), (2, 1, 4, 4), steps=11)
