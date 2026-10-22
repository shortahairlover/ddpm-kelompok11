"""Test model U-Net (Fase 3): embedding, blok, shape output, gradien."""

from pathlib import Path

import pytest
import torch

from src.model import (
    AttentionBlock,
    ResBlock,
    TimestepEmbedding,
    build_model,
    count_parameters,
    sinusoidal_embedding,
)
from src.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


def _cfg(name):
    return load_config(ROOT / "configs" / f"{name}.yaml")


def test_sinusoidal_embedding_shape_and_values():
    t = torch.tensor([0, 1, 500, 999])
    emb = sinusoidal_embedding(t, 32)
    assert emb.shape == (4, 32)
    assert torch.allclose(emb[0, :16], torch.zeros(16))  # sin(0) = 0
    assert torch.allclose(emb[0, 16:], torch.ones(16))  # cos(0) = 1
    assert emb.abs().max() <= 1.0
    assert not torch.allclose(emb[1], emb[2])  # timestep berbeda -> embedding berbeda


def test_sinusoidal_embedding_odd_dim_raises():
    with pytest.raises(ValueError):
        sinusoidal_embedding(torch.tensor([1]), 7)


def test_timestep_embedding_mlp_shape():
    emb = TimestepEmbedding(32, 128)(torch.randint(0, 1000, (5,)))
    assert emb.shape == (5, 128)


def test_resblock_shapes_and_identity_init():
    block = ResBlock(32, 64, emb_dim=128, dropout=0.0, groups=8)
    x = torch.randn(2, 32, 16, 16)
    out = block(x, torch.randn(2, 128))
    assert out.shape == (2, 64, 16, 16)
    same = ResBlock(32, 32, emb_dim=128, dropout=0.0, groups=8)
    assert torch.allclose(same(x, torch.randn(2, 128)), x)  # conv2 nol -> identitas di awal


def test_attention_block_shape_and_mixing():
    attn = AttentionBlock(64, groups=8)
    torch.nn.init.normal_(attn.proj.weight, std=0.1)  # aktifkan proyeksi (default nol)
    x = torch.randn(2, 64, 8, 8)
    out = attn(x)
    assert out.shape == x.shape
    # mengubah satu piksel memengaruhi output di piksel lain (attention global)
    x2 = x.clone()
    x2[:, :, 0, 0] += 5.0
    diff = (attn(x2) - out).abs()
    assert diff[:, :, 7, 7].sum() > 0


@pytest.mark.parametrize("name, channels", [("mnist", 1), ("cifar10", 3)])
def test_unet_output_shape_matches_input(name, channels):
    cfg = _cfg(name)
    model = build_model(cfg)
    x = torch.randn(3, channels, 32, 32)
    t = torch.tensor([0, 500, 999])
    y = torch.tensor([0, 9, 10])  # 10 = token null
    assert model(x, t, y).shape == x.shape
    assert model(x, t, None).shape == x.shape  # tanpa label -> token null


def test_unet_parameter_counts_reasonable():
    n_mnist = count_parameters(build_model(_cfg("mnist")))
    n_cifar = count_parameters(build_model(_cfg("cifar10")))
    assert 0.5e6 < n_mnist < 5e6
    assert 5e6 < n_cifar < 20e6  # muat di VRAM 6 GB dengan batch 128 + AMP


def test_unet_gradients_reach_all_parameters():
    cfg = _cfg("mnist")
    model = build_model(cfg)
    # bobot nol (init) memblokir gradien ke lapisan sebelumnya; acak semua bobot dulu
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)
    x = torch.randn(2, 1, 32, 32)
    out = model(x, torch.tensor([3, 700]), torch.tensor([1, 10]))
    out.pow(2).mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    # embedding kelas hanya mendapat gradien di baris yang dipakai (1 dan 10) -> tetap non-nol
    assert missing == [], f"parameter tanpa gradien: {missing}"


def test_class_label_changes_output():
    cfg = _cfg("mnist")
    model = build_model(cfg)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)
    x = torch.randn(1, 1, 32, 32)
    t = torch.tensor([100])
    out_a = model(x, t, torch.tensor([3]))
    out_b = model(x, t, torch.tensor([7]))
    assert not torch.allclose(out_a, out_b)
