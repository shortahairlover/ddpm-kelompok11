"""Test utilitas Fase 1: config YAML, seed, device, run dir, simpan grid."""

from pathlib import Path

import pytest
import torch
import yaml

from src.utils import (
    Config,
    get_device,
    load_config,
    make_run_dir,
    save_image_grid,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [ROOT / "configs" / "mnist.yaml", ROOT / "configs" / "cifar10.yaml"]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_load_config_has_required_keys(path):
    cfg = load_config(path)
    for section in ["data", "diffusion", "model", "train", "sample", "eval"]:
        assert section in cfg, f"bagian '{section}' hilang di {path.name}"
    assert cfg.data.image_size == 32
    assert cfg.diffusion.timesteps == 1000
    assert isinstance(cfg.train.lr, float)
    assert isinstance(cfg.diffusion.beta_start, float)
    assert cfg.model.base_channels % cfg.model.norm_groups == 0


def test_config_values_match_planning():
    mnist, cifar = (load_config(p) for p in CONFIGS)
    assert (mnist.data.channels, cifar.data.channels) == (1, 3)
    assert (mnist.data.hflip, cifar.data.hflip) == (False, True)
    assert mnist.model.channel_mults == [1, 2, 2]
    assert cifar.model.channel_mults == [1, 2, 2, 2]
    assert (mnist.train.ema_decay, cifar.train.ema_decay) == (0.999, 0.9999)


def test_overrides_parse_types():
    cfg = load_config(CONFIGS[0], ["train.lr=1e-4", "train.epochs=3", "train.amp=false"])
    assert cfg.train.lr == pytest.approx(1e-4) and isinstance(cfg.train.lr, float)
    assert cfg.train.epochs == 3
    assert cfg.train.amp is False


def test_override_unknown_key_raises():
    with pytest.raises(KeyError):
        load_config(CONFIGS[0], ["train.tidak_ada=1"])
    with pytest.raises(ValueError):
        load_config(CONFIGS[0], ["train.lr"])


def test_base_config_inheritance(tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        yaml.safe_dump({"base": str(CONFIGS[1]), "name": "e2", "train": {"lr": 1.0e-4}}),
        encoding="utf8",
    )
    cfg = load_config(exp)
    assert cfg.name == "e2"
    assert cfg.train.lr == pytest.approx(1e-4)
    assert cfg.train.epochs == 100  # tetap dari base
    assert "base" not in cfg


def test_config_attribute_access_and_to_dict():
    cfg = Config({"a": {"b": 1}})
    assert cfg.a.b == 1
    cfg.c = {"d": 2}
    assert cfg.c.d == 2
    assert cfg.to_dict() == {"a": {"b": 1}, "c": {"d": 2}}
    with pytest.raises(AttributeError):
        _ = cfg.tidak_ada


def test_set_seed_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_get_device():
    assert get_device("cpu").type == "cpu"
    assert get_device("auto").type in {"cpu", "cuda"}


def test_make_run_dir_creates_structure(tmp_path):
    cfg = load_config(CONFIGS[0])
    run1 = make_run_dir(cfg, root=tmp_path)
    run2 = make_run_dir(cfg, root=tmp_path)
    assert run1 != run2
    for run in (run1, run2):
        assert (run / "checkpoints").is_dir()
        assert (run / "samples").is_dir()
        assert load_config(run / "config.yaml") == cfg


def test_save_image_grid(tmp_path):
    images = torch.rand(16, 3, 32, 32) * 2 - 1  # (B, C, H, W) di [-1, 1]
    out = tmp_path / "sub" / "grid.png"
    save_image_grid(images, out, nrow=4)
    assert out.exists() and out.stat().st_size > 0


def test_config_pickle_and_deepcopy_roundtrip():
    """Config harus bisa di-pickle (DataLoader worker di Windows memakai spawn) & di-deepcopy."""
    import copy
    import pickle

    cfg = load_config(CONFIGS[1])
    restored = pickle.loads(pickle.dumps(cfg))
    assert restored == cfg and isinstance(restored.train, Config)
    clone = copy.deepcopy(cfg)
    clone.train.lr = 9.0
    assert cfg.train.lr == pytest.approx(2e-4)
