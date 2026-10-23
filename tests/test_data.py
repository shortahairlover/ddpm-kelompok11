"""Test data pipeline (Fase 2): shape, channel, rentang nilai, label, augmentasi."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data import build_transform, class_names, denormalize, get_dataloader, get_dataset
from src.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = {"mnist": 1, "cifar10": 3}


def _cfg(name: str, *overrides: str):
    return load_config(ROOT / "configs" / f"{name}.yaml", list(overrides))


@pytest.mark.parametrize("name", ["mnist", "cifar10"])
def test_debug_batch_shape_range_labels(name):
    cfg = _cfg(name, "data.batch_size=32")
    loader = get_dataloader(cfg, train=True, debug=True, num_workers=0)
    images, labels = next(iter(loader))
    assert images.shape == (32, CHANNELS[name], 32, 32)
    assert images.dtype == torch.float32
    assert images.min() >= -1.0 and images.max() <= 1.0
    assert images.min() < -0.9 and images.max() > 0.9  # rentang benar-benar terpakai
    assert labels.dtype == torch.int64
    assert labels.min() >= 0 and labels.max() <= 9


@pytest.mark.parametrize(
    "name, n_train, n_test", [("mnist", 60000, 10000), ("cifar10", 50000, 10000)]
)
def test_full_dataset_sizes(name, n_train, n_test):
    cfg = _cfg(name)
    assert len(get_dataset(cfg, train=True)) == n_train
    assert len(get_dataset(cfg, train=False)) == n_test


def test_debug_subset_size_and_deterministic():
    cfg = _cfg("mnist")
    a = get_dataset(cfg, debug=True)
    b = get_dataset(cfg, debug=True)
    assert len(a) == cfg.data.debug_subset
    assert a.indices == b.indices


def test_mnist_padding_keeps_border_black():
    cfg = _cfg("mnist")
    x, _ = get_dataset(cfg, train=False)[0]
    # padding 2 px di tepi harus bernilai -1 (hitam) setelah normalisasi
    border = torch.cat([x[:, :2].flatten(), x[:, -2:].flatten(), x[:, :, :2].flatten(),
                        x[:, :, -2:].flatten()])
    assert torch.all(border == -1.0)


def test_hflip_only_for_cifar_train():
    mnist, cifar = _cfg("mnist"), _cfg("cifar10")
    def names(transform):
        return [type(s).__name__ for s in transform.transforms]

    assert "RandomHorizontalFlip" not in names(build_transform(mnist.data, train=True))
    assert "RandomHorizontalFlip" in names(build_transform(cifar.data, train=True))
    assert "RandomHorizontalFlip" not in names(build_transform(cifar.data, train=False))


def test_transform_maps_extremes_to_minus_one_and_one():
    cfg = _cfg("cifar10")
    tf = build_transform(cfg.data, train=False)
    black = tf(Image.new("RGB", (32, 32), (0, 0, 0)))
    white = tf(Image.new("RGB", (32, 32), (255, 255, 255)))
    assert torch.allclose(black, torch.full_like(black, -1.0))
    assert torch.allclose(white, torch.full_like(white, 1.0))


def test_denormalize_inverse_and_clamp():
    x = torch.tensor([-1.0, 0.0, 1.0, 1.5, -2.0])
    assert torch.allclose(denormalize(x), torch.tensor([0.0, 0.5, 1.0, 1.0, 0.0]))


def test_class_names():
    assert len(class_names("mnist")) == 10
    assert class_names("cifar10")[3] == "cat"


def test_dataloader_with_workers_on_windows():
    """num_workers > 0 memakai spawn di Windows: transform & dataset harus bisa di-pickle."""
    cfg = _cfg("cifar10", "data.batch_size=16")
    loader = get_dataloader(cfg, train=True, debug=True, num_workers=2)
    images, labels = next(iter(loader))
    assert images.shape == (16, 3, 32, 32) and labels.shape == (16,)
