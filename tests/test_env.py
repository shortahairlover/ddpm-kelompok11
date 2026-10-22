"""Test lingkungan (Fase 0): PyTorch, torchvision, dan GPU siap dipakai."""

import pytest
import torch

from scripts.check_env import check_amp, check_matmul

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="butuh GPU CUDA")


def test_torch_and_torchvision_importable():
    import torchvision

    assert torch.__version__
    assert torchvision.__version__


@requires_cuda
def test_matmul_gpu_matches_cpu():
    assert check_matmul(torch.device("cuda")) < 1e-1


@requires_cuda
def test_amp_step_finite():
    assert check_amp(torch.device("cuda"))
