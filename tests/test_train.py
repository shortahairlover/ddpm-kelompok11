"""Test training loop, resume, sampling, dan plot (Fase 4)."""

import csv
from pathlib import Path

import pytest
import torch

from scripts.plot_metrics import moving_average, plot_runs
from src import sample as sample_mod
from src import train as train_mod
from src.train import class_grid_labels, lr_at_step

ROOT = Path(__file__).resolve().parents[1]


def test_lr_warmup_schedule():
    assert lr_at_step(0, 2e-4, 100) == pytest.approx(2e-6)
    assert lr_at_step(49, 2e-4, 100) == pytest.approx(1e-4)
    assert lr_at_step(99, 2e-4, 100) == pytest.approx(2e-4)
    assert lr_at_step(5000, 2e-4, 100) == pytest.approx(2e-4)
    assert lr_at_step(0, 2e-4, 0) == pytest.approx(2e-4)


def test_class_grid_labels_one_row_per_class():
    labels = class_grid_labels(3, 2, torch.device("cpu"))
    assert labels.tolist() == [0, 0, 1, 1, 2, 2]


def test_moving_average():
    assert moving_average([1, 2, 3, 4], 2) == [1, 1.5, 2.5, 3.5]
    assert moving_average([1, 2], 1) == [1, 2]


def _read_csv(path):
    with open(path, newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


@pytest.mark.slow
def test_debug_train_resume_sample_plot(tmp_path):
    """End-to-end: train --debug (2 epoch) -> resume ke epoch 3 -> sample -> plot."""
    summary = train_mod.main(["--config", str(ROOT / "configs" / "mnist.yaml"), "--debug",
                              "--runs-root", str(tmp_path)])
    run_dir = Path(summary["run_dir"])
    assert summary["epoch"] == 2 and summary["step"] == 8  # 512 gambar / batch 128 = 4 step
    assert torch.isfinite(torch.tensor(summary["train_loss"]))
    for name in ["config.yaml", "train.log", "metrics.csv", "steps.csv",
                 "samples/epoch_001.png", "samples/epoch_002.png",
                 "checkpoints/epoch_002.pt", "checkpoints/last.pt"]:
        assert (run_dir / name).exists(), f"{name} tidak dibuat"

    # resume: epoch dan step berlanjut, CSV tersambung (bukan ditimpa)
    resumed = train_mod.main(["--resume", str(run_dir), "--epochs", "3"])
    assert resumed["epoch"] == 3 and resumed["step"] == 12
    rows = _read_csv(run_dir / "metrics.csv")
    assert [int(r["epoch"]) for r in rows] == [1, 2, 3]
    state = torch.load(run_dir / "checkpoints" / "last.pt", map_location="cpu",
                       weights_only=False)
    assert state["epoch"] == 3 and state["ema"]["num_updates"] == 12

    # sampling dari checkpoint: shape & rentang benar, EMA vs non-EMA
    saved = sample_mod.main(["--ckpt", str(run_dir / "checkpoints" / "last.pt"),
                             "--guidance", "0", "2", "--frames", "--out", str(tmp_path / "gen")])
    assert len(saved) == 3 and all(p.exists() for p in saved)
    cfg, model, diffusion, info = sample_mod.load_checkpoint(
        run_dir / "checkpoints" / "last.pt", torch.device("cpu"), use_ema=False)
    assert info["epoch"] == 3 and not model.training

    # plot kurva loss
    out = plot_runs([run_dir], ["debug"], tmp_path / "loss.png", source="steps", smooth=2)
    assert out.exists()
