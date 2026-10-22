"""Test utilitas evaluasi FID (Fase 6) yang tidak membutuhkan GPU/Inception."""

from pathlib import Path

import torch

from src.evaluate import ROOT, append_csv, balanced_labels, display_path, to_fid_input


def test_balanced_labels_equal_per_class():
    labels = balanced_labels(5000, 10, torch.device("cpu"))
    assert labels.shape == (5000,)
    assert torch.bincount(labels).tolist() == [500] * 10


def test_to_fid_input_range_and_channels():
    gray = torch.tensor([-1.0, 0.0, 1.0]).reshape(1, 1, 1, 3)  # (B, 1, H, W)
    out = to_fid_input(gray)
    assert out.shape == (1, 3, 1, 3)
    assert torch.allclose(out[0, :, 0], torch.tensor([0.0, 0.5, 1.0]).expand(3, 3))


def test_display_path_relative_inside_repo():
    p = ROOT / "runs" / "e1" / "checkpoints" / "last.pt"
    assert display_path(p) == "runs/e1/checkpoints/last.pt"
    outside = Path("C:/lain/model.pt")
    assert display_path(outside) == outside.as_posix()


def test_append_csv_writes_header_once(tmp_path):
    out = tmp_path / "fid.csv"
    row = {"waktu": "x", "label": "E1", "ckpt": "a", "epoch": 1, "guidance": 3.0,
           "sampler": "ddim", "steps": 100, "num_samples": 10, "fid": "1.00",
           "durasi_menit": "0.1"}
    append_csv(out, row)
    append_csv(out, row)
    lines = out.read_text(encoding="utf8").strip().splitlines()
    assert len(lines) == 3 and lines[0].startswith("waktu,label")
