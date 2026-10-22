"""Test overfit 1 batch (T3.13): versi singkat dari scripts/overfit_one_batch.py."""

import pytest

from scripts.overfit_one_batch import run_overfit, summarize


@pytest.mark.slow
def test_overfit_fixed_batch_loss_drops():
    history = run_overfit("fixed", steps=150, lr=1e-3, batch_size=16)
    first, last = summarize(history, window=10)
    assert last < 0.1 * first, f"loss tidak turun cukup: {first:.4f} -> {last:.4f}"
