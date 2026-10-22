"""Exponential Moving Average (EMA) bobot model (T3.11).

shadow <- decay_eff * shadow + (1 - decay_eff) * param, dengan
decay_eff = min(decay, (1 + n) / (10 + n)) agar di awal training (n kecil) EMA cepat
mengikuti model dan tidak "terjebak" di bobot inisialisasi.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    """Menyimpan salinan model dengan bobot rata-rata eksponensial (dipakai untuk sampling)."""

    def __init__(self, model: nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError("decay harus di (0, 1)")
        self.decay = decay
        self.num_updates = 0
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def current_decay(self) -> float:
        return min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Perbarui shadow dari bobot model saat ini (panggil setelah optimizer.step())."""
        decay = self.current_decay()
        for ema_p, p in zip(self.model.parameters(), model.parameters(), strict=True):
            ema_p.lerp_(p.detach(), 1.0 - decay)
        for ema_b, b in zip(self.model.buffers(), model.buffers(), strict=True):
            ema_b.copy_(b)
        self.num_updates += 1

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict(), "num_updates": self.num_updates,
                "decay": self.decay}

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.num_updates = state["num_updates"]
        self.decay = state["decay"]
