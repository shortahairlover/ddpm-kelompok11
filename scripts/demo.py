"""Demo presentasi (T7.8): generate gambar per kelas langsung dari checkpoint terbaik.

Jalankan dari root repo:
    python -m scripts.demo                              # CIFAR-10, semua kelas, DDIM-100 (cepat)
    python -m scripts.demo --kelas cat dog --n 8        # kelas tertentu
    python -m scripts.demo --dataset mnist --kelas 3 7
    python -m scripts.demo --sampler ddpm               # DDPM 1000 langkah (lebih lambat)

Checkpoint default = checkpoint terbaik (reports/eksperimen.md); bisa diganti dengan --ckpt.
Gambar disimpan ke runs/demo/ dan dibuka otomatis (Windows) kecuali --no-open.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from src.data import class_names
from src.sample import load_checkpoint
from src.utils import get_device, save_image_grid

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {  # checkpoint terbaik hasil Fase 5-6
    "cifar10": ("runs/e1_baseline_*/checkpoints/epoch_100.pt", 2.0),
    "mnist": ("runs/mnist_sanity_*/checkpoints/epoch_020.pt", 2.0),
}


def find_checkpoint(dataset: str) -> Path:
    """Checkpoint default untuk dataset (run terbaru yang cocok dengan pola)."""
    pattern = DEFAULTS[dataset][0]
    matches = sorted(ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"checkpoint {pattern} tidak ditemukan; pakai --ckpt")
    return matches[-1]


def parse_classes(tokens: list[str] | None, names: list[str]) -> list[int]:
    """Nama/indeks kelas -> indeks. None -> semua kelas."""
    if not tokens:
        return list(range(len(names)))
    out = []
    for tok in tokens:
        if tok.isdigit() and int(tok) < len(names) and tok not in names:
            out.append(int(tok))
        elif tok in names:
            out.append(names.index(tok))
        else:
            raise ValueError(f"kelas tidak dikenal: {tok!r}; pilihan: {', '.join(names)}")
    return out


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Demo generate gambar per kelas")
    parser.add_argument("--dataset", choices=list(DEFAULTS), default="cifar10")
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--kelas", nargs="*", help="nama atau indeks kelas (default: semua)")
    parser.add_argument("--n", type=int, default=6, help="gambar per kelas")
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, help="default: acak (berbeda tiap demo)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)

    device = get_device(args.device)
    ckpt = args.ckpt or find_checkpoint(args.dataset)
    cfg, model, diffusion, info = load_checkpoint(ckpt, device)
    names = class_names(cfg.data.dataset)
    classes = parse_classes(args.kelas, names)
    w = DEFAULTS[cfg.data.dataset][1] if args.guidance is None else args.guidance
    seed = args.seed if args.seed is not None else int(time.time()) % 100000

    y = torch.tensor(classes, device=device).repeat_interleave(args.n)  # (K * n,)
    shape = (y.shape[0], cfg.data.channels, cfg.data.image_size, cfg.data.image_size)
    generator = torch.Generator(device=device).manual_seed(seed)
    print(f"checkpoint : {ckpt.relative_to(ROOT) if ckpt.is_relative_to(ROOT) else ckpt}")
    print(f"kelas      : {', '.join(names[c] for c in classes)} ({args.n} gambar/kelas)")
    print(f"sampler    : {args.sampler}"
          f"{'-' + str(args.steps) if args.sampler == 'ddim' else '-' + str(diffusion.timesteps)}"
          f", guidance w = {w:g}, seed {seed}")
    t0 = time.perf_counter()
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=device.type == "cuda"):
        if args.sampler == "ddim":
            images = diffusion.ddim_sample(model, y, shape, steps=args.steps,
                                           guidance_scale=w, generator=generator)
        else:
            images, _ = diffusion.sample(model, y, shape, guidance_scale=w, generator=generator)
    out = ROOT / "runs" / "demo" / f"{cfg.data.dataset}_{datetime.now():%H%M%S}.png"
    save_image_grid(images.float(), out, nrow=args.n)
    print(f"selesai {time.perf_counter() - t0:.1f}s -> {out.relative_to(ROOT)}")
    print("setiap baris = satu kelas (urutan sesuai --kelas)")
    if not args.no_open and sys.platform == "win32":
        os.startfile(out)  # buka gambar untuk demo
    return out


if __name__ == "__main__":
    main()
