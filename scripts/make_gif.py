"""GIF proses denoising: noise murni -> gambar, satu gambar per kelas (T6.6).

Jalankan dari root repo:
    python -m scripts.make_gif --ckpt runs/<run>/checkpoints/epoch_100.pt \
        --out reports/figures/denoise_cifar10.gif [--guidance 3] [--device cpu]

Setiap frame adalah grid 2x5 (10 kelas), diperbesar 4x (nearest) agar jelas saat presentasi.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from src.data import denormalize
from src.sample import denoise_frames, load_checkpoint
from src.utils import get_device


def frames_to_images(frames: list[torch.Tensor], nrow: int, scale: int) -> list[np.ndarray]:
    """List frame (N, C, H, W) di [-1, 1] -> list gambar uint8 (H', W', 3)."""
    images = []
    for f in frames:
        f = denormalize(f.float())
        if f.shape[1] == 1:
            f = f.repeat(1, 3, 1, 1)
        f = F.interpolate(f, scale_factor=scale, mode="nearest")
        grid = make_grid(f, nrow=nrow, padding=2 * scale, pad_value=1.0)  # (3, H', W')
        images.append((grid.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
    return images


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Buat GIF proses denoising per kelas")
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--guidance", type=float, help="default cfg.sample.guidance_scale")
    parser.add_argument("--frames", type=int, default=50, help="jumlah frame proses")
    parser.add_argument("--hold", type=int, default=15, help="frame akhir diulang (jeda)")
    parser.add_argument("--duration-ms", type=int, default=80)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = get_device(args.device)
    cfg, model, diffusion, info = load_checkpoint(args.ckpt, device)
    w = cfg.sample.guidance_scale if args.guidance is None else args.guidance
    labels = list(range(cfg.data.num_classes))
    frames = denoise_frames(model, diffusion, cfg, device, labels, w,
                            n_frames=args.frames, seed=args.seed)
    images = frames_to_images(frames, nrow=5, scale=args.scale)
    images += [images[-1]] * args.hold
    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, np.stack(images), duration=args.duration_ms, loop=0)
    print(f"GIF {len(images)} frame (epoch {info['epoch']}, w={w:g}) -> {args.out}")
    return args.out


if __name__ == "__main__":
    main()
