"""Generate gambar dari checkpoint (T4.5).

Contoh (dari root repo):
    python -m src.sample --ckpt runs/<run>/checkpoints/last.pt
    python -m src.sample --ckpt <ckpt> --guidance 0 1 3 5          # variasi guidance (S1)
    python -m src.sample --ckpt <ckpt> --frames                     # strip proses denoising
    python -m src.sample --ckpt <ckpt> --no-ema                     # bobot non-EMA (S2)

Output default: folder `generated/` di dalam folder run checkpoint.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.diffusion import Diffusion
from src.model import build_model
from src.train import class_grid_labels, generate_grid
from src.utils import Config, get_device, save_image_grid


def load_checkpoint(path: str | Path, device: torch.device, use_ema: bool = True):
    """Muat checkpoint -> (cfg, model eval, diffusion, info). Bobot EMA dipakai secara default."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(state["config"])
    model = build_model(cfg)
    model.load_state_dict(state["ema"]["model"] if use_ema else state["model"])
    model.to(device).eval()
    diffusion = Diffusion.from_config(cfg).to(device)
    return cfg, model, diffusion, {"epoch": state["epoch"], "step": state["step"]}


@torch.no_grad()
def denoise_frames(
    model: torch.nn.Module, diffusion: Diffusion, cfg: Config, device: torch.device,
    labels: list[int], guidance_scale: float, n_frames: int = 20, seed: int = 0,
) -> list[torch.Tensor]:
    """Frame proses denoising untuk label tertentu (satu gambar per label).

    Return list frame (urut dari noise ke gambar), masing-masing (len(labels), C, H, W).
    """
    y = torch.tensor(labels, device=device)
    shape = (len(labels), cfg.data.channels, cfg.data.image_size, cfg.data.image_size)
    every = max(1, diffusion.timesteps // n_frames)
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=bool(cfg.train.amp) and device.type == "cuda"):
        _, frames = diffusion.sample(model, y, shape, guidance_scale=guidance_scale,
                                     generator=generator, save_every=every)
    return [f.float() for f in frames]


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description="Generate sampel dari checkpoint DDPM")
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--out", type=str, help="folder output (default: <run>/generated)")
    parser.add_argument("--guidance", type=float, nargs="*", help="satu/lebih guidance scale")
    parser.add_argument("--n-per-class", type=int, help="default dari config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", action="store_true", help="simpan strip proses denoising")
    parser.add_argument("--no-ema", action="store_true", help="pakai bobot non-EMA")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args(argv)

    device = get_device(args.device)
    ckpt = Path(args.ckpt)
    cfg, model, diffusion, info = load_checkpoint(ckpt, device, use_ema=not args.no_ema)
    out_dir = Path(args.out) if args.out else ckpt.parent.parent / "generated"
    tag = f"epoch{info['epoch']:03d}_{'raw' if args.no_ema else 'ema'}"
    n = args.n_per_class or cfg.sample.n_per_class
    scales = args.guidance if args.guidance else [cfg.sample.guidance_scale]

    saved = []
    for w in scales:
        t0 = time.perf_counter()
        grid = generate_grid(model, diffusion, cfg, device, guidance_scale=w,
                             n_per_class=n, seed=args.seed)
        path = out_dir / f"{tag}_w{w:g}.png"
        save_image_grid(grid, path, nrow=n)
        saved.append(path)
        print(f"w={w:g}: {grid.shape[0]} gambar -> {path} ({time.perf_counter() - t0:.1f}s)")

    if args.frames:
        labels = class_grid_labels(cfg.data.num_classes, 1, device).tolist()
        frames = denoise_frames(model, diffusion, cfg, device, labels, scales[0], seed=args.seed)
        strip = torch.stack(frames, dim=1).flatten(0, 1)  # (labels * frames, C, H, W)
        path = out_dir / f"{tag}_w{scales[0]:g}_frames.png"
        save_image_grid(strip, path, nrow=len(frames))  # satu baris per label
        saved.append(path)
        print(f"frames: {len(frames)} langkah -> {path}")
    return saved


if __name__ == "__main__":
    main()
