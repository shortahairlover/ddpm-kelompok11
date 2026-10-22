"""Ukur kecepatan & VRAM satu langkah training dan sampling (T5.1-T5.2).

Jalankan dari root repo:
    python -m scripts.benchmark_step --config configs/cifar10.yaml [--set data.batch_size=64]

Mencetak waktu per step training, perkiraan waktu per epoch & total training, VRAM puncak,
serta perkiraan waktu sampling grid per kelas.
"""

from __future__ import annotations

import argparse
import time

import torch

from src.data import get_dataset
from src.diffusion import Diffusion
from src.ema import EMA
from src.model import build_model, count_parameters
from src.train import lr_at_step
from src.utils import get_device, load_config, set_seed


def benchmark(cfg, device: torch.device, warmup: int = 10, steps: int = 40) -> dict:
    """Jalankan `warmup + steps` langkah training dengan data acak berukuran sama."""
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    model = build_model(cfg).to(device)
    diffusion = Diffusion.from_config(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.train.amp)
    ema = EMA(model, cfg.train.ema_decay)
    b, c, s = cfg.data.batch_size, cfg.data.channels, cfg.data.image_size
    x0 = torch.rand(b, c, s, s, device=device) * 2 - 1  # (B, C, H, W)
    y = torch.randint(0, cfg.data.num_classes, (b,), device=device)

    torch.cuda.reset_peak_memory_stats(device)
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        for g in opt.param_groups:
            g["lr"] = lr_at_step(i, cfg.train.lr, cfg.train.warmup_steps)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.train.amp):
            loss = diffusion.training_loss(model, x0, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(opt)
        scaler.update()
        ema.update(model)
    torch.cuda.synchronize()
    step_time = (time.perf_counter() - t0) / steps
    train_mem = torch.cuda.max_memory_allocated(device) / 2**20

    # sampling: ukur 20 langkah reverse untuk grid per kelas (dengan CFG -> batch 2N)
    n = cfg.data.num_classes * cfg.sample.n_per_class
    ema.model.eval()
    yy = torch.arange(cfg.data.num_classes, device=device).repeat_interleave(cfg.sample.n_per_class)
    xx = torch.randn(n, c, s, s, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for i in range(25):
            if i == 5:
                torch.cuda.synchronize()
                t1 = time.perf_counter()
            t = torch.full((n,), 999 - i, device=device, dtype=torch.long)
            xx = diffusion.p_sample(ema.model, xx, t, yy, cfg.sample.guidance_scale)
    torch.cuda.synchronize()
    sample_step = (time.perf_counter() - t1) / 20

    train_size = len(get_dataset(cfg, train=True))
    steps_per_epoch = train_size // b
    return {
        "params": count_parameters(model),
        "step_time_s": step_time,
        "epoch_time_min": step_time * steps_per_epoch / 60,
        "total_hours": step_time * steps_per_epoch * cfg.train.epochs / 3600,
        "train_mem_mb": train_mem,
        "sample_grid_min": sample_step * diffusion.timesteps / 60,
        "steps_per_epoch": steps_per_epoch,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Benchmark kecepatan & VRAM training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KUNCI=NILAI")
    args = parser.parse_args(argv)
    device = get_device("cuda")
    cfg = load_config(args.config, args.set)
    r = benchmark(cfg, device)
    print(f"config            : {args.config} {' '.join(args.set)}")
    print(f"parameter         : {r['params']:,}")
    print(f"waktu/step        : {r['step_time_s'] * 1000:.0f} ms (batch {cfg.data.batch_size})")
    print(f"waktu/epoch       : {r['epoch_time_min']:.1f} menit ({r['steps_per_epoch']} step)")
    print(f"total {cfg.train.epochs} epoch    : {r['total_hours']:.1f} jam (tanpa sampling)")
    print(f"VRAM puncak train : {r['train_mem_mb']:.0f} MB")
    n_grid = cfg.data.num_classes * cfg.sample.n_per_class
    print(f"sampling 1 grid   : {r['sample_grid_min']:.1f} menit "
          f"({n_grid} gambar, w={cfg.sample.guidance_scale})")
    return r


if __name__ == "__main__":
    main()
