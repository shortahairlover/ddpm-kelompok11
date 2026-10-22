"""Training loop DDPM class-conditional (T4.1-T4.4).

Contoh (dari root repo):
    python -m src.train --config configs/mnist.yaml
    python -m src.train --config configs/mnist.yaml --debug          # uji cepat (subset, T=50)
    python -m src.train --resume runs/mnist_20261022-150000          # lanjutkan run
    python -m src.train --config configs/cifar10.yaml --set train.lr=1e-4 --name e2_lr1e-4

Isi folder run: config.yaml, train.log, metrics.csv (per epoch), steps.csv (per log_every step),
tensorboard/, samples/epoch_XXX.png, checkpoints/epoch_XXX.pt (bobot) + last.pt (state penuh).
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from src.data import get_dataloader
from src.diffusion import Diffusion
from src.ema import EMA
from src.model import build_model, count_parameters
from src.utils import (
    Config,
    get_device,
    get_logger,
    load_config,
    make_run_dir,
    save_config,
    save_image_grid,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]

# Pengaturan --debug: cepat dijalankan di CPU maupun GPU
DEBUG_OVERRIDES = [
    "train.epochs=2", "train.log_every=1", "train.sample_every=1", "train.ckpt_every=1",
    "diffusion.timesteps=50", "data.num_workers=0", "sample.n_per_class=2",
]  # fmt: skip


def lr_at_step(step: int, base_lr: float, warmup_steps: int) -> float:
    """Warmup linear dari ~0 ke base_lr selama warmup_steps, lalu konstan."""
    if warmup_steps <= 0:
        return base_lr
    return base_lr * min(1.0, (step + 1) / warmup_steps)


def class_grid_labels(num_classes: int, n_per_class: int, device: torch.device) -> torch.Tensor:
    """Label [0,0,..,1,1,..] -> grid dengan nrow=n_per_class: satu baris per kelas. (N,)"""
    return torch.arange(num_classes, device=device).repeat_interleave(n_per_class)


@torch.no_grad()
def generate_grid(
    model: torch.nn.Module, diffusion: Diffusion, cfg: Config, device: torch.device,
    guidance_scale: float | None = None, n_per_class: int | None = None, seed: int = 0,
    use_amp: bool | None = None,
) -> torch.Tensor:
    """Sampel per kelas (num_classes * n_per_class gambar). Return (N, C, H, W) di [-1, 1]."""
    n = cfg.sample.n_per_class if n_per_class is None else n_per_class
    w = cfg.sample.guidance_scale if guidance_scale is None else guidance_scale
    amp = cfg.train.amp if use_amp is None else use_amp
    y = class_grid_labels(cfg.data.num_classes, n, device)
    shape = (y.shape[0], cfg.data.channels, cfg.data.image_size, cfg.data.image_size)
    generator = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()  # matikan dropout saat sampling
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=amp and device.type == "cuda"):
        images, _ = diffusion.sample(model, y, shape, guidance_scale=w, generator=generator)
    model.train(was_training)
    return images.float()


class CSVLogger:
    """Tulis baris dict ke CSV (header dibuat otomatis, mode append untuk resume)."""

    def __init__(self, path: Path, fields: list[str]):
        self.path, self.fields = path, fields
        if not path.exists():
            with open(path, "w", newline="", encoding="utf8") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def log(self, row: dict) -> None:
        with open(self.path, "a", newline="", encoding="utf8") as f:
            csv.DictWriter(f, fieldnames=self.fields).writerow(row)


def save_checkpoint(path: Path, **state) -> None:
    """Simpan atomik: tulis ke file sementara lalu ganti nama (aman bila proses terputus)."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def train(cfg: Config, run_dir: Path, device: torch.device, debug: bool = False,
          resume_state: dict | None = None) -> dict:
    """Jalankan training sesuai cfg. Return ringkasan (epoch terakhir, loss terakhir, dll.)."""
    logger = get_logger(f"train.{run_dir.name}", run_dir / "train.log")
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = device.type == "cuda"

    loader = get_dataloader(cfg, train=True, debug=debug)
    model = build_model(cfg).to(device)
    diffusion = Diffusion.from_config(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    ema = EMA(model, cfg.train.ema_decay)

    start_epoch, step = 1, 0
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        ema.load_state_dict(resume_state["ema"])
        opt.load_state_dict(resume_state["optimizer"])
        scaler.load_state_dict(resume_state["scaler"])
        start_epoch, step = resume_state["epoch"] + 1, resume_state["step"]
        torch.set_rng_state(resume_state["rng_cpu"])
        if device.type == "cuda" and resume_state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(resume_state["rng_cuda"])
        logger.info(f"resume dari epoch {resume_state['epoch']} (step {step})")

    epochs = cfg.train.epochs
    logger.info(f"run dir: {run_dir}")
    logger.info(f"device: {device} | amp: {use_amp} | data: {cfg.data.dataset} "
                f"({len(loader.dataset)} gambar, {len(loader)} batch/epoch) | debug: {debug}")
    logger.info(f"parameter model: {count_parameters(model):,} | epoch {start_epoch}..{epochs}")

    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    epoch_csv = CSVLogger(run_dir / "metrics.csv",
                          ["epoch", "step", "train_loss", "lr", "epoch_time_s", "images_per_s",
                           "max_mem_mb"])
    step_csv = CSVLogger(run_dir / "steps.csv", ["step", "epoch", "loss", "lr"])

    last_loss = math.nan
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        loss_sum, n_batches, interval_sum, interval_n = 0.0, 0, 0.0, 0

        for x0, y in loader:
            x0 = x0.to(device, non_blocking=True)  # (B, C, 32, 32) di [-1, 1]
            y = y.to(device, non_blocking=True)  # (B,)
            lr = lr_at_step(step, cfg.train.lr, cfg.train.warmup_steps)
            for group in opt.param_groups:
                group["lr"] = lr

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = diffusion.training_loss(model, x0, y)
            if not torch.isfinite(loss):
                logger.error(f"loss tidak finite ({loss.item()}) di epoch {epoch} step {step}")
                raise FloatingPointError("loss NaN/inf: turunkan lr atau matikan AMP")

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)  # agar clipping bekerja pada gradien asli
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            step += 1

            loss_value = loss.item()
            loss_sum += loss_value
            n_batches += 1
            interval_sum += loss_value
            interval_n += 1
            if step % cfg.train.log_every == 0:
                avg = interval_sum / interval_n
                writer.add_scalar("train/loss_step", avg, step)
                writer.add_scalar("train/lr", lr, step)
                step_csv.log({"step": step, "epoch": epoch, "loss": f"{avg:.6f}", "lr": lr})
                interval_sum, interval_n = 0.0, 0

        epoch_time = time.perf_counter() - t0
        last_loss = loss_sum / max(n_batches, 1)
        max_mem = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0
        n_images = n_batches * cfg.data.batch_size
        writer.add_scalar("train/loss_epoch", last_loss, epoch)
        epoch_csv.log({
            "epoch": epoch, "step": step, "train_loss": f"{last_loss:.6f}", "lr": lr,
            "epoch_time_s": f"{epoch_time:.1f}", "images_per_s": f"{n_images / epoch_time:.1f}",
            "max_mem_mb": f"{max_mem:.0f}",
        })
        logger.info(f"epoch {epoch}/{epochs} | loss {last_loss:.4f} | lr {lr:.2e} | "
                    f"{epoch_time:.1f}s | {n_images / epoch_time:.0f} img/s | "
                    f"VRAM puncak {max_mem:.0f} MB")

        is_last = epoch == epochs
        if epoch % cfg.train.sample_every == 0 or is_last:
            t_s = time.perf_counter()
            grid = generate_grid(ema.model, diffusion, cfg, device, seed=cfg.seed)
            path = run_dir / "samples" / f"epoch_{epoch:03d}.png"
            save_image_grid(grid, path, nrow=cfg.sample.n_per_class)
            writer.add_images("samples/ema", (grid[:: cfg.sample.n_per_class] + 1) / 2, epoch)
            logger.info(f"sampel disimpan: {path.name} ({time.perf_counter() - t_s:.1f}s)")

        if epoch % cfg.train.ckpt_every == 0 or is_last:
            weights = {"model": model.state_dict(), "ema": ema.state_dict(),
                       "config": cfg.to_dict(), "epoch": epoch, "step": step}
            save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", **weights)
            save_checkpoint(
                run_dir / "checkpoints" / "last.pt", **weights,
                optimizer=opt.state_dict(), scaler=scaler.state_dict(),
                rng_cpu=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            )
            logger.info(f"checkpoint disimpan: epoch_{epoch:03d}.pt + last.pt")

    writer.close()
    logger.info("training selesai")
    return {"epoch": epochs, "step": step, "train_loss": last_loss, "run_dir": str(run_dir)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training Mini DDPM class-conditional")
    parser.add_argument("--config", type=str, help="path config YAML (wajib kecuali --resume)")
    parser.add_argument("--resume", type=str, help="folder run yang dilanjutkan (berisi last.pt)")
    parser.add_argument("--device", type=str, default="auto", help="auto | cuda | cpu")
    parser.add_argument("--debug", action="store_true", help="subset kecil, 2 epoch, T=50")
    parser.add_argument("--epochs", type=int, help="override train.epochs")
    parser.add_argument("--name", type=str, help="nama run (default: cfg.name)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KUNCI=NILAI",
                        help="override config, misal train.lr=1e-4")
    parser.add_argument("--runs-root", type=str, default=str(ROOT / "runs"))
    args = parser.parse_args(argv)
    if not args.config and not args.resume:
        parser.error("butuh --config atau --resume")
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    device = get_device(args.device)

    if args.resume:
        run_dir = Path(args.resume)
        cfg = load_config(run_dir / "config.yaml", args.set)
        resume_state = torch.load(run_dir / "checkpoints" / "last.pt", map_location="cpu",
                                  weights_only=False)
        debug = bool(cfg.get("debug_run", False))
    else:
        debug = args.debug
        overrides = (DEBUG_OVERRIDES if debug else []) + list(args.set)
        cfg = load_config(args.config, overrides)
        cfg.debug_run = debug
        resume_state = None
        root = Path(args.runs_root) / ("debug" if debug else "")
        run_dir = make_run_dir(cfg, root=root, name=args.name)

    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    save_config(cfg, run_dir / "config.yaml")  # simpan nilai akhir (termasuk --epochs)
    return train(cfg, run_dir, device, debug=debug, resume_state=resume_state)


if __name__ == "__main__":
    main()
