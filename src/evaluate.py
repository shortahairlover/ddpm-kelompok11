"""Evaluasi FID (Frechet Inception Distance) checkpoint DDPM (T6.1-T6.2).

FID membandingkan statistik fitur Inception-v3 (2048 dim) gambar hasil generate dengan gambar
asli CIFAR-10 (50.000 gambar train, tanpa augmentasi). Makin kecil makin baik.
Inception hanya dipakai sebagai alat ukur (bobot pretrained bawaan torchmetrics/torch-fidelity),
bukan bagian dari model generatif.

Contoh (dari root repo):
    python -m src.evaluate --ckpt runs/e1/checkpoints/last.pt --labels E1 --guidance 3
    python -m src.evaluate --ckpt a.pt b.pt --labels E1 E2 --guidance 0 1 3 5 --sampler ddim

Catatan: FID dengan 5.000 sampel bias ke atas dibanding FID 50k di paper; angka hanya untuk
membandingkan eksperimen di proyek ini (jumlah sampel, sampler, dan seed sama).
Hasil ditambahkan ke CSV (default reports/fid_results.csv).
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets

from src.data import _data_root, build_transform, denormalize
from src.sample import load_checkpoint
from src.utils import Config, get_device

ROOT = Path(__file__).resolve().parents[1]
CSV_FIELDS = ["waktu", "label", "ckpt", "epoch", "guidance", "sampler", "steps", "num_samples",
              "fid", "durasi_menit"]


def balanced_labels(num_samples: int, num_classes: int, device: torch.device) -> torch.Tensor:
    """Label seimbang per kelas: [0..9, 0..9, ...] sepanjang num_samples. (N,)"""
    return torch.arange(num_samples, device=device) % num_classes


@torch.no_grad()
def add_real_features(fid: FrechetInceptionDistance, cfg: Config, device: torch.device,
                      batch_size: int = 500, limit: int | None = None) -> int:
    """Masukkan gambar asli (train, tanpa flip) ke metrik FID. Return jumlah gambar."""
    data_cfg = cfg.data
    tf = build_transform(data_cfg, train=False)
    root = _data_root(data_cfg)
    if data_cfg.dataset == "cifar10":
        ds = datasets.CIFAR10(root, train=True, download=True, transform=tf)
    else:
        ds = datasets.MNIST(root, train=True, download=True, transform=tf)
    if limit is not None:
        ds = torch.utils.data.Subset(ds, range(min(limit, len(ds))))
    n = 0
    for x, _ in DataLoader(ds, batch_size=batch_size, num_workers=2):
        fid.update(to_fid_input(x.to(device)), real=True)
        n += x.shape[0]
    return n


def to_fid_input(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] (B, C, H, W) -> [0, 1] float, 3 channel (MNIST diulang ke RGB)."""
    x = denormalize(x.float())
    return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x


@torch.no_grad()
def generate_samples(model, diffusion, cfg: Config, device: torch.device, num_samples: int,
                     batch_size: int, guidance: float, sampler: str, steps: int, seed: int,
                     on_batch=None) -> None:
    """Generate num_samples gambar per batch; panggil on_batch(images [-1,1]) tiap batch."""
    labels = balanced_labels(num_samples, cfg.data.num_classes, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    c, s = cfg.data.channels, cfg.data.image_size
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    for start in range(0, num_samples, batch_size):
        y = labels[start : start + batch_size]
        shape = (y.shape[0], c, s, s)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            if sampler == "ddim":
                images = diffusion.ddim_sample(model, y, shape, steps=steps,
                                               guidance_scale=guidance, generator=generator)
            else:
                images, _ = diffusion.sample(model, y, shape, guidance_scale=guidance,
                                             generator=generator)
        on_batch(images.float())


def display_path(path: Path) -> str:
    """Path relatif terhadap root repo (portabel di CSV); path di luar repo dibiarkan."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main(argv: list[str] | None = None) -> list[dict]:
    parser = argparse.ArgumentParser(description="Hitung FID checkpoint DDPM")
    parser.add_argument("--ckpt", nargs="+", required=True, type=Path)
    parser.add_argument("--labels", nargs="+", help="nama tiap checkpoint (default: nama run)")
    parser.add_argument("--guidance", nargs="+", type=float, default=[3.0])
    parser.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    parser.add_argument("--steps", type=int, default=100, help="langkah DDIM")
    parser.add_argument("--num-samples", type=int, help="default cfg.eval.fid_num_samples")
    parser.add_argument("--batch-size", type=int, help="default cfg.eval.fid_batch_size")
    parser.add_argument("--real-limit", type=int, help="batasi jumlah gambar asli (uji cepat)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "fid_results.csv")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    labels = args.labels or [p.parent.parent.name for p in args.ckpt]
    if len(labels) != len(args.ckpt):
        parser.error("jumlah --labels harus sama dengan jumlah --ckpt")

    device = get_device(args.device)
    fid = FrechetInceptionDistance(feature=2048, normalize=True,
                                   reset_real_features=False).to(device)
    results, real_cfg = [], None
    for ckpt, label in zip(args.ckpt, labels, strict=True):
        cfg, model, diffusion, info = load_checkpoint(ckpt, device, use_ema=True)
        if real_cfg is None:  # statistik gambar asli dihitung sekali, dipakai semua checkpoint
            t0 = time.perf_counter()
            n_real = add_real_features(fid, cfg, device, limit=args.real_limit)
            print(f"fitur gambar asli: {n_real} gambar ({time.perf_counter() - t0:.0f}s)")
            real_cfg = cfg
        elif cfg.data.dataset != real_cfg.data.dataset:
            raise ValueError("semua checkpoint harus dari dataset yang sama")
        n = args.num_samples or cfg.eval.fid_num_samples
        bs = args.batch_size or cfg.eval.fid_batch_size
        steps = args.steps if args.sampler == "ddim" else diffusion.timesteps
        for w in args.guidance:
            fid.reset()  # hanya fitur palsu yang direset (reset_real_features=False)
            t0 = time.perf_counter()
            generate_samples(model, diffusion, cfg, device, n, bs, w, args.sampler, steps,
                             args.seed, on_batch=lambda x: fid.update(to_fid_input(x), real=False))
            score = float(fid.compute())
            minutes = (time.perf_counter() - t0) / 60
            row = {"waktu": datetime.now().strftime("%Y-%m-%d %H:%M"), "label": label,
                   "ckpt": display_path(ckpt), "epoch": info["epoch"], "guidance": w,
                   "sampler": args.sampler, "steps": steps, "num_samples": n,
                   "fid": f"{score:.2f}", "durasi_menit": f"{minutes:.1f}"}
            append_csv(args.out, row)
            results.append(row)
            print(f"{label} ep{info['epoch']} w={w:g} {args.sampler}-{steps}: FID {score:.2f} "
                  f"({minutes:.1f} menit)")
    return results


if __name__ == "__main__":
    main()
