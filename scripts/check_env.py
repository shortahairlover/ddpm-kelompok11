"""Cek lingkungan: versi library, GPU, dan uji komputasi CUDA + AMP.

Jalankan: python scripts/check_env.py
Exit code 0 kalau semua cek lolos, 1 kalau ada yang gagal.
"""

import platform
import sys
import time

import torch


def check_matmul(device: torch.device) -> float:
    """Kalikan dua matriks 2048x2048 dan bandingkan dengan hasil CPU. Return selisih maks."""
    a = torch.randn(2048, 2048)
    b = torch.randn(2048, 2048)
    expected = a @ b
    result = (a.to(device) @ b.to(device)).cpu()
    return (result - expected).abs().max().item()


def check_amp(device: torch.device) -> bool:
    """Satu langkah forward/backward dengan autocast + GradScaler; True kalau loss & grad finite."""
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.SiLU(), torch.nn.Conv2d(16, 3, 3, padding=1)
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device.type)
    x = torch.randn(8, 3, 32, 32, device=device)  # x: (B, C, H, W)
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        loss = torch.nn.functional.mse_loss(model(x), x)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    grads_ok = all(torch.isfinite(p.grad).all() for p in model.parameters())
    return bool(torch.isfinite(loss).item() and grads_ok)


def main() -> int:
    os_info = f"{platform.system()} build {platform.version()}"
    print(f"python          : {sys.version.split()[0]} ({os_info})")
    print(f"torch           : {torch.__version__}")
    try:
        import torchvision

        print(f"torchvision     : {torchvision.__version__}")
    except ImportError:
        print("torchvision     : TIDAK TERPASANG")
        return 1

    cuda_ok = torch.cuda.is_available()
    print(f"cuda available  : {cuda_ok}")
    if not cuda_ok:
        print("GAGAL: CUDA tidak tersedia (cek build PyTorch / driver).")
        return 1

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"cuda (torch)    : {torch.version.cuda}")
    print(f"gpu             : {props.name}")
    print(f"vram            : {props.total_memory / 1024**3:.2f} GB")
    print(f"bf16 supported  : {torch.cuda.is_bf16_supported()}")

    t0 = time.perf_counter()
    err = check_matmul(device)
    torch.cuda.synchronize()
    print(f"matmul GPU      : selisih maks vs CPU = {err:.2e} ({time.perf_counter() - t0:.2f} s)")
    # TF32/urutan reduksi bisa membuat sedikit beda; toleransi longgar untuk matriks 2048x2048
    matmul_ok = err < 1e-1

    amp_ok = check_amp(device)
    print(f"AMP fwd/bwd     : {'OK' if amp_ok else 'GAGAL'}")

    all_ok = matmul_ok and amp_ok
    print("STATUS          : " + ("SEMUA CEK LOLOS" if all_ok else "ADA CEK YANG GAGAL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
