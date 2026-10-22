# Eksperimen CIFAR-10

## Estimasi sumber daya (T5.1–T5.2)
Diukur dengan `python -m scripts.benchmark_step --config configs/cifar10.yaml` pada RTX 4050 Laptop 6 GB (AMP fp16):

| Item | Nilai |
|---|---|
| Parameter U-Net | 8.954.883 |
| Waktu per step (batch 128) | ±290 ms |
| Waktu per epoch (390 step) | ±1,9 menit |
| Estimasi E1 100 epoch | ±3,1 jam + sampling (±2,1 menit × 10) ≈ **3,5 jam** |
| VRAM puncak training | ±3,7 GB (muat di 6 GB, batch 128 tidak perlu diturunkan) |
| Sampling 1 grid (80 gambar, 1000 langkah, CFG w = 3) | ±2,1 menit |

Estimasi < 8 jam, sehingga E1 dijalankan penuh 100 epoch sesuai Bagian D.

## Daftar eksperimen
| ID | Perubahan dari baseline | Epoch | Status |
|---|---|---|---|
| E1 | Baseline (`configs/cifar10.yaml`) | 100 | selesai (22/10 16:04–19:52, ±3,8 jam) |
| E2 | lr 1e-4 | 30 | berjalan |
| E3 | lr 4e-4 | 30 | menunggu |
| E4 | batch 64 | 30 | menunggu |
| E5 | cosine schedule | 30 | menunggu |
| S1 | guidance w = 0, 1, 3, 5 (sampling saja) | — | menunggu |
| S2 | EMA vs non-EMA (sampling saja) | — | menunggu |

## Hasil E1 — baseline (100 epoch)
- Run: `runs/e1_baseline_20261022-160411/` (tidak di-commit), log ±123 s/epoch, VRAM puncak 3.698 MB.
- Loss per epoch: 0,5590 (ep 1) → 0,0464 (ep 2) → 0,0334 (ep 10) → 0,0321 (ep 30) → 0,0311 (ep 50) → **0,0302 (ep 100)**. Grafik: `reports/figures/loss_cifar10_e1.png`.
- Visual (`reports/figures/cifar10_e1_epoch100_w3.png`, EMA, w = 3): setiap baris jelas sesuai kelas (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck); bentuk & warna objek meyakinkan, detail halus masih buram (wajar untuk model 9 juta parameter di 32×32).
