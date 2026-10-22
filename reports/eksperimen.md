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
| E1 | Baseline (`configs/cifar10.yaml`) | 100 | berjalan |
| E2 | lr 1e-4 | 30 | menunggu |
| E3 | lr 4e-4 | 30 | menunggu |
| E4 | batch 64 | 30 | menunggu |
| E5 | cosine schedule | 30 | menunggu |
| S1 | guidance w = 0, 1, 3, 5 (sampling saja) | — | menunggu |
| S2 | EMA vs non-EMA (sampling saja) | — | menunggu |
