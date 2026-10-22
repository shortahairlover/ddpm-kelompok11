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
| E2 | lr 1e-4 | 30 | selesai (19:52–21:02) |
| E3 | lr 4e-4 | 30 | selesai (21:02–22:14) |
| E4 | batch 64 | 30 | selesai (22:14–23:33) |
| E5 | cosine schedule | 30 | selesai (23:33–00:42) |
| S1 | guidance w = 0, 1, 3, 5 (sampling saja, checkpoint E1 epoch 100) | — | selesai |
| S2 | EMA vs non-EMA (sampling saja, checkpoint E1 epoch 100) | — | selesai |

## Hasil E1 — baseline (100 epoch)
- Run: `runs/e1_baseline_20261022-160411/` (tidak di-commit), log ±123 s/epoch, VRAM puncak 3.698 MB.
- Loss per epoch: 0,5590 (ep 1) → 0,0464 (ep 2) → 0,0334 (ep 10) → 0,0321 (ep 30) → 0,0311 (ep 50) → **0,0302 (ep 100)**. Grafik: `reports/figures/loss_cifar10_e1.png`.
- Visual (`reports/figures/cifar10_e1_epoch100_w3.png`, EMA, w = 3): setiap baris jelas sesuai kelas (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck); bentuk & warna objek meyakinkan, detail halus masih buram (wajar untuk model 9 juta parameter di 32×32).

## Perbandingan E1–E5 pada epoch 30 (T5.5, T5.8)
Semua eksperimen memakai seed 42 dan sampling dengan seed yang sama (grid EMA, w = 3), sehingga perbedaan gambar berasal dari hyperparameter.

| ID | Config | Step s/d ep 30 | Waktu 30 epoch | VRAM puncak | Loss ep 30 | Rata-rata loss ep 26–30 | FID (5k) | Catatan visual (ep 30) |
|---|---|---|---|---|---|---|---|---|
| E1 | lr 2e-4, batch 128, linear | 11.700 | ±62 menit | 3.698 MB | 0,0321 | 0,0315 | Fase 6 | Objek sesuai kelas, bentuk jelas |
| E2 | **lr 1e-4** | 11.700 | ±63 menit | 3.698 MB | 0,0325 | 0,0319 | Fase 6 | Sangat mirip E1; loss turun paling lambat di awal (0,729 di ep 1) |
| E3 | **lr 4e-4** | 11.700 | ±65 menit | 3.698 MB | 0,0320 | 0,0313 | Fase 6 | Sangat mirip E1; stabil (tidak ada NaN/lonjakan loss) |
| E4 | **batch 64** | 23.430 | ±72 menit | 1.953 MB | 0,0310 | 0,0312 | Fase 6 | Sangat mirip E1; 2× jumlah update, VRAM ±separuh |
| E5 | **cosine schedule** | 11.700 | ±63 menit | 3.698 MB | 0,0591 | 0,0583* | Fase 6 | Komposisi berbeda dari E1–E4 (lintasan noise berbeda), warna sedikit lebih pudar |

\* Loss E5 **tidak sebanding** dengan E1–E4: schedule cosine menaruh lebih banyak timestep pada tingkat noise rendah–menengah sehingga rata-rata MSE prediksi noise per batch berbeda skalanya. Perbandingan kualitas yang adil memakai FID (Fase 6).

Gambar: `reports/figures/cmp_experiments_epoch30.png`, `reports/figures/loss_cmp_experiments.png`.

**Pengamatan sementara (sebelum FID):** pada rentang yang diuji, lr dan batch size hampir tidak mengubah loss akhir (selisih < 0,001) maupun tampilan sampel; perbedaan terbesar ada pada kecepatan turun di epoch awal (lr kecil paling lambat). **Checkpoint terbaik sementara: E1 epoch 100** (satu-satunya run 100 epoch; E2–E4 setara E1 di epoch 30). Dikonfirmasi dengan FID di Fase 6 (T6.2).

## S1 — Efek guidance scale (T5.6)
Checkpoint E1 epoch 100, EMA, seed sampling sama. Gambar: `reports/figures/s1_guidance_cifar10.png`.

| w | Waktu 80 gambar | Pengamatan |
|---|---|---|
| 0 | 57 s | Tanpa kondisi: gambar realistis tetapi **kelas acak** (baris tidak konsisten) |
| 1 | 62 s | Kondisional murni: sebagian besar sesuai kelas, beberapa masih meleset (mis. baris kucing/anjing) |
| 3 | 134 s | Setiap baris jelas sesuai kelas, objek lebih tegas |
| 5 | 134 s | Paling sesuai kelas, warna lebih jenuh/kontras, **variasi antar sampel berkurang** |

w = 0 dan w = 1 lebih cepat karena hanya butuh satu forward per langkah (tanpa batch cond+uncond).

## S2 — EMA vs non-EMA (T5.7)
Checkpoint E1 epoch 100, w = 3. Gambar: `reports/figures/s2_ema_vs_raw.png`.
- **EMA (0,9999):** sampel lebih halus dan bersih, latar belakang konsisten.
- **Non-EMA:** tekstur lebih berisik, lebih banyak artefak warna/bintik, beberapa objek kurang utuh.
- Kesimpulan: EMA memperbaiki kualitas sampel secara nyata tanpa biaya training tambahan; semua hasil akhir memakai bobot EMA.
