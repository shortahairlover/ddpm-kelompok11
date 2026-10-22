# LOG INTERAKSI LLM — KELOMPOK 11

- **Tanggal Sesi:** 22/10/2026
- **Target Modul:** Fase 4 — Training Loop & Sanity Run MNIST (`src/train.py`, `src/sample.py`, `scripts/plot_metrics.py`)
- **LLM:** Claude Code (Opus 5.5)
- **Tautan Sesi Chat:** transkrip sesi diekspor ke `logs/raw/`

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Meminta Kode Awal)
- **Prompt Kelompok:**
  > "anda harus run task proyek ini dari awal hingga akhir itu tanpa berhenti."

  > "apakah sudah? ini akan langsung menjalankan semua fase kan?"
- **Alasan Khusus:** Training CIFAR-10 di laptop memakan waktu berjam-jam dan bisa terputus (sleep, listrik). Karena itu training loop dirancang **bisa dilanjutkan** (`--resume`) dengan checkpoint atomik, semua metrik tercatat ke file (bukan hanya konsol), dan ada mode `--debug` supaya seluruh pipeline bisa diuji dalam hitungan detik sebelum menjalankan training panjang.
- **Output Model:**
  - `src/train.py`:
    - AdamW + **warmup lr linear**, **AMP fp16** (`autocast` + `GradScaler`), `unscale_` sebelum **grad clipping 1.0**, update **EMA** setiap step.
    - Penjaga NaN: training berhenti dengan pesan jelas kalau loss tidak finite.
    - Logging: `train.log`, `metrics.csv` (loss/epoch, waktu, img/s, VRAM puncak), `steps.csv` (loss per `log_every` step), TensorBoard.
    - Grid sampel per kelas dari model EMA (satu baris per kelas, seed tetap agar progres antar-epoch bisa dibandingkan).
    - Checkpoint `epoch_XXX.pt` (bobot + EMA + config) dan `last.pt` (state penuh: optimizer, scaler, RNG); disimpan atomik (`.tmp` → rename).
    - CLI: `--config`, `--resume`, `--debug`, `--epochs`, `--name`, `--set kunci=nilai`, `--device`.
  - `src/sample.py`: muat checkpoint (EMA atau `--no-ema`), grid per kelas untuk beberapa guidance scale, strip proses denoising (`--frames`).
  - `scripts/plot_metrics.py`: kurva loss dari satu/beberapa run (per epoch atau per step + smoothing).
  - `tests/test_train.py`: warmup lr, label grid, moving average, dan **test end-to-end**: train `--debug` → resume ke epoch 3 → sampling → plot.

  Cuplikan krusial — langkah optimisasi dengan AMP:
  ```python
  with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
      loss = diffusion.training_loss(model, x0, y)
  opt.zero_grad(set_to_none=True)
  scaler.scale(loss).backward()
  scaler.unscale_(opt)  # agar clipping bekerja pada gradien asli
  torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
  scaler.step(opt)
  scaler.update()
  ema.update(model)
  ```

### 2. TAHAP PENANGANAN EROR / DEBUGGING
**Uji `--debug` dan `--resume`**
- `python -m src.train --config configs/mnist.yaml --debug` → 2 epoch × 4 step, sampel & checkpoint tersimpan, VRAM puncak ±2,3 GB.
- `python -m src.train --resume <run> --epochs 3` → lanjut dari epoch 3 / step 8, lr warmup berlanjut (4,8e-6), `metrics.csv` berisi epoch 1, 2, 3 (tersambung, tidak ditimpa).
- **Temuan:** epoch pertama setiap proses jauh lebih lambat (18,5 s vs 0,7 s untuk 4 step). **Analisis:** bukan bug; itu waktu inisialisasi CUDA + `cudnn.benchmark` yang mencari algoritma konvolusi tercepat di awal. Pada training penuh, overhead ini hanya sekali di awal.
- **Temuan:** `--frames` dengan T = 50 menghasilkan 25 frame (bukan 20) karena interval = T // 20 = 2. Pada T = 1000 interval = 50 → 20 frame sesuai harapan; perilaku ini didokumentasikan.

### 3. TAHAP OPTIMASI / HASIL SANITY RUN MNIST
- **Perintah:** `python -m src.train --config configs/mnist.yaml --name mnist_sanity` (20 epoch, 9.360 step, batch 128, lr 2e-4, EMA 0,999, AMP).
- **Kecepatan:** ±76 s/epoch (±780 img/s), VRAM puncak ±2,25 GB, sampling 80 gambar × 1000 langkah + CFG ±65 s. Total ±27 menit.
- **Loss:** 0,356 (epoch 1, termasuk fase warmup) → 0,0227 (epoch 2) → **0,0167** (epoch 20). Grafik: `reports/figures/loss_mnist.png`.
- **Hasil visual** (`reports/figures/mnist_grid_final.png`, guidance w = 2, model EMA): setiap baris berisi angka yang **sesuai label** 0–9, terbaca jelas, dan gaya tulisannya bervariasi (tidak *mode collapse*). → **Sanity run LULUS**: pipeline data → model → loss → training → EMA → sampling CFG terbukti benar end-to-end.
