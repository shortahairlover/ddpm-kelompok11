# LOG INTERAKSI LLM — KELOMPOK 11

- **Tanggal Sesi:** 22/10/2026
- **Target Modul:** Fase 5 — Training CIFAR-10 & Eksperimen Tuning (benchmark, E1 baseline, E2–E5, S1, S2)
- **LLM:** Claude Code (Opus 5.5)
- **Tautan Sesi Chat:** transkrip sesi diekspor ke `logs/raw/`

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Meminta Kode Awal)
- **Prompt Kelompok:**
  > "untuk Mini Diffusion Model (DDPM) training nya pakai data apa?"

  > "buatlah planning nya untuk mengerjakan project 1 Mini Diffusion Model (DDPM) ini menggunakan cifar-10 dan mnist, ini mengerjakan di lokal"

  > "anda harus run task proyek ini dari awal hingga akhir itu tanpa berhenti."
- **Alasan Khusus:** Laptop hanya punya 1 GPU 6 GB, jadi lama training harus diukur dulu sebelum menjalankan 100 epoch (aturan T5.2: kalau > 8 jam, epoch dikurangi). Eksperimen tuning dibuat **satu variabel per eksperimen** dengan pewarisan config (`base: ../cifar10.yaml`) agar perbandingan adil, dan dijalankan 30 epoch lalu dibandingkan dengan E1 pada epoch yang sama.
- **Output Model:**
  - `scripts/benchmark_step.py`: mengukur waktu/step, VRAM puncak, estimasi waktu per epoch & total, serta waktu sampling satu grid.
  - `configs/experiments/e2_lr1e-4.yaml`, `e3_lr4e-4.yaml`, `e4_batch64.yaml`, `e5_cosine.yaml` + test yang memastikan **hanya satu kunci** yang berbeda dari baseline (selain nama & epoch).
  - `reports/eksperimen.md`: estimasi sumber daya dan tabel eksperimen.

  Hasil benchmark (RTX 4050 Laptop 6 GB, AMP):
  ```
  parameter         : 8,954,883
  waktu/step        : 290 ms (batch 128)
  waktu/epoch       : 1.9 menit (390 step)
  total 100 epoch    : 3.1 jam (tanpa sampling)
  VRAM puncak train : 3697 MB
  sampling 1 grid   : 2.1 menit (80 gambar, w=3.0)
  ```

### 2. TAHAP PENANGANAN EROR / DEBUGGING
**Eror 1 — Lint di skrip benchmark**
- **Pesan Eror:**
  ```
  E501 Line too long (101 > 100)
     --> scripts\benchmark_step.py:101:101
  ```
- **Solusi:** jumlah gambar grid dipindah ke variabel `n_grid`. `ruff check .` → `All checks passed!`.

**Temuan 2 — Pola `.gitignore` duplikat**
- Pola `e*_console.log` yang ditambahkan untuk log eksperimen ternyata sudah tercakup oleh `*_console.log` (dicek dengan `git check-ignore -v e1_console.log` → `.gitignore:22:*_console.log`). Pola duplikat dihapus.

**Pemantauan E1 (epoch awal)**
- `epoch 1/100 | loss 0.5590 | 142.9s` (termasuk warmup lr & inisialisasi cuDNN), `epoch 2/100 | loss 0.0464 | 122.8s`, VRAM puncak 3.697 MB, GPU 100 %, suhu 72 °C → training stabil, tidak ada NaN.

**Bug 3 — Skrip `run_experiments.ps1` (Windows PowerShell 5.1)**
- **Gejala:** log konsol E2 tersimpan sebagai UTF-16 (terbaca `2 0 2 6 - 1 0 - 2 2 ...` saat dibaca alat lain).
- **Analisis:** redirect `*>` di PowerShell 5.1 menulis UTF-16. Saat menelusuri, ditemukan risiko yang lebih serius: dengan `$ErrorActionPreference = "Stop"`, satu baris stderr dari Python (misal warning library) akan diubah menjadi `NativeCommandError` dan **menghentikan seluruh rantai eksperimen**. Dicek: log E1 & MNIST tidak berisi baris non-INFO (`grep -vc "| INFO |"` → 0), sehingga run E2 yang sudah berjalan aman.
- **Solusi:** `$ErrorActionPreference = "Continue"` + keberhasilan dicek lewat `$LASTEXITCODE`, output ditulis dengan `2>&1 | Out-File -Encoding utf8`. Diuji tanpa GPU dengan nama eksperimen fiktif:
  ```
  GAGAL: tidak_ada (exit 1), lihat ...\tidak_ada_console.log
  tidak_ada_console.log: Unicode text, UTF-8 (with BOM) text
  FileNotFoundError: [Errno 2] No such file or directory: 'configs\experiments\tidak_ada.yaml'
  ```
- **Pelajaran:** perilaku shell berbeda antar versi; skrip otomasi harus dites jalur gagalnya, bukan hanya jalur sukses.

### 3. HASIL E1 (baseline, 100 epoch)
- Loss 0,5590 → 0,0464 (ep 2) → 0,0334 (ep 10) → **0,0302 (ep 100)**; ±123 s/epoch; total ±3,8 jam.
- Grid EMA w = 3 (`reports/figures/cifar10_e1_epoch100_w3.png`): setiap baris sesuai kelas CIFAR-10.

### 4. HASIL EKSPERIMEN E2–E5, S1, S2
- Rantai E2→E5 berjalan tanpa gangguan 19:52–00:42 (`semua eksperimen selesai`, exit 0).
- Rata-rata loss epoch 26–30: E1 0,0315 · E2 0,0319 · E3 0,0313 · E4 0,0312 · E5 0,0583*.
- **Temuan analisis:** loss E5 (cosine) ±2× lebih besar, tetapi itu **bukan berarti lebih buruk**: schedule berbeda → distribusi tingkat noise per timestep berbeda → skala MSE berbeda. Karena itu tabel diberi catatan dan perbandingan kualitas ditunda ke FID (Fase 6).
- **Temuan reprodusibilitas:** grafik perbandingan sempat dibuat dengan memotong CSV E1 secara manual di folder sementara. Agar bisa direproduksi, `plot_metrics.py` ditambah opsi `--max-x` (+ test `test_read_curve_max_x`) dan grafik dibuat ulang dengan satu perintah.
- **S1 (guidance):** w = 0 kelas acak; w = 1 sebagian besar sesuai; w = 3 jelas sesuai kelas; w = 5 paling sesuai tetapi warna jenuh & variasi berkurang (trade-off kualitas–keragaman CFG).
- **S2 (EMA):** sampel EMA lebih halus dan bersih; non-EMA berisik dan banyak artefak.
- Ringkasan lengkap: `reports/eksperimen.md`.
