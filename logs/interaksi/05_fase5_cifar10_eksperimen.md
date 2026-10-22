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
