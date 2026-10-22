# LOG INTERAKSI LLM — KELOMPOK 11

- **Tanggal Sesi:** 23/10/2026
- **Target Modul:** Fase 6 — Evaluasi & Visualisasi (FID, sampler DDIM, GIF denoising, grid progres, grafik final)
- **LLM:** Claude Code (Opus 5.5)
- **Tautan Sesi Chat:** transkrip sesi diekspor ke `logs/raw/`

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Meminta Kode Awal)
- **Prompt Kelompok:**
  > "saya ingin sebelum anda lanjut ke fase selanjutnya, tolong lakukan audit dari fase yang sudah di kerjakan tersebut, jika ada error atau bug tolong langsung solve sebelum di commit dan push."

  > "anda harus run task proyek ini dari awal hingga akhir itu tanpa berhenti."
- **Alasan Khusus:** Fase 6 membutuhkan bukti **kuantitatif** (FID) agar perbandingan eksperimen tidak hanya berdasarkan loss — terutama E5 (cosine) yang skala loss-nya berbeda — dan bukti **visual** untuk presentasi (GIF denoising, progres per epoch, efek guidance).
- **Output Model:**
  - `src/diffusion.py` → `ddim_sample`: sampler DDIM (eta, jumlah langkah bebas) memakai model yang sama tanpa training ulang.
  - `src/evaluate.py`: FID `torchmetrics` (Inception-v3 2048 fitur), gambar asli = 50.000 gambar train CIFAR-10 tanpa flip (dihitung sekali per proses, `reset_real_features=False`), label sampel seimbang per kelas, hasil ditambahkan ke `reports/fid_results.csv`.
  - `scripts/run_fid.ps1`: semua evaluasi FID berurutan (E1–E5 epoch 30; guidance sweep E1 epoch 100; DDIM vs DDPM).
  - `scripts/make_gif.py`: GIF proses denoising 10 kelas (grid 2×5, diperbesar nearest).
  - Grid progres per epoch: `progress_cifar10_e1.png`, `progress_mnist.png`.

  Cuplikan krusial — satu langkah DDIM:
  ```python
  x0_hat = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1.0, 1.0)
  eps = (x - a_t.sqrt() * x0_hat) / (1 - a_t).sqrt()  # konsisten dengan x0 ter-clamp
  sigma = eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
  x = a_prev.sqrt() * x0_hat + (1 - a_prev - sigma**2).clamp(min=0).sqrt() * eps + sigma * noise
  ```

### 2. TAHAP PENANGANAN EROR / DEBUGGING

**Masalah 1 — FID dengan DDPM-1000 terlalu lama**
- **Analisis:** sampling DDPM 80 gambar (w = 3) butuh ±134 s → 5.000 sampel ±2,3 jam per evaluasi; rencana T6.2 butuh ±9 evaluasi (±20 jam).
- **Solusi:** implementasi DDIM (sebelumnya opsional, T6.8) dengan 100 langkah (±10× lebih cepat). Keputusan dicatat di `PLANNING.md` Bagian H. Agar tidak sekadar "percaya", kebenaran DDIM dibuktikan: dengan `eta = 1` dan `steps = T`, mean & varians DDIM secara aljabar sama dengan posterior DDPM, sehingga hasilnya harus identik dengan urutan noise yang sama → `test_ddim_eta1_full_steps_equals_ddpm` lolos (`atol 1e-4`). Satu pembanding DDPM-1000 vs DDIM-100 (2.000 sampel) tetap dihitung.

**Masalah 2 — VRAM hampir penuh saat FID**
- **Gejala:** `nvidia-smi` → `100 %, 5905 MiB` saat FID (batch 250, CFG → 500 per forward).
- **Solusi:** pekerjaan lain Fase 6 dijalankan **di CPU** selama FID berjalan (`make_gif --device cpu`) agar tidak terjadi OOM yang menggagalkan rantai FID.

**Temuan 3 — Jumlah frame GIF terbaca 50, bukan 65**
- **Analisis:** encoder GIF (Pillow) menggabungkan frame identik berurutan (15 frame "jeda" di akhir) menjadi satu frame berdurasi lebih panjang. Bukan bug: jeda akhir tetap ada. Frame awal–tengah–akhir dicek visual: noise → angka 0–9 sesuai kelas.

**Temuan 4 — Sisa kode tidak berguna**
- Saat menulis `make_gif.py` sempat tertinggal baris `noise_start = ... if False else None  # noqa: F841` (tidak pernah dipakai, disembunyikan dari lint dengan `noqa`). Dihapus sebelum commit — `noqa` tidak boleh dipakai untuk menyembunyikan kode mati.

**Temuan 5 — Path absolut di `reports/fid_results.csv`**
- **Gejala:** baris FID pertama menyimpan `D:/Files Mata Kuliah/.../runs/e1_baseline_.../epoch_030.pt` (tidak portabel, membocorkan struktur folder pribadi ke repo).
- **Solusi:** fungsi `display_path` (path relatif terhadap root repo) + `tests/test_evaluate.py`. Perubahan dibuat kecil dan diuji di CPU karena tahap rantai FID berikutnya memuat ulang `src/evaluate.py`. Baris dari tahap yang sudah berjalan dinormalkan setelah rantai selesai.
- **Hasil FID pertama:** E1 epoch 30, w = 3, DDIM-100, 5.000 sampel → **FID 31,00** (22,4 menit; statistik 50.000 gambar asli ±12 menit).

### 3. TAHAP OPTIMASI / HASIL EVALUASI
- Rantai `scripts/run_fid.ps1` selesai 01:03–06:40 tanpa error (12 nilai FID di `reports/fid_results.csv`; path absolut tahap pertama dinormalkan menjadi relatif).
- **Eksperimen epoch 30 (w = 3):** E4 batch 64 **23,37** < E1 31,00 < E2 34,08 < E3 35,00 < E5 cosine 45,26.
  - Analisis kelompok: E4 unggul per epoch karena melakukan 2× jumlah update; lr baseline 2e-4 berada di titik tengah terbaik; loss training lebih kecil (E3) tidak menjamin FID lebih baik; cosine kalah pada CIFAR-10 32×32.
- **Guidance sweep (E1 epoch 100):** w = 0/1/2/3/5 → 26,52 / 22,70 / **16,43** / 18,01 / 26,30 — kurva U (trade-off kualitas–keragaman CFG). Default `sample.guidance_scale` CIFAR-10 diubah 3,0 → **2,0** (dicatat di Bagian H).
- **DDIM-100 vs DDPM-1000 (2.000 sampel):** FID 26,23 vs 26,64, waktu 8,9 vs 85,7 menit → DDIM valid sebagai sampler evaluasi dan ±10× lebih cepat.
- **Model terbaik: E1 epoch 100, EMA, w = 2 → FID 16,43.**
- Visual: grid final `cifar10_final_w2.png`, guidance `s1_guidance_cifar10.png` (dengan nilai FID per panel) & `guidance_mnist.png`, GIF `denoise_cifar10.gif` & `denoise_mnist.gif`, progres `progress_cifar10_e1.png` & `progress_mnist.png`, grafik `fid_experiments_epoch30.png` & `fid_guidance_e1.png`.
