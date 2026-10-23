# Log Interaksi LLM — Kelompok 11

Rekam jejak kolaborasi dengan **Claude Code (Opus 5.5)** selama Proyek 1 (Mini DDPM), disusun kronologis:
**inisiasi → penanganan eror → optimasi**. Format setiap log mengikuti template dosen
([`TEMPLATE_log_interaksi.md`](TEMPLATE_log_interaksi.md)).

## Cara membaca
- `interaksi/` — log terstruktur per fase: prompt kelompok (apa adanya), alasan, output model, eror utuh + analisis + solusi.
- `raw/` — transkrip lengkap sesi Claude Code (dikonversi dari ekspor sesi dengan `scripts/transcript_to_md.py`; Claude Code tidak menyediakan *share link*). Data mahasiswa kelompok lain yang sempat terbaca dari file Daftar Kelompok disensor (output tool pembaca dokumen + daftar nama/NIM yang disimpan di luar repo, opsi `--redact-file`).
- Bukti audit setiap fase ada di [`../audits/`](../audits/), riwayat kode di Pull Request GitHub per fase.

## Indeks kronologis

| # | Tanggal | Fase / modul | Inisiasi (ringkas) | Eror & debugging utama | Optimasi / hasil |
|---|---|---|---|---|---|
| 00 | 22/10/2026 | [Fase 0 — Persiapan tools & GitHub](interaksi/00_fase0_persiapan_tools.md) | GitHub CLI, venv Python 3.11, PyTorch 2.14.1 + CUDA 13.0, `check_env.py` | GitHub timeout (jaringan), `ruff E501`, login `gh` belum tersimpan | GPU RTX 4050 terverifikasi (matmul + AMP) |
| 01 | 22/10/2026 | [Fase 1 — Fondasi repo](interaksi/01_fase1_fondasi.md) | `CLAUDE.md` (system prompt), config YAML, `utils.py` | Heredoc shell gagal parse; override `1e-4` terbaca string | Pewarisan config `base:`, run dir anti-tabrakan |
| 02 | 22/10/2026 | [Fase 2 — Data pipeline](interaksi/02_fase2_data_pipeline.md) | Loader MNIST (pad 32) & CIFAR-10 (flip), normalisasi [-1, 1] | `Lambda` tidak bisa di-pickle (worker Windows) → `Normalize`; unduhan CIFAR lambat | Grid batch & 11 test data |
| 03 | 22/10/2026 | [Fase 3 — Model & diffusion](interaksi/03_fase3_model_diffusion.md) | Schedule, U-Net + attention manual, loss, sampling CFG, EMA | Toleransi test `q_sample` t = 0 salah; gradien terblokir init nol | Overfit 1 batch: loss 1,0 → 0,0001 |
| 04 | 22/10/2026 | [Fase 4 — Training & MNIST](interaksi/04_fase4_training_mnist.md) | Training loop (AMP, EMA, resume), `sample.py` | Overhead epoch pertama (cuDNN), jumlah frame GIF | MNIST 20 epoch: loss 0,0167, angka sesuai label |
| 05 | 22–23/10/2026 | [Fase 5 — CIFAR-10 & eksperimen](interaksi/05_fase5_cifar10_eksperimen.md) | Benchmark, E1 100 epoch, E2–E5, S1, S2 | Skrip PowerShell: UTF-16 & risiko berhenti karena stderr | E1 loss 0,0302; perbandingan lr/batch/schedule |
| 06 | 23/10/2026 | [Fase 6 — Evaluasi](interaksi/06_fase6_evaluasi.md) | FID, sampler DDIM, GIF, grid progres | FID DDPM terlalu lama → DDIM (+ bukti setara); VRAM penuh; path absolut di CSV | **FID terbaik 16,43** (E1 ep 100, w = 2) |
| 07 | 23/10/2026 | [Fase 7 — Dokumentasi](interaksi/07_fase7_dokumentasi.md) | Transkrip, README, laporan, slide, demo | Konverter transkrip: label prompt salah & pesan Claude hilang | Dokumentasi lengkap + demo 2,5 detik |
| 08 | 23/10/2026 | [Fase 8 — Finalisasi](interaksi/08_fase8_finalisasi.md) | Uji reproduksi clone bersih, Release `v1.0` | `WinError 206` (MAX_PATH), junction salah, nama kelompok lain di pola grep transkrip | 74 test lolos di clone bersih; checkpoint terverifikasi checksum |

## Transkrip lengkap
| Berkas | Cakupan |
|---|---|
| [`raw/sesi_2026-10-22_s.d_2026-10-23_fase0-6.md`](raw/sesi_2026-10-22_s.d_2026-10-23_fase0-6.md) | Pemilihan topik, perencanaan, Fase 0–6 |
| [`raw/sesi_2026-10-22_s.d_2026-10-23_final.md`](raw/sesi_2026-10-22_s.d_2026-10-23_final.md) | Seluruh sesi termasuk Fase 7–8 (diekspor di akhir proyek; langkah setelah ekspor tercatat di log Fase 8 dan `audits/fase-8.md`) |
