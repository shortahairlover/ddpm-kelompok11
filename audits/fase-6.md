# Audit Fase 6 — Evaluasi & Visualisasi

- **Tanggal:** 23/10/2026
- **Branch:** `fase-6-evaluasi`
- **Jumlah putaran audit:** 1
- **Status akhir:** ✅ **LOLOS (0 error, 0 bug)**

## Hasil pemeriksaan

| # | Pemeriksaan | Perintah | Hasil |
|---|---|---|---|
| 1 | Kelengkapan task | checkbox T6.1–T6.8 | 8/8 tercentang (T6.8 DDIM dikerjakan, tidak dilewati) |
| 2 | Lint | `ruff check .` | `All checks passed!` |
| 3 | Test (regresi) | `python -m pytest` | `73 passed` (termasuk DDIM ≡ DDPM saat eta = 1, utilitas FID) |
| 4 | Kriteria selesai | `reports/fid_results.csv`, `reports/eksperimen.md`, `reports/figures/` | 12 nilai FID; model terbaik E1 ep 100 w = 2 (FID 16,43); semua gambar/GIF/grafik ada |
| 5 | Smoke test end-to-end | train `--debug` → `src.sample` → `src.evaluate` (20 sampel, DDIM-10) | Ketiganya berjalan tanpa error |
| 6 | Kepatuhan scope | cari `diffusers`, `denoising_diffusion`, `from_pretrained`, `torch.hub` | Tidak ada. Inception (pretrained) hanya dipakai sebagai alat ukur FID, bukan model generatif |
| 7 | Review kode | `git diff main...HEAD`: rumus DDIM (σ, koreksi eps setelah clamp, langkah terakhir a_prev = 1), FID (`normalize=True` → input [0, 1], `reset_real_features=False`, label seimbang), `plot_fid` (filter kolom), `make_gif` | Tidak ada bug |
| 8 | Kebersihan repo | `git ls-files` untuk `data/`, `runs/`, `*.pt`, `*.log`; cari path absolut | Bersih. Path lokal hanya di `PLANNING.md` (keterangan folder kerja) dan kutipan singkat di log — disengaja. File terbesar GIF 3,1 MB |
| 9 | Dokumentasi | `logs/interaksi/06_fase6_evaluasi.md`, `reports/eksperimen.md`, `PLANNING.md` Bagian H | Lengkap |

## Temuan & perbaikan
| Tahap | Temuan | Perbaikan |
|---|---|---|
| Implementasi | FID DDPM-1000 ±20 jam untuk semua evaluasi | Sampler DDIM-100 + bukti kesetaraan (test) + pembanding DDPM vs DDIM |
| Implementasi | VRAM ±5,9 GB saat FID | GIF dibuat di CPU selama FID berjalan |
| Implementasi | Baris kode mati dengan `noqa` di `make_gif.py` | Dihapus |
| Implementasi | Path absolut di CSV FID (dan sempat ter-commit lokal) | `display_path` + test; CSV dikeluarkan dari commit yang belum di-push, dinormalkan sebelum di-commit |
| Implementasi | Label angka grafik guidance menempel judul | Batas sumbu-y dilonggarkan |
| Audit | — | — |
