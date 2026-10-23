# Audit Fase 8 — Finalisasi & Penyerahan (audit akhir seluruh repo)

- **Tanggal:** 23/10/2026
- **Branch:** `fase-8-finalisasi`
- **Cakupan:** seluruh repo (bukan hanya diff fase), sesuai T8.4
- **Jumlah putaran audit:** 1
- **Status akhir:** ✅ **LOLOS (0 error, 0 bug)**

## Hasil pemeriksaan

| # | Pemeriksaan | Perintah | Hasil |
|---|---|---|---|
| 1 | Kelengkapan task | checkbox `PLANNING.md` | 82/82 task (T8.4 dicentang bersama laporan ini) + checklist deliverable & prasyarat I.5 |
| 2 | Lint | `ruff check .` | `All checks passed!` |
| 3 | Test (regresi) | `python -m pytest` | `75 passed` |
| 4 | Kriteria selesai | uji reproduksi dari clone GitHub bersih (venv baru, `pip install -r requirements.txt`) | `check_env` SEMUA CEK LOLOS, `74 passed` (sebelum test sensor ditambahkan), train `--debug` + `--resume` + sample + demo + evaluate + plot OK, `git status` bersih |
| 5 | Smoke test end-to-end | train `--debug` MNIST & CIFAR-10 → `src.sample` → `src.evaluate` → `scripts.demo` (checkpoint terbaik) | Semua berjalan |
| 6 | Kepatuhan scope | `git grep` import terlarang | Tidak ada |
| 7 | Review kode | `git diff main...HEAD` (`transcript_to_md.py --redact-file`, `demo.py`, `test_data.py`, README) | Tidak ada bug |
| 8 | Kebersihan repo | file terlarang, file > 5 MB, token `gho_/ghp_`, nama/NIM mahasiswa kelompok lain (daftar di luar repo, kontrol positif 6 file) | Semua 0 |
| 9 | Dokumentasi | tautan relatif di README, indeks log, laporan, eksperimen, CLAUDE.md, tanya-jawab; log 00–08; audit 0–8 | 0 tautan rusak; lengkap |

## Rilis
- GitHub Release **`v1.0`**: `cifar10_e1_epoch100.pt`, `mnist_epoch020.pt`, `SHA256SUMS.txt` — diunduh ulang, `sha256sum -c` → OK, demo dari berkas unduhan berhasil.

## Temuan & perbaikan
| Tahap | Temuan | Perbaikan |
|---|---|---|
| Uji reproduksi | `WinError 206` (MAX_PATH) saat `pip install` di path dalam | Catatan di README; uji ulang di path pendek |
| Uji reproduksi | Junction `data` salah target (escaping Git Bash) → 9 test gagal | Junction dibuat dengan PowerShell → lolos; dihapus dengan aman tanpa menyentuh dataset asli |
| Pembersihan | `noqa: S606` tidak perlu, lambda `noqa: E731` di test | Dihapus / diganti `def` |
| Transkrip final | 3 nama kelompok lain di pola `grep` | `--redact-file` (daftar di luar repo) + test; 0 kebocoran |
| Laporan | Jumlah test berubah 74 → 75 | Angka & PDF diperbarui |
| Audit | — | — |
