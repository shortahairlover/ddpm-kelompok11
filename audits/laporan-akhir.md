# Audit — Laporan Akhir sesuai Panduan (28/10/2026)

- **Branch:** `laporan-akhir-panduan`
- **Status akhir:** ✅ **LOLOS (0 error, 0 bug)**

| # | Pemeriksaan | Hasil |
|---|---|---|
| 1 | Kesesuaian dengan Panduan (Word) | Identitas + job desk 4 peran (bagian 2), alasan pemilihan model & tantangan DDPM — noise schedule dan timestep embedding (bagian 3), pemetaan ke 4 kriteria rubrik (bagian 4), 9 Log Interaksi dengan format template bagian 5, dokumentasi chat LLM (transkrip + video) |
| 2 | Lint & test | `ruff check .` bersih; `python -m pytest` → `75 passed` |
| 3 | Gambar | 13 gambar dirujuk, 0 hilang |
| 4 | Tautan | 4 hyperlink aktif di PDF: repositori, video Google Drive, transkrip, Release v1.0 |
| 5 | Privasi | 0 nama / 0 NIM mahasiswa kelompok lain di Markdown & PDF |
| 6 | Visual QA | Render 20 halaman (PyMuPDF, di luar repo): sampul, tabel, kotak log, grafik; perbaikan: daftar Markdown menyatu & judul terpisah dari isi (orphan heading) |
