# LOG INTERAKSI LLM — KELOMPOK 11

- **Tanggal Sesi:** 23/10/2026
- **Target Modul:** Fase 7 — Dokumentasi (transkrip, indeks log, README, laporan akhir + PDF, slide, latihan tanya-jawab, skrip demo)
- **LLM:** Claude Code (Opus 5.5)
- **Tautan Sesi Chat:** `logs/raw/` (lihat `logs/README.md`)

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Meminta Kode Awal)
- **Prompt Kelompok:**
  > "buatlah planning nya untuk mengerjakan project 1 Mini Diffusion Model (DDPM) ini menggunakan cifar-10 dan mnist, ini mengerjakan di lokal jadi buat planning dari awal (install software yang dibutuhin dll) sampai akhir selesai projectnya"

  > "anda harus run task proyek ini dari awal hingga akhir itu tanpa berhenti."
- **Alasan Khusus:** Rubrik memberi bobot 25 % pada Log Interaksi dan menilai penguasaan materi saat presentasi, sehingga dokumentasi harus bisa dibaca dosen (bukan JSON mentah), angka di laporan/slide harus sama dengan hasil evaluasi, dan setiap anggota butuh bahan latihan untuk bagiannya.
- **Output Model:**
  - `scripts/transcript_to_md.py`: ekspor sesi Claude Code (JSONL) → Markdown berisi prompt kelompok apa adanya, jawaban Claude, ringkasan tool & output (dipotong); blok *thinking*, gambar, dan metadata aplikasi dibuang; **output yang membaca file Daftar Kelompok disensor** (berisi nama/NIM mahasiswa kelompok lain).
  - `logs/README.md`: indeks kronologis inisiasi → eror → optimasi per fase.
  - `README.md` final: hasil utama, setup, tabel perintah, struktur.
  - `reports/laporan_akhir.md` + `scripts/build_report_pdf.py` (Markdown → HTML → PDF lewat Edge headless) → `reports/laporan_akhir.pdf` (8 halaman).
  - `scripts/build_slides.py` → `reports/slides_presentasi.pptx` (14 slide, chart PowerPoint asli, angka FID dibaca langsung dari `reports/fid_results.csv`, catatan pembicara).
  - `reports/latihan_tanya_jawab.md`: 23 pertanyaan + jawaban per peran.
  - `scripts/demo.py`: generate kelas pilihan audiens dari checkpoint terbaik (12 gambar ±2,5 detik).

### 2. TAHAP PENANGANAN EROR / DEBUGGING

**Bug 1 — Konverter transkrip melabeli pesan sistem sebagai prompt kelompok**
- **Gejala:** daftar "Kelompok" berisi `<task-notification>`, `Your response above was cut off mid-stream...`, dan `[Image: original 2400x585 ...]` — bukan ketikan kelompok.
- **Analisis:** semua entri bertipe `user` dianggap prompt. Di transkrip, sumber pesan tercatat di `origin.kind` (`human` / `task-notification`) dan pesan otomatis ditandai `isMeta`. Pesan yang dikirim kelompok saat Claude bekerja tersimpan sebagai `attachment` bertipe `queued_command`.
- **Solusi:** label "Kelompok" hanya untuk `origin.kind == "human"` (+ `queued_command` manusia), notifikasi ditampilkan sebagai "notifikasi sistem", `isMeta` dibuang. Hasil: 12 prompt kelompok yang benar.

**Bug 2 — Seluruh pesan Claude hilang setelah perbaikan Bug 1**
- **Gejala:** ukuran transkrip turun dari 206 KB ke 89 KB.
- **Analisis:** cabang baru `if kind != "user" or entry.get("isMeta"): pass` ditempatkan sebelum cabang `assistant`, sehingga semua entri `assistant` ikut "ditelan".
- **Solusi:** cabang `assistant` dipindah ke paling atas, cabang lama yang menjadi kode mati dihapus, dan ditambah `tests/test_transcript.py` (memastikan pesan Claude ada, *thinking* tidak ada, data kelompok lain tersensor, pesan meta dibuang). Hasil: 201 KB, 12 prompt, 73 pesan Claude, 248 pemanggilan tool, 0 nama/NIM kelompok lain.
- **Pelajaran:** perubahan kecil pada urutan `if/elif` bisa diam-diam membuang data; ukuran output adalah sinyal cepat yang patut dicek.

**Eror 3 — PDF: daftar bernomor menyatu dengan paragraf**
- **Gejala:** di PDF, "Tujuan: 1. ... 2. ..." tercetak sebagai satu paragraf; baris identitas di kepala laporan menyatu.
- **Analisis:** Python-Markdown (berbeda dengan GitHub) mewajibkan baris kosong sebelum daftar.
- **Solusi:** baris kosong ditambahkan sebelum daftar + line break eksplisit di kepala laporan; diverifikasi dengan screenshot HTML via Edge headless (poppler tidak tersedia untuk merender PDF).

**Temuan 4 — QA visual slide (render lewat PowerPoint COM)**
- Putaran 1 menemukan: angka "100 epoch" terbungkus menimpa label (slide 2), gambar guidance terlalu kecil (slide 5), judul 2 baris menabrak gambar (slide 6), gambar EMA terlalu kecil (slide 10), kartu terlalu tinggi (slide 11), baris perintah terbungkus (slide 13). Semua diperbaiki dan dirender ulang.
- Patch pertama gagal karena karakter `±` rusak saat skrip dikirim lewat stdin (encoding Windows) — perbaikan dilakukan dengan tool edit file langsung.

**Temuan 5 — Klaim angka yang belum benar**
- Draf laporan/slide menulis "75+ test" dan "±3 detik untuk 18 gambar", padahal saat itu 74 test dan demo terukur 12 gambar ±2,5 detik. Diganti dengan angka terukur / pernyataan yang benar; jumlah test final dicek ulang di Fase 8.
