# Planning Proyek 1 — Mini Diffusion Model (DDPM)
**Kelompok 11 · CAK4KBB3 2026-1 · Vibe Coding & Generative AI from Scratch**

> **Fungsi dokumen ini:** konteks kerja untuk Claude (Claude Code). Semua task di bawah dikerjakan oleh Claude atas arahan kelompok. Sebelum mengerjakan task apa pun, Claude **wajib membaca dokumen ini**, hanya mengerjakan task yang tercantum, dan tidak keluar dari scope (Bagian C). Perubahan scope atau keputusan teknis harus disetujui kelompok dan dicatat di Bagian H. **Cara menjalankan proyek (tanpa berhenti, branch per fase, audit sebelum lanjut) diatur di Bagian I.**

---

## A. Identitas Proyek

| | |
|---|---|
| Mata kuliah | CAK4KBB3 2026-1 — Proyek 1: Vibe Coding |
| Kelompok | 11 |
| Model | Mini DDPM (Denoising Diffusion Probabilistic Model), class-conditional + classifier-free guidance (CFG) |
| Dataset | MNIST (uji kebenaran kode) → CIFAR-10 (dataset utama) |
| Bahasa / Framework | Python + PyTorch |
| LLM | Claude Code (Opus 5.5) |
| Lingkungan | Lokal, Windows 11 |
| Folder kerja | `D:\Files Mata Kuliah\Semester 7\genai\Kerja Kelompok\ddpm-kelompok11\` |
| Repo GitHub | `ddpm-kelompok11` (private), satu branch per fase → PR → merge ke `main` |

**Anggota** (dicatat untuk keperluan laporan dan Log Interaksi):

| Nama | NIM | Peran formal (sesuai panduan) |
|---|---|---|
| Naufal Rafif Hidayat | 103012300403 | Anggota 1 — Prompt Engineer & LLM Navigator (**Ketua**) |
| Ardhian Calwa Nugraha | 103012300404 | Anggota 2 — Data Pipeline & Preprocessing Engineer |
| Ali Hizqil Syauqani | 103012300036 | Anggota 3 — Training Loop & Hyperparameter Tuning Engineer |
| Muhammad Akhdan Ghifari | 103012300148 | Anggota 4 — Lead Documentarian & Error Handler |

---

## B. Spesifikasi Mesin (hasil pengecekan 22/10/2026)

| Komponen | Kondisi |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop, **6 GB VRAM**, driver 617.42 |
| CPU / RAM | Intel Core i5-13420H / 16 GB |
| Disk D: | ±218 GB kosong |
| Python | 3.11.9 (dipakai untuk proyek, venv `.venv`) dan 3.14 (tidak dipakai) |
| CUDA (driver) | UMD 13.4 → PyTorch 2.14.1 + cu130 |
| Git | 2.55 ✅ |
| VS Code | 1.132 ✅ |
| winget | 1.29 ✅ |
| GitHub CLI (`gh`) | ✅ 2.103.0 (di-install 22/10/2026) |
| Koneksi GitHub | ✅ bisa diakses (22/10/2026, setelah sebelumnya timeout) |
| Identitas Git | ✅ `shortahairlover` / naufalraf21@gmail.com |

Konsekuensi: training dilakukan di GPU lokal ini. Dengan VRAM 6 GB, model kecil + AMP + batch 128 pada 32×32 seharusnya muat. Kalau ternyata tidak, turunkan batch (lihat Bagian D).

---

## C. Scope

**Termasuk scope:**
- Implementasi DDPM **from scratch** dengan PyTorch: noise schedule, forward process, timestep embedding, class embedding, U-Net kecil, loss, sampling DDPM, classifier-free guidance.
- Data pipeline MNIST & CIFAR-10 via torchvision.
- Training, checkpoint/resume, EMA, AMP, logging metrik.
- Eksperimen hyperparameter, evaluasi FID, visualisasi (grid, GIF).
- Log Interaksi LLM, README, laporan akhir, bahan presentasi.
- Opsional (hanya kalau inti sudah selesai): sampler DDIM.

**Di luar scope (jangan dikerjakan):**
- Library diffusion siap pakai (`diffusers`, `denoising-diffusion-pytorch`, dsb.) atau bobot pretrained apa pun untuk model generatif.
- Menyalin kode dari tutorial atau repo publik.
- Resolusi di atas 32×32, latent diffusion, text-to-image.
- Web app / UI / deployment.
- Dataset selain MNIST & CIFAR-10 (kecuali disetujui kelompok).
- Pekerjaan Proyek 2.

---

## D. Keputusan Teknis (tetap, kecuali diubah lewat Bagian H)

| Item | Nilai |
|---|---|
| Ukuran gambar | 32×32 (MNIST di-pad 2 px dari 28×28) |
| Normalisasi | [-1, 1] (`x * 2 - 1` setelah `ToTensor`) |
| Augmentasi | CIFAR-10: random horizontal flip · MNIST: tidak ada |
| Timesteps T | 1000 |
| Noise schedule | Linear β 1e-4 → 0.02 (default); cosine untuk eksperimen |
| Target prediksi | Noise ε, loss = MSE |
| Kondisi kelas | `nn.Embedding(11, dim)`, indeks 10 = token null; label dropout 10% |
| Guidance | `ε = ε_uncond + w · (ε_cond − ε_uncond)` |
| U-Net MNIST | base channel 32, mult (1, 2, 2), attention di 16×16 |
| U-Net CIFAR-10 | base channel 64, mult (1, 2, 2, 2), attention di 16×16, dropout 0.1 |
| Optimizer | AdamW, lr 2e-4, grad clip 1.0 |
| Batch | 128 (turun ke 64 kalau OOM) |
| EMA | 0.999 (MNIST), 0.9999 (CIFAR-10) — sampel selalu dari model EMA |
| Presisi | AMP (mixed precision) di GPU |
| Logging | TensorBoard + `metrics.csv` per run |
| Seed | 42 (default) |
| Config | Semua hyperparameter di file YAML, tidak di-hardcode |

---

## E. Aturan Kerja Claude

1. **Baca dokumen ini** di awal setiap sesi. Kerjakan task sesuai urutan fase dan protokol di Bagian I.
2. **From scratch**: tulis kode berdasarkan konsep, bukan menyalin tutorial. Dokumentasi resmi PyTorch hanya untuk referensi fungsi dasar.
3. **Kode modular & jelas**: setiap fungsi diberi docstring singkat dan komentar **shape tensor** (misal `# x: (B, C, H, W)`).
4. **Satu task selesai** → jalankan verifikasi task → kalau lolos, centang task di dokumen ini dan buat commit lokal di branch fase dengan pesan berawalan ID task (misal `T2.3: tambah loader CIFAR-10`). Kode yang verifikasinya gagal **tidak di-commit**.
5. **Push hanya setelah audit fase lolos** dengan 0 error/bug (Bagian I.3).
6. **Log Interaksi** (Fase 7): setiap task atau bug penting dicatat di `logs/interaksi/` mengikuti template dosen. Prompt kelompok dicatat **apa adanya**. Error dicatat **utuh** beserta analisis dan solusinya.
7. **Debugging**: semua error/bug langsung dianalisis dan diperbaiki oleh Claude tanpa menunggu, lalu dicatat di Log Interaksi (pesan error utuh, penyebab, perbaikan, pelajaran).
8. **Login/kredensial selalu dilakukan kelompok sendiri** (`gh auth login`, akun GitHub). Claude tidak pernah mengetik password atau token.
9. Proses panjang (training, generate FID) dijalankan **di background** dengan checkpoint berkala, lalu dipantau sampai selesai.
10. Folder `data/`, `runs/`, `.venv/`, dan file `*.pt` **tidak di-commit**. Claude tidak menghapus data/checkpoint, kecuali hasil run `--debug` miliknya sendiri.
11. Laporkan hasil apa adanya: kalau tes gagal atau hasil jelek, tulis apa adanya beserta datanya.

---

## F. Struktur Repo Target

```
ddpm-kelompok11/
├── CLAUDE.md              ← aturan ringkas untuk Claude + rujukan ke PLANNING ini
├── PLANNING.md            ← salinan dokumen ini (sumber kebenaran)
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── mnist.yaml
│   ├── cifar10.yaml
│   └── experiments/       ← e2_lr1e-4.yaml, e3_lr4e-4.yaml, ...
├── src/
│   ├── __init__.py
│   ├── data.py            ← dataset, transform, dataloader, denormalize
│   ├── schedule.py        ← beta schedule linear & cosine + konstanta turunan
│   ├── model.py           ← timestep embedding, ResBlock, Attention, U-Net
│   ├── diffusion.py       ← q_sample, loss, p_sample, sampling loop, CFG, (DDIM)
│   ├── ema.py
│   ├── train.py
│   ├── sample.py
│   ├── evaluate.py        ← FID
│   └── utils.py           ← seed, device, config loader, save grid, logger
├── scripts/
│   ├── check_env.py
│   ├── show_batch.py
│   ├── visualize_forward.py
│   ├── plot_metrics.py
│   └── make_gif.py
├── tests/                 ← pytest: shape, schedule, embedding, model, overfit
├── logs/
│   ├── TEMPLATE_log_interaksi.md
│   ├── interaksi/         ← Log Interaksi per task/bug
│   └── raw/               ← ekspor transkrip sesi Claude
├── data/                  ← dataset (diunduh otomatis, tidak di-commit)
├── runs/                  ← checkpoint, sampel, tensorboard (tidak di-commit)
└── reports/
    ├── figures/           ← gambar final (di-commit)
    ├── eksperimen.md
    └── laporan_akhir.md
```

---

## G. Fase & Task

### Fase 0 — Persiapan Tools & Koneksi GitHub
**Tujuan:** semua tools siap, GPU bisa dipakai PyTorch, Claude terhubung ke GitHub, dan repo GitHub sudah ada.
**Branch:** `fase-0-persiapan-tools`

*Koneksi GitHub (harus beres lebih dulu):*
- [x] **T0.1** Cek ulang hardware & tools (`nvidia-smi`, `py -0`, `git --version`, `winget --version`) dan perbarui Bagian B kalau ada yang berubah.
- [x] **T0.2** Pastikan github.com bisa diakses (`Test-NetConnection github.com -Port 443`). Kalau gagal → kondisi berhenti (Bagian I.4); kelompok memperbaiki jaringan (ganti Wi-Fi/hotspot, cek firewall/antivirus/DNS).
- [x] **T0.3** Install GitHub CLI: `winget install --id GitHub.cli -e`.
- [x] **T0.4** Kelompok menjalankan `gh auth login` (GitHub.com → HTTPS → login via browser). Claude memverifikasi dengan `gh auth status` dan menjalankan `gh auth setup-git`.
- [x] **T0.5** Identitas Git (`git config --global user.name` / `user.email`) diisi sesuai data dari kelompok.

*Repo:*
- [x] **T0.6** Buat folder `ddpm-kelompok11/`, `git init -b main`, buat `.gitignore` (`.venv/`, `data/`, `runs/`, `*.pt`, `__pycache__/`, `.ipynb_checkpoints/`, `.pytest_cache/`, `.ruff_cache/`).
- [x] **T0.7** Salin dokumen ini ke `ddpm-kelompok11/PLANNING.md` (mulai titik ini, file itulah yang diperbarui) + `README.md` singkat → commit awal di `main`.
- [x] **T0.8** Buat repo GitHub private: `gh repo create ddpm-kelompok11 --private --source . --remote origin --push`. Verifikasi repo terlihat di `gh repo view`.
- [x] **T0.9** Buat branch `fase-0-persiapan-tools` dari `main`.

*Lingkungan Python:*
- [x] **T0.10** Buat virtual environment Python 3.11 di repo: `py -3.11 -m venv .venv`, lalu upgrade `pip`.
- [x] **T0.11** Install PyTorch + torchvision **versi CUDA** sesuai perintah resmi di pytorch.org (pilih build CUDA yang didukung driver 617.42).
- [x] **T0.12** Install library pendukung: `numpy matplotlib tqdm pyyaml pillow imageio tensorboard "torchmetrics[image]" torch-fidelity pytest ruff`. Simpan `requirements.txt` (versi dipin).
- [x] **T0.13** Buat `scripts/check_env.py`: print versi Python/PyTorch/CUDA, nama GPU, VRAM, dan uji perkalian matriks + AMP di GPU.
- [x] **T0.14** Tambahkan konfigurasi `ruff` minimal (`pyproject.toml`) untuk audit.

**Output:** repo GitHub `ddpm-kelompok11`, `.venv/`, `requirements.txt`, `pyproject.toml`, `scripts/check_env.py`.
**Selesai kalau:** `gh auth status` OK, repo ada di GitHub, `python scripts/check_env.py` menampilkan `cuda available: True` + nama RTX 4050 + uji GPU lolos, lalu audit Fase 0 lolos dan branch sudah di-push & di-merge (Bagian I).

---

### Fase 1 — Inisialisasi Repo & Fondasi
**Tujuan:** kerangka proyek, aturan, dan alat bantu dasar tersedia.
**Branch:** `fase-1-fondasi`

- [x] **T1.1** Buat branch `fase-1-fondasi` dari `main` terbaru.
- [x] **T1.2** Buat seluruh struktur folder sesuai Bagian F (file kosong/placeholder seperlunya).
- [x] **T1.3** Tambahkan folder `audits/` untuk laporan audit tiap fase (Bagian I.3).
- [x] **T1.4** Tulis `CLAUDE.md`: ringkasan proyek, rujukan wajib ke `PLANNING.md`, aturan Bagian C–E dalam bentuk singkat. Ini berfungsi sebagai **system prompt awal** kelompok.
- [x] **T1.5** Tulis `README.md` awal: deskripsi, cara setup, cara menjalankan (diperbarui di akhir).
- [x] **T1.6** Buat `src/utils.py`: `set_seed`, `get_device`, `load_config` (YAML → dict/namespace), `save_image_grid`, `make_run_dir` (`runs/<nama>_<timestamp>/` + salinan config).
- [x] **T1.7** Buat `configs/mnist.yaml` dan `configs/cifar10.yaml` dengan nilai dari Bagian D.
- [x] **T1.8** Buat `logs/TEMPLATE_log_interaksi.md` sesuai template dosen.
- [x] **T1.9** `tests/test_utils.py`: `load_config` membaca kedua YAML, `set_seed` deterministik, `make_run_dir` membuat folder + salinan config.

**Selesai kalau:** struktur repo lengkap, test utils lolos, `CLAUDE.md` merujuk ke `PLANNING.md`.

---

### Fase 2 — Data Pipeline
**Tujuan:** data MNIST & CIFAR-10 masuk ke model dengan shape dan rentang nilai yang benar.
**Branch:** `fase-2-data-pipeline`

- [x] **T2.1** `src/data.py`: transform MNIST (`ToTensor` → `Pad(2)` → [-1, 1]) dan CIFAR-10 (`RandomHorizontalFlip` → `ToTensor` → [-1, 1]).
- [ ] **T2.2** `get_dataset(cfg)` memakai `torchvision.datasets.MNIST` / `CIFAR10` dengan `root="data"` dan `download=True` (unduhan otomatis: MNIST ±12 MB, CIFAR-10 ±170 MB). Kalau gagal, unduh manual `cifar-10-python.tar.gz` dari situs resmi CIFAR ke `data/`.
- [x] **T2.3** `get_dataloader(cfg)`: batch size, shuffle, `num_workers`, `pin_memory`, `drop_last` dari config. Pastikan aman di Windows (guard `if __name__ == "__main__":` di skrip pemanggil).
- [x] **T2.4** Mode debug: subset kecil (misal 512 gambar) lewat flag config/CLI.
- [x] **T2.5** `denormalize()` ([-1, 1] → [0, 1]) untuk menyimpan gambar.
- [ ] **T2.6** `scripts/show_batch.py`: simpan grid 1 batch per dataset ke `reports/figures/`, print shape, dtype, min/max, dan distribusi label.
- [ ] **T2.7** `tests/test_data.py`: assert shape `(B, C, 32, 32)`, C = 1 (MNIST) / 3 (CIFAR), nilai di [-1, 1], label 0–9.

**Output:** `src/data.py`, `scripts/show_batch.py`, `reports/figures/batch_mnist.png`, `reports/figures/batch_cifar10.png`.
**Selesai kalau:** test data lolos dan kedua grid terlihat benar.

---

### Fase 3 — Komponen Diffusion & Model
**Tujuan:** semua komponen DDPM jalan dan terbukti bisa belajar.
**Branch:** `fase-3-model-diffusion`

- [ ] **T3.1** `src/schedule.py`: schedule linear & cosine, hitung `betas`, `alphas`, `alphas_cumprod`, `alphas_cumprod_prev`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`, `posterior_variance`. Fungsi bantu `extract(a, t, shape)`.
- [ ] **T3.2** `src/diffusion.py`: `q_sample(x0, t, noise)` (forward process).
- [ ] **T3.3** `scripts/visualize_forward.py`: satu gambar pada t = 0, 100, 250, 500, 750, 999 → `reports/figures/forward_*.png`.
- [ ] **T3.4** `src/model.py`: sinusoidal timestep embedding + MLP.
- [ ] **T3.5** Class embedding dengan token null (indeks 10), digabung dengan embedding waktu.
- [ ] **T3.6** ResBlock (GroupNorm → SiLU → Conv, injeksi embedding, dropout, skip 1×1 bila channel berubah).
- [ ] **T3.7** Self-attention block (dipakai di resolusi 16×16).
- [ ] **T3.8** U-Net lengkap (down, middle, up, skip connection) yang dikonfigurasi dari YAML. Print jumlah parameter.
- [ ] **T3.9** `diffusion.py`: `training_loss(model, x0, y)`: sampling t acak, noise, label dropout, MSE.
- [ ] **T3.10** `diffusion.py`: `p_sample` + `sample_loop` DDPM dengan CFG (`w`), clamp akhir ke [-1, 1], opsi menyimpan intermediate untuk GIF.
- [ ] **T3.11** `src/ema.py`: kelas EMA (update, copy ke model, state_dict).
- [ ] **T3.12** `tests/`: schedule (monoton, rentang benar), `q_sample` (t = 0 ≈ x0, t = T−1 ≈ noise), embedding (shape), U-Net (output shape = input shape untuk MNIST & CIFAR config).
- [ ] **T3.13** **Tes overfit 1 batch**: 1 batch kecil, ratusan step → loss turun jauh. Simpan grafiknya.

**Selesai kalau:** semua test lolos dan tes overfit menunjukkan loss turun konsisten.

---

### Fase 4 — Training Loop & Sanity Run MNIST
**Tujuan:** pipeline training end-to-end terbukti benar pada dataset mudah.
**Branch:** `fase-4-training-mnist`

- [ ] **T4.1** `src/train.py`: argumen `--config`, `--device`, `--debug`, `--resume`, `--epochs` (override). Loop epoch/step dengan AdamW, AMP (GradScaler), grad clip, update EMA.
- [ ] **T4.2** Logging loss per step & per epoch ke TensorBoard + `metrics.csv`, plus lr dan waktu per epoch.
- [ ] **T4.3** Simpan checkpoint (model, EMA, optimizer, scaler, epoch, step) tiap N epoch + `last.pt`; `--resume` melanjutkan dengan benar.
- [ ] **T4.4** Simpan grid sampel per kelas (10 kelas × N) dari model EMA tiap N epoch.
- [ ] **T4.5** `src/sample.py`: muat checkpoint → generate grid per kelas, grid variasi guidance `w`, dan frame denoising untuk GIF.
- [ ] **T4.6** `scripts/plot_metrics.py`: plot loss dari `metrics.csv` (satu atau beberapa run).
- [ ] **T4.7** Uji `--debug` (beberapa step) termasuk uji `--resume`.
- [ ] **T4.8** **Sanity run MNIST** penuh (±20 epoch) di background.
- [ ] **T4.9** Evaluasi visual: angka terbaca dan sesuai label → simpan `reports/figures/mnist_grid_final.png` + grafik loss MNIST.

**Selesai kalau:** angka 0–9 hasil generate terbaca jelas dan sesuai label yang diminta.

---

### Fase 5 — Training CIFAR-10 & Eksperimen Tuning
**Tujuan:** model utama terlatih dan ada bukti eksperimen tuning.
**Branch:** `fase-5-cifar10-eksperimen`

- [ ] **T5.1** Uji cepat CIFAR-10 `--debug`: cek VRAM dan kecepatan per step. Sesuaikan batch kalau OOM, dan catat di Bagian H.
- [ ] **T5.2** Estimasi lama training dari kecepatan step dan catat di `reports/eksperimen.md`. Kalau estimasi E1 > 8 jam, kurangi epoch E1 secukupnya dan catat di Bagian H.
- [ ] **T5.3** **E1 Baseline** (konfigurasi Bagian D, target 100+ epoch) di background dengan checkpoint berkala.
- [ ] **T5.4** Buat config eksperimen di `configs/experiments/`:

  | ID | Perubahan dari baseline |
  |---|---|
  | E2 | lr 1e-4 |
  | E3 | lr 4e-4 |
  | E4 | batch 64 |
  | E5 | cosine schedule |

  E2–E5 dijalankan lebih pendek (misal 30 epoch) lalu dibandingkan dengan E1 pada epoch yang sama.
- [ ] **T5.5** Jalankan E2–E5 satu per satu (setelah E1 selesai, karena VRAM hanya cukup untuk satu run).
- [ ] **T5.6** **S1** (tanpa training ulang): sampling dengan guidance w = 0, 1, 3, 5 dari checkpoint terbaik.
- [ ] **T5.7** **S2** (tanpa training ulang): bandingkan sampel model EMA vs non-EMA.
- [ ] **T5.8** Isi `reports/eksperimen.md`: config, lama training, loss akhir, FID (dari Fase 6), dan catatan visual tiap eksperimen.

**Selesai kalau:** E1 selesai, E2–E5 + S1 + S2 terdokumentasi, checkpoint terbaik ditentukan.

---

### Fase 6 — Evaluasi & Visualisasi
**Tujuan:** bukti kuantitatif & visual yang valid untuk laporan dan presentasi.
**Branch:** `fase-6-evaluasi`

- [ ] **T6.1** `src/evaluate.py`: FID dengan `torchmetrics` (5.000 sampel generate vs gambar asli CIFAR-10). Catat bahwa FID 5k hanya untuk perbandingan antar eksperimen.
- [ ] **T6.2** Hitung FID untuk E1–E5 (checkpoint pada epoch yang sama) dan untuk guidance w terbaik.
- [ ] **T6.3** Grafik loss: MNIST, CIFAR-10 baseline, dan perbandingan E1–E5 dalam satu plot.
- [ ] **T6.4** Grid per kelas final (10 × 8) untuk MNIST & CIFAR-10.
- [ ] **T6.5** Grid efek guidance scale (w = 0 → 5).
- [ ] **T6.6** `scripts/make_gif.py`: GIF proses denoising (noise → gambar) untuk beberapa kelas.
- [ ] **T6.7** Grid progres kualitas sampel per epoch (bukti model makin baik).
- [ ] **T6.8** *(Opsional)* Sampler DDIM 50–100 langkah + perbandingan waktu & kualitas dengan DDPM.

**Output:** semua gambar di `reports/figures/`, tabel FID di `reports/eksperimen.md`.

---

### Fase 7 — Dokumentasi & Log Interaksi
**Tujuan:** memenuhi rubrik Log Interaksi (25%) dan Vibe Coding (30%). Fase ini **berjalan terus sejak Fase 0**, bukan hanya di akhir.
**Branch:** log per task/bug di-commit di branch fase masing-masing; finalisasi dokumen (T7.4–T7.8) di branch `fase-7-dokumentasi`.

- [ ] **T7.1** Setiap task/modul selesai → buat `logs/interaksi/NN_<modul>.md` sesuai template: tanggal, target modul, rujukan transkrip, Tahap Inisiasi (prompt kelompok apa adanya, alasan, ringkasan output/cuplikan kode penting).
- [ ] **T7.2** Setiap bug penting → bagian Tahap Debugging: pesan error utuh, prompt perbaikan dari kelompok, analisis & solusi, pelajaran.
- [ ] **T7.3** Ekspor transkrip sesi Claude ke `logs/raw/` di akhir setiap sesi kerja.
- [ ] **T7.4** `logs/README.md`: indeks kronologis semua log (inisiasi → error → optimasi).
- [ ] **T7.5** Perbarui `README.md` final: setup, cara training, sampling, evaluasi, contoh hasil.
- [ ] **T7.6** `reports/laporan_akhir.md` (lalu ekspor ke PDF), berisi:
  1. Pendahuluan & tujuan
  2. Teori singkat DDPM (forward/reverse, prediksi noise, CFG)
  3. Metodologi vibe coding & isi `CLAUDE.md`
  4. Data pipeline
  5. Arsitektur model (diagram U-Net + jumlah parameter)
  6. Training & eksperimen (tabel + grafik)
  7. Hasil (grid, GIF, FID)
  8. Analisis error & pelajaran dari debugging
  9. Pembagian peran & riwayat integrasi (commit)
  10. Keterbatasan & pengembangan lanjut
  11. Lampiran: daftar Log Interaksi + transkrip
- [ ] **T7.7** Bahan presentasi: slide (alur, arsitektur, hasil, demo) + daftar pertanyaan latihan beserta jawabannya, misalnya:
  - Kenapa normalisasi [-1, 1]?
  - Kenapa model memprediksi noise, bukan gambar?
  - Apa fungsi timestep embedding?
  - Bagaimana CFG bekerja?
  - Kenapa memakai EMA?
  - Error tersulit apa, dan bagaimana menyelesaikannya?
- [ ] **T7.8** Skrip demo: generate gambar per kelas langsung dari checkpoint terbaik (cepat dijalankan saat presentasi).

---

### Fase 8 — Finalisasi & Penyerahan
**Tujuan:** proyek bisa dijalankan ulang dan siap dikumpulkan.
**Branch:** `fase-8-finalisasi`

- [ ] **T8.1** Uji reproduksi: clone bersih → setup venv dari `requirements.txt` → jalankan `pytest` + `--debug` training + sampling dari checkpoint.
- [ ] **T8.2** Bersihkan repo: tidak ada file besar/rahasia, `.gitignore` benar, kode tanpa sisa debug.
- [ ] **T8.3** Pastikan semua task di dokumen ini tercentang atau diberi alasan kalau dilewati.
- [ ] **T8.4** Audit akhir menyeluruh (Bagian I.3 untuk seluruh repo) → push → PR → merge ke `main`. Pastikan `main` berisi semua fase.
- [ ] **T8.5** Siapkan checkpoint terbaik untuk dibagikan terpisah (Google Drive), karena file `.pt` tidak di-commit.
- [ ] **T8.6** Pengingat untuk kelompok (dikerjakan kelompok sendiri):
  - isi kolom Project 1 di Daftar Kelompok (Algoritma: *Mini DDPM class-conditional, MNIST & CIFAR-10*);
  - perbaiki NIM Akhdan menjadi 103012300148;
  - kumpulkan laporan sesuai instruksi dosen.

---

## Checklist Deliverable Akhir
- [ ] Repo GitHub rapi dengan riwayat commit per task
- [ ] `CLAUDE.md` (system prompt awal) + `PLANNING.md`
- [ ] Kode modular + test + README
- [ ] Grafik loss MNIST & CIFAR-10, tabel eksperimen, FID
- [ ] Grid per kelas, grid guidance, grid progres, GIF denoising
- [ ] Log Interaksi lengkap + transkrip
- [ ] Laporan akhir (PDF)
- [ ] Slide presentasi + skrip demo

---

## H. Catatan Perubahan & Keputusan

| Tanggal | Perubahan / Keputusan | Alasan |
|---|---|---|
| 22/10/2026 | Topik: Mini DDPM class-conditional, MNIST → CIFAR-10 | Tidak bentrok dengan kelompok lain (U-Net, Mini GPT, CGAN Fashion-MNIST) |
| 22/10/2026 | Ketua: Naufal Rafif Hidayat; NIM Akhdan 103012300148 | Arahan kelompok |
| 22/10/2026 | Semua task dikerjakan Claude; planning tanpa jadwal & PIC per fase | Arahan kelompok |
| 22/10/2026 | Training lokal di RTX 4050 Laptop 6 GB, Python 3.11 | Hasil pengecekan mesin |
| 22/10/2026 | Repo GitHub dibuat di Fase 0; satu branch per fase; push hanya setelah audit 0 error/bug | Arahan kelompok |
| 22/10/2026 | Proyek dijalankan tanpa berhenti dari Fase 0 sampai Fase 8 (Bagian I); debugging langsung oleh Claude | Arahan kelompok |
| 22/10/2026 | GitHub sempat tidak bisa diakses (timeout 443), lalu bisa diakses saat Fase 0 dimulai | Hasil pengecekan |
| 22/10/2026 | PyTorch 2.14.1 + torchvision 0.29.1 build CUDA 13.0 | Versi stabil terbaru; driver mendukung CUDA 13.4 |

---

## I. Protokol Running Proyek

### I.1 Prinsip
- Setelah kelompok memberi perintah **"mulai"**, Claude menjalankan seluruh task **dari Fase 0 sampai Fase 8 tanpa berhenti**: tidak menunggu konfirmasi di antara task maupun antar fase.
- Satu-satunya alasan berhenti adalah **kondisi berhenti** di I.4.
- Status kemajuan selalu tercermin di checkbox dokumen ini dan di folder `audits/`, sehingga pekerjaan bisa dilanjutkan dari titik terakhir kalau sesi terputus.

### I.2 Siklus setiap fase
1. `git checkout main && git pull` → buat branch fase (`git checkout -b fase-N-...`).
2. Kerjakan task fase berurutan. Setiap task: implementasi → verifikasi task → commit lokal (pesan `TN.x: ...`) → catat Log Interaksi.
3. **Audit fase** (I.3). Kalau ada error/bug → perbaiki → commit perbaikan (`TN.audit: perbaiki ...`) → **ulangi audit dari awal** sampai hasilnya **0 error / 0 bug**.
4. Tulis `audits/fase-N.md` dan commit.
5. `git push -u origin fase-N-...` → `gh pr create --base main` (judul: "Fase N — ...", isi: ringkasan task + hasil audit) → `gh pr merge --merge` (branch tidak dihapus, supaya riwayat per fase tetap terlihat).
6. Pastikan push & merge berhasil (`gh pr view` status MERGED). **Baru setelah itu** lanjut ke fase berikutnya.

### I.3 Checklist audit fase (semua harus lolos)
| # | Pemeriksaan | Perintah / cara | Syarat lolos |
|---|---|---|---|
| 1 | Kelengkapan task | Cek checkbox fase di `PLANNING.md` | Semua tercentang (atau dilewati dengan alasan tertulis, khusus task opsional) |
| 2 | Lint | `ruff check .` | 0 error |
| 3 | Test (regresi seluruh repo) | `python -m pytest -q` | Semua lolos, 0 gagal |
| 4 | Kriteria "Selesai kalau" fase | Jalankan perintah verifikasi fase | Terpenuhi, dengan bukti (output/gambar) |
| 5 | Smoke test end-to-end (mulai Fase 4) | `python -m src.train --config configs/mnist.yaml --debug` lalu `python -m src.sample` dari checkpoint debug | Jalan tanpa error |
| 6 | Kepatuhan scope | Cari import terlarang (`diffusers`, dsb.), hyperparameter hard-code, kode di luar Bagian C | Tidak ada pelanggaran |
| 7 | Review kode fase | Baca ulang diff fase (`git diff main...HEAD`): logika, shape, edge case, komentar shape | Tidak ada bug yang ditemukan |
| 8 | Kebersihan repo | `git status`, cek file ter-track di `data/`, `runs/`, `*.pt`, file > 10 MB | Bersih, tidak ada file terlarang |
| 9 | Dokumentasi | Log Interaksi fase ada; `PLANNING.md` diperbarui | Lengkap |

Format `audits/fase-N.md`: tanggal, branch, hasil tiap pemeriksaan (perintah + ringkasan output), daftar temuan, perbaikan yang dilakukan, jumlah putaran audit, dan **status akhir: LOLOS (0 error, 0 bug)**.

### I.4 Kondisi berhenti (hanya ini)
Claude berhenti dan melapor ke kelompok **hanya** kalau:
1. Butuh tindakan yang hanya bisa dilakukan kelompok: login GitHub (`gh auth login`), data identitas Git, atau persetujuan UAC/administrator Windows.
2. GitHub tidak bisa diakses sehingga push/merge gagal. Pekerjaan lokal fase itu boleh diselesaikan dulu, tapi tidak boleh lanjut ke fase berikutnya.
3. Masalah hardware/sistem: GPU tidak terdeteksi, disk penuh, driver crash.
4. Bug yang tidak bisa diselesaikan setelah 3 pendekatan perbaikan berbeda tanpa mengubah Bagian C/D. Claude melapor dengan analisis dan usulan perubahan keputusan.

Setelah kondisi teratasi, Claude melanjutkan dari task terakhir yang belum tercentang.

### I.5 Prasyarat sebelum perintah "mulai" (dikerjakan kelompok)
- [ ] Laptop bisa mengakses github.com (cek di browser).
- [ ] Siap login GitHub saat T0.4 (akun yang akan memiliki repo).
- [ ] Nama & email untuk identitas Git sudah diberikan ke Claude.
- [ ] Laptop dicolok ke listrik, mode sleep/hibernate dimatikan selama proyek berjalan (Settings → System → Power).
- [ ] Mode izin Claude Code memperbolehkan menjalankan perintah tanpa konfirmasi satu per satu (supaya benar-benar tanpa berhenti).
