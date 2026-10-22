# LOG INTERAKSI LLM — KELOMPOK 11

- **Tanggal Sesi:** 22/10/2026
- **Target Modul:** Fase 2 — Data Pipeline MNIST & CIFAR-10 (T2.1–T2.7)
- **LLM:** Claude Code (Opus 5.5)
- **Tautan Sesi Chat:** transkrip sesi diekspor ke `logs/raw/`

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Meminta Kode Awal)
- **Prompt Kelompok:**
  > "untuk Mini Diffusion Model (DDPM) training nya pakai data apa?"

  > "buatlah planning nya untuk mengerjakan project 1 Mini Diffusion Model (DDPM) ini menggunakan cifar-10 dan mnist, ini mengerjakan di lokal"

  > "dan untuk dataset nya apakah kamu bisa download atau harus kita?"
- **Alasan Khusus:** MNIST dipakai sebagai uji kebenaran kode karena cepat konvergen; CIFAR-10 sebagai dataset utama karena berwarna dan berlabel (bisa class-conditional). Keduanya diunduh otomatis lewat torchvision (`download=True`) supaya tidak ada langkah manual dan pipeline bisa direproduksi di laptop mana pun. Ukuran disamakan 32×32 agar satu arsitektur U-Net dipakai untuk kedua dataset.
- **Output Model:**
  - `src/data.py`:
    - `build_transform`: MNIST `Pad(2)` (28→32), CIFAR-10 `RandomHorizontalFlip` (hanya saat train), lalu `ToTensor` + `Normalize(0.5, 0.5)` → nilai **[-1, 1]**.
    - `get_dataset`: MNIST/CIFAR10 torchvision, folder `data/` relatif ke root repo, mode debug = subset acak tetap (seed config).
    - `get_dataloader`: batch size & `num_workers` dari config, `pin_memory` saat ada GPU, `drop_last` saat train, `persistent_workers`.
    - `denormalize`: [-1, 1] → [0, 1].
  - `scripts/show_batch.py`: grid 8×8 + statistik batch (shape, dtype, min/max, distribusi label).
  - `tests/test_data.py`: shape & rentang batch, ukuran dataset penuh, subset debug deterministik, padding MNIST hitam, flip hanya untuk CIFAR-10 train, transform hitam/putih → -1/1, `denormalize`, DataLoader dengan 2 worker.

  Cuplikan krusial:
  ```python
  steps.append(transforms.ToTensor())  # (C, H, W) di [0, 1]
  steps.append(transforms.Normalize(mean=[0.5] * channels, std=[0.5] * channels))  # -> [-1, 1]
  ```
  Hasil `show_batch` MNIST: `shape=(128, 1, 32, 32) dtype=torch.float32`, `min=-1.000 max=1.000`, label 0–9 tersebar.

### 2. TAHAP PENANGANAN EROR / DEBUGGING

**Keputusan desain untuk mencegah eror Windows — `Normalize` bukan `Lambda`**
- **Masalah yang diantisipasi:** di Windows, worker DataLoader dibuat dengan metode *spawn* sehingga dataset beserta transform-nya harus di-pickle. `transforms.Lambda(lambda x: x * 2 - 1)` tidak bisa di-pickle (`Can't pickle <function <lambda>>`).
- **Solusi:** pakai `Normalize(mean=0.5, std=0.5)` yang setara `x * 2 - 1` dan bisa di-pickle. Dikunci dengan `test_dataloader_with_workers_on_windows` (2 worker).

**Eror 1 — Lint pada test**
- **Pesan Eror:**
  ```
  E501 Line too long (103 > 100)
    --> tests\test_data.py:33:101
  ```
- **Solusi:** argumen `parametrize` dipecah ke beberapa baris. `ruff check .` → `All checks passed!`.

**Kendala 2 — Unduhan CIFAR-10 sangat lambat**
- **Gejala:** file `data/cifar-10-python.tar.gz` (±163 MB) bertambah hanya ±30–65 KB/s dari server resmi (cs.toronto.edu), sementara MNIST selesai dalam hitungan detik. Unduhan akhirnya selesai dan lolos verifikasi checksum torchvision (50.000 train, 10.000 test).
- **Analisis & Solusi:** bukan eror kode (ukuran file terus bertambah), melainkan bandwidth server. Sumber dataset tidak diganti agar tetap memakai sumber resmi torchvision; unduhan dibiarkan berjalan di background sambil menyelesaikan bagian MNIST. Unduhan hanya terjadi sekali karena `data/` dipakai ulang.

### 3. HASIL VERIFIKASI
- `python -m scripts.show_batch` → MNIST `(128, 1, 32, 32)`, CIFAR-10 `(128, 3, 32, 32)`, keduanya `min=-1.000 max=1.000`, label tersebar di 10 kelas.
- Grid: `reports/figures/batch_mnist.png`, `reports/figures/batch_cifar10.png`.
- `python -m pytest` → `26 passed`.
