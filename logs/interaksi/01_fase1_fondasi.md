# LOG INTERAKSI LLM — KELOMPOK 11

- **Tanggal Sesi:** 22/10/2026
- **Target Modul:** Fase 1 — Fondasi Repo (struktur, `CLAUDE.md`, `src/utils.py`, config YAML, template log)
- **LLM:** Claude Code (Opus 5.5)
- **Tautan Sesi Chat:** transkrip sesi diekspor ke `logs/raw/`

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Meminta Kode Awal)
- **Prompt Kelompok:**
  > "planning ini berfungsi sebagai konteks anda untuk mengerjakan task task agar tidak keluar dari scope. semua task harus di kerjakan dari anda"

  > "anda harus run task proyek ini dari awal hingga akhir itu tanpa berhenti."
- **Alasan Khusus:** Kelompok memakai `PLANNING.md` sebagai konteks tetap dan `CLAUDE.md` sebagai *system prompt* yang otomatis dibaca Claude Code di setiap sesi, supaya logika kode konsisten dari fase ke fase (tugas Prompt Engineer). Semua hyperparameter dipusatkan di YAML agar eksperimen Fase 5 cukup mengganti config, bukan kode.
- **Output Model:**
  - `CLAUDE.md`: ringkasan proyek, kewajiban membaca `PLANNING.md`, aturan *from scratch*, konvensi data/diffusion, aturan Git & audit.
  - `configs/mnist.yaml` & `configs/cifar10.yaml`: bagian `data`, `diffusion`, `model`, `train`, `sample`, `eval` sesuai Bagian D `PLANNING.md` (+ `warmup_steps`, `norm_groups`).
  - `src/utils.py`: `Config` (dict dengan akses atribut), `load_config` (dukung `base:` untuk config eksperimen dan override `kunci=nilai`), `save_config`, `set_seed`, `get_device`, `make_run_dir`, `save_image_grid`, `get_logger`.
  - `tests/test_utils.py`: 11 test (kunci config, nilai sesuai planning, override, pewarisan `base`, seed, device, run dir, grid).
  - `logs/TEMPLATE_log_interaksi.md` sesuai template dosen.

  Cuplikan krusial — pewarisan config untuk eksperimen:
  ```python
  base = data.pop("base", None)
  if base is not None:
      base_path = Path(base)
      if not base_path.is_absolute() and not base_path.exists():
          base_path = path.parent / base
      data = _deep_merge(load_config(base_path).to_dict(), data)
  ```

### 2. TAHAP PENANGANAN EROR / DEBUGGING

**Eror 1 — Perintah shell gagal diparse**
- **Pesan Eror:**
  ```
  /usr/bin/bash: -c: line 216: unexpected EOF while looking for matching `''
  ```
- **Analisis & Solusi:** beberapa file YAML/Python ditulis lewat satu perintah bash panjang berisi banyak heredoc dan tanda kutip; shell gagal mencocokkan tanda kutip sehingga **tidak ada file yang tertulis**. Solusi: setiap file ditulis terpisah dengan tool penulisan file, bukan lewat heredoc.
- **Pelajaran:** untuk file kode panjang, hindari heredoc bersarang di satu perintah shell.

**Bug 2 — Override `1e-4` terbaca sebagai string** (ditemukan saat review sebelum test)
- **Gejala:** `yaml.safe_load("1e-4")` menghasilkan `'1e-4'` (string), karena PyYAML (YAML 1.1) hanya mengenali notasi ilmiah yang punya titik desimal (`1.0e-4`). Override `train.lr=1e-4` akan membuat optimizer menerima string.
- **Solusi:** fungsi `_parse_value` mencoba `float()` bila hasil YAML berupa string. Dikunci dengan `test_overrides_parse_types` (memastikan `train.lr` bertipe `float`), dan semua nilai di file YAML ditulis dengan titik desimal (`2.0e-4`).
- **Pelajaran:** tipe data dari parser config harus diuji, bukan diasumsikan.

### 3. TAHAP OPTIMASI
- `make_run_dir` menambahkan sufiks `-1`, `-2`, ... kalau dua run dibuat pada detik yang sama, sehingga hasil run tidak saling menimpa (diuji di `test_make_run_dir_creates_structure`).
