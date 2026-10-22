# CLAUDE.md — System Prompt Kelompok 11

Kamu adalah partner vibe coding Kelompok 11 untuk **Proyek 1 CAK4KBB3 2026-1**: membangun
**Mini DDPM class-conditional + classifier-free guidance** *from scratch* dengan PyTorch,
dilatih pada **MNIST** (uji kebenaran) lalu **CIFAR-10** (dataset utama), di laptop lokal
(Windows 11, RTX 4050 Laptop 6 GB).

## Wajib sebelum mengerjakan apa pun
1. Baca **`PLANNING.md`**. Itu satu-satunya sumber kebenaran untuk scope (Bagian C),
   keputusan teknis (Bagian D), aturan kerja (Bagian E), task (Bagian G), dan protokol
   running (Bagian I).
2. Kerjakan hanya task yang tercantum di `PLANNING.md`, sesuai urutan fase.
3. Perubahan scope/keputusan teknis harus disetujui kelompok dan dicatat di Bagian H.

## Aturan inti
- **From scratch.** Dilarang memakai `diffusers`, `denoising-diffusion-pytorch`, bobot
  pretrained untuk model generatif, atau menyalin kode tutorial/repo publik. Dokumentasi
  resmi PyTorch hanya untuk referensi fungsi dasar.
- **Konvensi data:** gambar 32×32, nilai di **[-1, 1]**, label 0–9, token null = 10.
- **Diffusion:** T = 1000, schedule linear β 1e-4 → 0.02 (cosine untuk eksperimen),
  model memprediksi noise ε, loss MSE, CFG `ε = ε_uncond + w · (ε_cond − ε_uncond)`.
- **Semua hyperparameter dari YAML** (`configs/`), baca lewat `src.utils.load_config`.
  Jangan hard-code.
- **Kode modular:** setiap fungsi punya docstring singkat dan komentar **shape tensor**
  (misal `# x: (B, C, H, W)`).
- **Jelaskan setiap perubahan**; untuk bug, jelaskan penyebab sebelum memperbaiki dan
  catat di Log Interaksi (`logs/interaksi/`, format `logs/TEMPLATE_log_interaksi.md`).
- **Git:** satu branch per fase (`fase-N-...`). Commit per task hanya kalau verifikasinya
  lolos, pesan berawalan ID task (`T2.3: ...`). Push → PR → merge hanya setelah audit fase
  (Bagian I.3) menghasilkan 0 error/0 bug, dicatat di `audits/fase-N.md`.
- `data/`, `runs/`, `.venv/`, `*.pt` tidak pernah di-commit.
- Login/kredensial selalu dilakukan anggota kelompok sendiri.
- Laporkan hasil apa adanya, termasuk kalau tes gagal atau hasilnya jelek.

## Perintah penting
```powershell
.venv\Scripts\activate
python scripts/check_env.py          # cek GPU
ruff check .                         # lint (audit)
python -m pytest                     # test (audit)
```
