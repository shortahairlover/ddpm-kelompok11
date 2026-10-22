# Mini DDPM — Kelompok 11

Implementasi **Denoising Diffusion Probabilistic Model (DDPM)** skala kecil *from scratch* dengan PyTorch,
class-conditional dengan classifier-free guidance, dilatih pada **MNIST** dan **CIFAR-10** (32×32).

Proyek 1 mata kuliah CAK4KBB3 2026-1 — *Vibe Coding & Generative AI from Scratch*, dikerjakan bersama Claude Code.

| Nama | NIM | Peran |
|---|---|---|
| Naufal Rafif Hidayat | 103012300403 | Prompt Engineer & LLM Navigator (Ketua) |
| Ardhian Calwa Nugraha | 103012300404 | Data Pipeline & Preprocessing Engineer |
| Ali Hizqil Syauqani | 103012300036 | Training Loop & Hyperparameter Tuning Engineer |
| Muhammad Akhdan Ghifari | 103012300148 | Lead Documentarian & Error Handler |

## Setup (Windows, PowerShell)

Prasyarat: Python 3.11, Git, GPU NVIDIA dengan driver yang mendukung CUDA 13.0 (opsional; tanpa GPU kode tetap jalan di CPU, tetapi lambat).

```powershell
git clone https://github.com/shortahairlover/ddpm-kelompok11.git
cd ddpm-kelompok11
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt     # termasuk torch 2.14.1 + torchvision 0.29.1 build CUDA 13.0
python scripts/check_env.py         # harus berakhir dengan "SEMUA CEK LOLOS"
```

## Struktur

| Folder | Isi |
|---|---|
| `configs/` | Hyperparameter YAML (`mnist.yaml`, `cifar10.yaml`, `experiments/`) |
| `src/` | Kode inti: data, schedule, model U-Net, diffusion, training, sampling, evaluasi |
| `scripts/` | Skrip bantu (cek lingkungan, visualisasi, plot) |
| `tests/` | Test `pytest` |
| `logs/` | Log Interaksi LLM (`interaksi/`) dan transkrip (`raw/`) |
| `audits/` | Laporan audit tiap fase |
| `reports/` | Gambar, tabel eksperimen, laporan akhir |

## Pengembangan

```powershell
ruff check .          # lint
python -m pytest      # test
```

Rencana kerja dan aturan lengkap ada di [`PLANNING.md`](PLANNING.md); aturan untuk LLM ada di [`CLAUDE.md`](CLAUDE.md).
Cara training, sampling, dan evaluasi akan dilengkapi seiring proyek berjalan.
