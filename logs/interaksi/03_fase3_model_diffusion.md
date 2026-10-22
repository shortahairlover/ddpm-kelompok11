# LOG INTERAKSI LLM — KELOMPOK 11

- **Tanggal Sesi:** 22/10/2026
- **Target Modul:** Fase 3 — Komponen Diffusion & Model (noise schedule, forward process, timestep/class embedding, U-Net, loss, sampling + CFG, EMA, tes overfit)
- **LLM:** Claude Code (Opus 5.5)
- **Tautan Sesi Chat:** transkrip sesi diekspor ke `logs/raw/`

## KRONOLOGI PROMPT & RESPONS

### 1. TAHAP INISIASI (Meminta Kode Awal)
- **Prompt Kelompok:**
  > "kira kira untuk pemilihan untuk project 1 nya enaknya gimana"

  > "buatlah planning nya untuk mengerjakan project 1 Mini Diffusion Model (DDPM) ini menggunakan cifar-10 dan mnist"

  > "mulai"
- **Alasan Khusus:** Arsitektur dipecah menjadi modul kecil yang masing-masing bisa diuji terpisah (schedule → forward → embedding → U-Net → loss → sampling), sesuai rencana Prompt Engineer di `PLANNING.md` Fase 3. Keputusan teknis dikunci di `CLAUDE.md`: prediksi noise ε, T = 1000, β linear 1e-4 → 0.02, label null = 10 untuk classifier-free guidance.
- **Output Model:**
  - `src/schedule.py`: schedule linear & cosine; konstanta (ᾱ_t, √ᾱ_t, √(1−ᾱ_t), koefisien posterior, varians posterior) dihitung di **float64** lalu disimpan float32; `extract(a, t, shape)` → `(B, 1, 1, 1)`.
  - `src/model.py`: embedding sinusoidal + MLP, embedding kelas `nn.Embedding(11, 4·base)` dijumlah dengan embedding waktu, `ResBlock` (GroupNorm-SiLU-Conv, injeksi embedding, `conv2` di-init nol), `AttentionBlock` multi-head ditulis manual dengan `einsum` + softmax, U-Net encoder–bottleneck–decoder dengan skip connection. Parameter: **MNIST 1,62 juta**, **CIFAR-10 8,95 juta**.
  - `src/diffusion.py`: `q_sample`, `training_loss` (t acak, label drop 10%, MSE), `predict_eps` dengan CFG (cond & uncond dalam satu forward 2B), `predict_x0`, `p_sample` (x̂₀ di-clamp ke [-1, 1], mean posterior, tanpa noise di t = 0), `sample` (opsi simpan frame untuk GIF, generator untuk reprodusibilitas).
  - `src/ema.py`: EMA dengan warmup decay `min(decay, (1+n)/(10+n))`.
  - `scripts/visualize_forward.py` → `forward_mnist.png`, `forward_cifar10.png`, `schedule_alphas_cumprod.png`.
  - Test: `test_diffusion.py` (21), `test_model.py` (10), `test_overfit.py` (1).

  Cuplikan krusial — classifier-free guidance:
  ```python
  eps = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null])).float()
  eps_cond, eps_uncond = eps.chunk(2)
  return eps_uncond + guidance_scale * (eps_cond - eps_uncond)
  ```
  Cuplikan krusial — attention dari nol:
  ```python
  q, k, v = qkv.unbind(dim=1)  # masing-masing (B, heads, d, N), N = H*W
  scores = torch.einsum("bhdn,bhdm->bhnm", q, k) / math.sqrt(head_dim)
  out = torch.einsum("bhnm,bhdm->bhdn", scores.softmax(dim=-1), v).reshape(b, c, h, w)
  ```

### 2. TAHAP PENANGANAN EROR / DEBUGGING

**Eror 1 — Test `q_sample` gagal di t = 0**
- **Pesan Eror:**
  ```
  >       assert torch.allclose(diffusion.q_sample(x0, t0, noise), x0, atol=0.02)  # t=0 ~ x0
  E       assert False
  FAILED tests/test_diffusion.py::test_q_sample_matches_formula_and_extremes - ...
  1 failed, 55 passed in 17.65s
  ```
- **Analisis:** kodenya benar; asumsi test-nya yang salah. Di t = 0, `x_t = √ᾱ₀·x₀ + √(1−ᾱ₀)·ε` dengan √(1−ᾱ₀) = √(1e-4) = 0,01. Karena ε ~ N(0, 1) bisa bernilai ±4, selisih x_t − x₀ bisa mencapai ±0,04 > toleransi 0,02.
- **Solusi:** toleransi diturunkan dari rumus, bukan angka tebakan: `|x_t − x₀| ≤ 0,0101·|ε| + 1e-4`. Test lain (`x_t` sama dengan rumus manual untuk t acak, `predict_x0` membalik `q_sample`, variance preserving) memastikan implementasi benar. Hasil: `56 passed`.
- **Pelajaran:** toleransi numerik di test harus diturunkan dari matematika prosesnya.

**Antisipasi 2 — Gradien terblokir oleh inisialisasi nol**
- **Masalah:** `conv_out`, `conv2` di ResBlock, dan `proj` di attention sengaja di-init nol (blok mulai sebagai identitas, training lebih stabil). Akibatnya, pada model baru gradien ke lapisan sebelumnya bernilai nol, sehingga test "semua parameter dapat gradien" akan gagal palsu.
- **Solusi:** pada test gradien, semua bobot diacak dulu (`normal_(std=0.02)`) agar yang diuji adalah keterhubungan graf komputasi, bukan nilai inisialisasi. Tes overfit membuktikan model dengan inisialisasi asli tetap belajar.

### 3. TAHAP OPTIMASI / VERIFIKASI
- **Tes overfit 1 batch** (`python -m scripts.overfit_one_batch`, 32 gambar MNIST, 400 step, lr 1e-3):
  - t & noise tetap: loss **1,0 → 0,0001** (rata-rata 20 step: 0,4531 → 0,0001)
  - t & noise acak: loss 0,4483 → 0,0203 (turun 95,5%)
  - Grafik: `reports/figures/overfit_one_batch.png`
- Sampling dengan CFG menghasilkan shape `(B, C, 32, 32)` dan nilai tepat di [-1, 1].
