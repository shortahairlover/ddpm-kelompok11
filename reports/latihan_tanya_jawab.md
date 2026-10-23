# Latihan Tanya-Jawab Presentasi — Kelompok 11

Pertanyaan yang mungkin diajukan dosen, dikelompokkan per peran. Setiap anggota sebaiknya bisa menjawab pertanyaan bagiannya sendiri **tanpa membaca** (rubrik: penguasaan materi individu).

## Umum (semua anggota)

**1. Apa itu DDPM dalam satu kalimat?**
Model generatif yang belajar menghapus noise dari gambar secara bertahap; untuk membuat gambar baru, model mulai dari noise murni lalu menghapus noise selama 1000 langkah.

**2. Apa bedanya dengan GAN?**
GAN melatih generator melawan discriminator (sering tidak stabil, rawan *mode collapse*). DDPM hanya melatih satu jaringan dengan loss MSE sederhana (memprediksi noise) sehingga training stabil; kekurangannya, sampling lebih lambat karena banyak langkah.

**3. Kenapa memakai MNIST dulu sebelum CIFAR-10?**
Sebagai uji kebenaran yang murah: MNIST konvergen dalam ±27 menit. Kalau angka yang dihasilkan sesuai label, berarti seluruh pipeline (data → model → loss → training → sampling) benar sebelum menghabiskan ±3,8 jam untuk CIFAR-10.

**4. Apakah kode disalin dari tutorial?**
Tidak. Aturan *from scratch* dikunci di `CLAUDE.md` (dilarang `diffusers` dan menyalin tutorial), setiap modul punya test, dan audit setiap fase memeriksa import terlarang. Riwayat pembuatan ada di Log Interaksi dan transkrip.

## Naufal — Prompt Engineer & arsitektur

**5. Kenapa model memprediksi noise, bukan gambar langsung?**
Noise ε selalu berdistribusi N(0, 1) di semua timestep sehingga skala target dan loss stabil. Dari ε yang diprediksi, x₀ tetap bisa dihitung: x₀ = (x_t − √(1−ᾱ_t)·ε)/√ᾱ_t.

**6. Apa fungsi timestep embedding?**
Memberi tahu jaringan *seberapa banyak noise* di input. Tanpa itu, satu U-Net tidak tahu apakah harus menghapus sedikit noise (t kecil) atau hampir semuanya (t besar). Kami memakai embedding sinusoidal lalu MLP, ditambahkan ke setiap ResBlock.

**7. Bagaimana classifier-free guidance bekerja?**
Saat training, 10 % label diganti token "null", sehingga satu model belajar versi kondisional dan tanpa kondisi. Saat sampling: ε = ε_uncond + w·(ε_cond − ε_uncond). w lebih besar mendorong gambar lebih sesuai kelas. Hasil kami: w = 2 terbaik (FID 16,43).

**8. Kenapa ada skip connection di U-Net?**
Encoder memperkecil resolusi sehingga detail halus hilang; skip connection mengirim fitur resolusi tinggi langsung ke decoder agar detail bisa direkonstruksi.

**9. Kenapa beberapa lapisan diinisialisasi nol?**
Conv terakhir di ResBlock, proyeksi attention, dan conv output di-init nol agar setiap blok mulai sebagai fungsi identitas; training awal jadi lebih stabil. Tes overfit membuktikan model tetap belajar (loss 1,0 → 0,0001).

**10. Apa itu CLAUDE.md dan kenapa penting?**
*System prompt* kelompok yang dibaca otomatis oleh Claude Code di setiap sesi: scope, konvensi (data [−1, 1], token null = 10), dan aturan Git/audit. Ini menjaga konsistensi logika kode dari fase ke fase.

## Ardhian — Data pipeline

**11. Kenapa dinormalisasi ke [−1, 1], bukan [0, 1]?**
Agar skala data simetris dan sebanding dengan noise Gaussian (rata-rata 0). Clamp x̂₀ ke [−1, 1] saat sampling juga punya batas yang jelas.

**12. Kenapa MNIST di-pad menjadi 32×32?**
Supaya satu arsitektur U-Net dipakai untuk kedua dataset; 32 bisa dibagi 2 berkali-kali (32→16→8→4) untuk downsampling.

**13. Kenapa MNIST tidak di-flip, sedangkan CIFAR-10 di-flip?**
Angka yang dicerminkan bisa berubah makna atau tidak valid (misalnya 2, 3, 7); objek CIFAR-10 (kucing, mobil) tetap sama walau dicerminkan, sehingga flip menambah variasi data.

**14. Kenapa memakai Normalize(0.5, 0.5), bukan Lambda(x*2−1)?**
Hasilnya sama, tetapi di Windows worker DataLoader dibuat dengan *spawn* sehingga transform harus bisa di-pickle; fungsi lambda tidak bisa di-pickle. Diuji dengan test DataLoader 2 worker.

## Ali — Training & tuning

**15. Kenapa memakai EMA?**
Bobot model berfluktuasi antar-step; rata-rata eksponensial (decay 0,9999) meredam fluktuasi sehingga sampel lebih halus dan bersih. Perbandingan EMA vs non-EMA ada di `reports/figures/s2_ema_vs_raw.png`.

**16. Apa itu AMP dan kenapa dipakai?**
*Automatic mixed precision*: sebagian besar operasi di float16 → lebih cepat dan hemat VRAM (puncak 3,7 GB di GPU 6 GB). GradScaler mencegah gradien terlalu kecil hilang (underflow).

**17. Eksperimen mana yang terbaik dan kenapa?**
Pada epoch 30, batch 64 (FID 23,37) karena melakukan 2× jumlah update per epoch. lr 2e-4 adalah titik tengah: lr 1e-4 terlalu lambat, lr 4e-4 lebih berisik. Cosine schedule terburuk pada 32×32.

**18. Kenapa loss E3 lebih kecil tetapi FID lebih buruk?**
Loss MSE rata-rata atas semua timestep tidak sama dengan kualitas sampel; kualitas gambar akhir bergantung pada akurasi di timestep tertentu dan stabilitas bobot. Karena itu kami memakai FID untuk memutuskan.

**19. Kenapa FID dihitung dengan DDIM, bukan DDPM?**
DDPM-1000 untuk 5.000 sampel butuh ±2,3 jam per evaluasi (±20 jam untuk semua). DDIM-100 memakai model yang sama, 10× lebih cepat, dan FID-nya setara (26,23 vs 26,64 pada 2.000 sampel).

## Akhdan — Dokumentasi & error handling

**20. Error tersulit apa dan bagaimana menyelesaikannya?**
Contoh: skrip PowerShell eksperimen berisiko berhenti di tengah malam karena PowerShell 5.1 menganggap setiap baris stderr sebagai error. Diselesaikan dengan `ErrorActionPreference = Continue` + cek `$LASTEXITCODE`, lalu diuji jalur gagalnya dengan nama eksperimen fiktif.

**21. Bagaimana memastikan tidak ada bug sebelum lanjut fase?**
Audit 9 poin di setiap fase (task lengkap, `ruff`, `pytest`, kriteria fase, smoke test, scope, review diff, kebersihan repo, dokumentasi); hanya jika 0 error/0 bug baru push → PR → merge. Laporannya di folder `audits/`.

**22. Bagaimana Log Interaksi disusun?**
Per fase mengikuti template dosen: prompt kelompok apa adanya, alasan, output model, pesan eror utuh, analisis & solusi. Transkrip lengkap diekspor dari sesi Claude Code dan dikonversi ke Markdown; data mahasiswa kelompok lain disensor.

**23. Apa keterbatasan proyek ini?**
Resolusi 32×32 dan model 9 juta parameter (detail masih buram), FID 5.000 sampel (bias ke atas, hanya untuk perbandingan internal), eksperimen satu seed.
