# Jalankan seluruh evaluasi FID Fase 6 (T6.2, T6.8) secara berurutan.
# Pemakaian (PowerShell, dari root repo):
#   powershell -ExecutionPolicy Bypass -File scripts\run_fid.ps1
# Hasil ditambahkan ke reports\fid_results.csv; log konsol ke fid_console.log.

$ErrorActionPreference = "Continue"  # lihat catatan stderr PS 5.1 di run_experiments.ps1
$env:PYTHONIOENCODING = "utf-8"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
Set-Location $root
$log = Join-Path $root "fid_console.log"

function Get-Ckpt([string]$prefix, [string]$file) {
    $run = Get-ChildItem -Directory "runs" -Filter "$prefix*" | Sort-Object Name | Select-Object -Last 1
    if (-not $run) { throw "run $prefix tidak ditemukan" }
    return Join-Path $run.FullName "checkpoints\$file"
}

function Invoke-Eval([string]$title, [string[]]$evalArgs) {
    Write-Output "=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') mulai: $title ==="
    & $python -m src.evaluate @evalArgs 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        Write-Output "GAGAL: $title (exit $LASTEXITCODE), lihat $log"
        exit $LASTEXITCODE
    }
    Write-Output "=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') selesai: $title ==="
}

$e1_30 = Get-Ckpt "e1_baseline" "epoch_030.pt"
$e1_100 = Get-Ckpt "e1_baseline" "epoch_100.pt"

# 1) E1-E5 pada epoch yang sama (30), w = 3, DDIM-100, 5.000 sampel
Invoke-Eval "E1-E5 epoch 30" @(
    "--ckpt", $e1_30, (Get-Ckpt "e2_lr1e-4" "epoch_030.pt"), (Get-Ckpt "e3_lr4e-4" "epoch_030.pt"),
    (Get-Ckpt "e4_batch64" "epoch_030.pt"), (Get-Ckpt "e5_cosine" "epoch_030.pt"),
    "--labels", "E1", "E2", "E3", "E4", "E5", "--guidance", "3"
)

# 2) Guidance sweep pada checkpoint akhir E1 (epoch 100)
Invoke-Eval "E1 epoch 100 guidance sweep" @(
    "--ckpt", $e1_100, "--labels", "E1", "--guidance", "0", "1", "2", "3", "5"
)

# 3) DDPM-1000 vs DDIM-100 (2.000 sampel, w = 3)
Invoke-Eval "E1 epoch 100 DDIM-100 2k" @(
    "--ckpt", $e1_100, "--labels", "E1", "--guidance", "3", "--num-samples", "2000"
)
Invoke-Eval "E1 epoch 100 DDPM-1000 2k" @(
    "--ckpt", $e1_100, "--labels", "E1", "--guidance", "3", "--num-samples", "2000",
    "--sampler", "ddpm"
)
Write-Output "semua evaluasi FID selesai"
