# Jalankan eksperimen E2-E5 secara berurutan (satu GPU, satu run per waktu).
# Pemakaian (PowerShell, dari root repo, venv aktif atau tidak):
#   powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_experiments.ps1 -Only e3_lr4e-4,e5_cosine
param(
    [string[]]$Only = @("e2_lr1e-4", "e3_lr4e-4", "e4_batch64", "e5_cosine")
)

# "Continue": di Windows PowerShell 5.1, baris stderr dari program native (misal warning
# Python) diubah menjadi ErrorRecord; dengan "Stop" satu warning saja akan menghentikan skrip.
# Keberhasilan tiap run dicek lewat $LASTEXITCODE.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
Set-Location $root

foreach ($name in $Only) {
    $config = "configs\experiments\$name.yaml"
    $log = Join-Path $root "$($name)_console.log"
    Write-Output "=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') mulai $name ==="
    # Out-File utf8 agar log konsol tidak tersimpan sebagai UTF-16 (default redirect PS 5.1)
    & $python -m src.train --config $config --name $name 2>&1 | Out-File -FilePath $log -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        Write-Output "GAGAL: $name (exit $LASTEXITCODE), lihat $log"
        exit $LASTEXITCODE
    }
    Write-Output "=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') selesai $name ==="
}
Write-Output "semua eksperimen selesai"
