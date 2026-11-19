# Detect + track: YOLO server in its own window, then the C++ tracker on the camera.
# Usage: .\run_detect_track.ps1 [-Camera 2] [-Model models\deodorant_seg2.pt] [-Vtt C:\b\vtt\Release\vtt.exe]
param(
  [int]$Camera = 2,
  [string]$Model = 'models\deodorant_seg2.pt',
  [string]$Vtt = 'C:\b\vtt\Release\vtt.exe'
)
Set-Location $PSScriptRoot
$server = Start-Process python -ArgumentList @('vtt_detector_server.py', '--model', $Model) -PassThru
Write-Host 'Loading YOLO...' -ForegroundColor Cyan
Start-Sleep -Seconds 8
$env:PATH = "C:\opencv\build\x64\vc16\bin;$env:PATH"
try {
  & $Vtt --camera $Camera --detector 127.0.0.1:5601 --auto-lock --csv C:\b\live_fused.csv
} finally {
  Stop-Process -Id $server.Id -ErrorAction SilentlyContinue
}
