$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (Get-Process OrbbecViewer -ErrorAction SilentlyContinue) { Write-Host 'UYARI: Orbbec Viewer acik, kamerayi o tutuyor olabilir. Kapat.' -ForegroundColor Yellow }
Write-Host '== Paketler kuruluyor (ilk seferde birkac dakika surer) ==' -ForegroundColor Cyan
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'pip install basarisiz' }
Write-Host '== Kamera aciliyor. Kapatmak icin pencerede q tusuna bas ==' -ForegroundColor Cyan
cmd /c "python vtt_detect.py --source orbbec --track --classes ""bottle,person,cup,cell phone"" --csv bottle_test.csv 2> run.log"
Get-Content run.log -TotalCount 20
