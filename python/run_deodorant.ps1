Set-Location $PSScriptRoot
# The Femto stream sometimes refuses to open after sitting idle / after another program released it -> retry.
for ($try = 1; $try -le 4; $try++) {
  Start-Sleep -Seconds 3
  cmd /c "python vtt_detect.py --model models/deodorant_seg2.pt --track --conf 0.35 --source orbbec 2> run.log"
  if (-not (Select-String -Path run.log -Pattern 'no frames|Stopped after 0 frame' -Quiet)) { break }
  Write-Host "camera did not start (try $try), retrying..." -ForegroundColor Yellow
}
Get-Content run.log | Where-Object { $_ -notmatch 'grabFrame' } | Select-Object -Last 5
