Set-Location $PSScriptRoot
$steps = @(
  @{ s = 's1';   neg = $false; msg = '1/5: deodorant on ANOTHER CORNER of the desk - rotate, move closer/further' },
  @{ s = 's2';   neg = $false; msg = '2/5: hold deodorant IN YOUR HAND - rotate, tilt, different heights' },
  @{ s = 's3';   neg = $false; msg = '3/5: turn the CAMERA to a new background (door, sofa, wall)' },
  @{ s = 'neg1'; neg = $true;  msg = '4/5: NO deodorant! show other bottles, cup, phone, empty desk' },
  @{ s = 'val1'; neg = $false; msg = '5/5: deodorant in a BRAND NEW place (floor, shelf, window)' }
)
foreach ($st in $steps) {
  Write-Host $st.msg -ForegroundColor Cyan
  if ($st.neg) {
    python vtt_custom.py capture --name deodorant --session $st.s --negative --count 60 --interval 0.3 --hint $st.msg
  } else {
    python vtt_custom.py capture --name deodorant --session $st.s --count 80 --interval 0.3 --hint $st.msg
  }
}
Write-Host 'ALL DONE' -ForegroundColor Green
