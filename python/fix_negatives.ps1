Set-Location $PSScriptRoot
# Hard negative mining: frames WITHOUT the deodorant (jacket, shirt, person, chair) -> "not deodorant"
function Capture($session, $count, $interval, $hint) {
  for ($try = 1; $try -le 4; $try++) {
    Start-Sleep -Seconds 3   # the Orbbec driver needs a moment after another program releases the camera
    python vtt_custom.py capture --name deodorant --session $session --negative --autostart --count $count --interval $interval --hint $hint 2> capture.log
    $n = (Get-ChildItem "datasets/deodorant/raw/$session" -Filter *.jpg -ErrorAction SilentlyContinue).Count
    Write-Host "$session : $n images (try $try)"
    if ($n -ge $count) { return }
  }
  throw "camera did not deliver frames for $session"
}
Capture 'neg2' 120 0.25 'NO deodorant - recording automatically'
Capture 'negv' 40 0.3 'NO deodorant - test set'
python vtt_custom.py samlabel --name deodorant --session neg2 --session negv --redo
python vtt_custom.py build --name deodorant --val-session val1 --val-session negv
python vtt_custom.py train --name deodorant_seg2 --data-yaml datasets/deodorant/data.yaml --base models/deodorant_seg.pt --epochs 30
Write-Host 'DONE - new model: models/deodorant_seg2.pt' -ForegroundColor Green
