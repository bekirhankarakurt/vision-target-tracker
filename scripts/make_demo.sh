#!/usr/bin/env bash
# Render docs/demo.gif from the synthetic test sequence with the telemetry overlay.
# Usage: scripts/make_demo.sh [build-dir]   (requires ffmpeg)
set -euo pipefail

build="${1:-build}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

"$build/vtt_synth" --out "$work/synthetic.avi" --gt "$work/gt.csv" --noise 0
"$build/vtt" --input "$work/synthetic.avi" --gt "$work/gt.csv" --headless --output "$work/overlay.avi"
ffmpeg -loglevel error -y -i "$work/overlay.avi" \
  -vf "fps=12,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" \
  docs/demo.gif
ls -lh docs/demo.gif
