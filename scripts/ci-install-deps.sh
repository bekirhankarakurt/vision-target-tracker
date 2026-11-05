#!/usr/bin/env bash
# Install build dependencies on GitHub-hosted Ubuntu runners.
# apt occasionally stalls on the runner mirrors, so every attempt is time-boxed and retried.
# Usage: scripts/ci-install-deps.sh [extra packages...]
set -euo pipefail

packages=(libopencv-dev cmake ninja-build "$@")
apt_opts=(-o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)

for attempt in 1 2 3; do
  if sudo timeout 180 apt-get "${apt_opts[@]}" update &&
     sudo DEBIAN_FRONTEND=noninteractive timeout 420 apt-get "${apt_opts[@]}" install -y \
       --no-install-recommends "${packages[@]}"; then
    exit 0
  fi
  echo "apt attempt ${attempt} failed or timed out; retrying" >&2
  sleep $((attempt * 15))
done
echo "failed to install: ${packages[*]}" >&2
exit 1
