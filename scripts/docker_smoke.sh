#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-personalization-platform:smoke}"

docker build -t "${IMAGE_NAME}" .
docker run --rm "${IMAGE_NAME}" bash scripts/ci_smoke.sh
