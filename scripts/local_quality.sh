#!/usr/bin/env bash
set -euo pipefail

# Keep local quality as the fast smoke check.
# Use `bash scripts/ci_medium.sh` after substantive retrieval or ranking changes.
bash scripts/ci_smoke.sh
