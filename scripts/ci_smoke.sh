#!/usr/bin/env bash
set -euo pipefail

python -m compileall src
pytest
PYTHONPATH=src python -m personalization_platform.pipeline.show_blueprint --config configs/project_scaffold.yaml
