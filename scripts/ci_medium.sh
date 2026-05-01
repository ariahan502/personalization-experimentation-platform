#!/usr/bin/env bash
set -euo pipefail

run_step() {
  local label="$1"
  shift
  echo
  echo "==> ${label}"
  "$@"
}

run_step "Validate medium event-log config" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.validate_event_log_config --config configs/mind_medium.yaml
run_step "Build medium event log" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.build_event_log --config configs/mind_medium.yaml
run_step "Build medium candidates" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.build_candidates --config configs/candidates_medium.yaml
run_step "Build medium ranking dataset" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.build_ranking_dataset --config configs/ranking_dataset_medium.yaml
run_step "Train medium baseline ranker" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.train_ranker --config configs/ranker_medium.yaml
run_step "Compare medium rankers" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.compare_rankers --config configs/ranker_compare_medium.yaml

echo
echo "Medium validation run completed successfully."
