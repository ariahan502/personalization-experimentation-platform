#!/usr/bin/env bash
set -euo pipefail

run_step() {
  local label="$1"
  shift
  echo
  echo "==> ${label}"
  "$@"
}

run_step "Compile source tree" python -m compileall src
run_step "Run lightweight tests" pytest -q
run_step "Build smoke event log" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.build_event_log --config configs/mind_smoke.yaml
run_step "Build smoke candidates" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.build_candidates --config configs/candidates_smoke.yaml
run_step "Build smoke ranking dataset" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.build_ranking_dataset --config configs/ranking_dataset_smoke.yaml
run_step "Train smoke ranker" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.train_ranker --config configs/ranker_smoke.yaml
run_step "Compare smoke rankers" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.compare_rankers --config configs/ranker_compare_smoke.yaml
run_step "Rerank smoke feed" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.rerank_feed --config configs/rerank_smoke.yaml
run_step "Monitor smoke quality" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.monitor_quality --config configs/monitoring_smoke.yaml
run_step "Build deterministic serving simulation" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.simulate_serving_logs --config configs/serving_simulation_smoke.yaml
run_step "Analyze simulated live experiment" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.analyze_live_experiment --config configs/live_experiment_simulated_smoke.yaml
run_step "Evaluate simulated lifecycle readiness" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.evaluate_model_lifecycle --config configs/model_lifecycle_simulated_smoke.yaml

echo
echo "Simulated live validation completed successfully."
