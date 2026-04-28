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
run_step "Validate project scaffold" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.show_blueprint --config configs/project_scaffold.yaml
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
run_step "Assign smoke experiment" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.assign_experiment --config configs/experiment_smoke.yaml
run_step "Analyze smoke experiment" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.analyze_experiment --config configs/experiment_analysis_smoke.yaml
run_step "Monitor smoke quality" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.monitor_quality --config configs/monitoring_smoke.yaml
run_step "Smoke local ranked-feed API" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.serve_ranked_feed --config configs/local_api.yaml
run_step "Build portfolio reporting bundle" \
  env PYTHONPATH=src python -m personalization_platform.pipeline.build_portfolio_report --config configs/portfolio_report_smoke.yaml

echo
echo "Smoke quality run completed successfully."
