from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from personalization_platform.monitoring.lifecycle import analyze_model_lifecycle
from personalization_platform.utils.artifacts import (
    attach_lineage,
    build_upstream_run_entry,
    create_run_dir,
    write_json,
    write_yaml,
)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model lifecycle promotion and rollback readiness.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = config.get("run_name", "model_lifecycle_smoke")
    run_dir = create_run_dir(run_name, base_dir=config["artifacts"]["base_dir"])

    summary, lifecycle_report = analyze_model_lifecycle(config)
    manifest = attach_lineage(
        {
            "lifecycle_name": summary["lifecycle_name"],
            "decision": summary["decision"],
            "candidate_model_name": summary["candidate_model_name"],
            "fallback_target": summary["fallback_target"],
            "input_dirs": lifecycle_report["input_dirs"],
            "config_snapshot": config,
        },
        run_dir=run_dir,
        config=config,
        upstream_runs=[
            build_upstream_run_entry(label="ranker", path=lifecycle_report["input_dirs"]["ranker_dir"]),
            build_upstream_run_entry(label="ranker_compare", path=lifecycle_report["input_dirs"]["ranker_compare_dir"]),
            build_upstream_run_entry(label="monitoring", path=lifecycle_report["input_dirs"]["monitoring_dir"]),
            build_upstream_run_entry(
                label="live_experiment_readout",
                path=lifecycle_report["input_dirs"]["live_experiment_dir"],
            ),
        ],
    )

    write_yaml(run_dir / "config.yaml", config)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "lifecycle.json", lifecycle_report)
    write_json(run_dir / "manifest.json", manifest)
    print(f"Wrote model lifecycle bundle to {run_dir}")


if __name__ == "__main__":
    main()
