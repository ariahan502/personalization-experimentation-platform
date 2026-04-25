from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from personalization_platform.data.event_log_config import validate_event_log_config
from personalization_platform.utils.artifacts import create_run_dir, write_json, write_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and summarize the event-log input config.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    return parser.parse_args()


def build_summary(validation_report: dict[str, Any]) -> dict[str, Any]:
    path_checks = validation_report["path_checks"]
    return {
        "contract_version": validation_report["contract_version"],
        "dataset": validation_report["dataset"],
        "source_mode": validation_report["source_mode"],
        "split": validation_report["split"],
        "output_tables": validation_report["output_tables"],
        "path_check_counts": {
            "ok": sum(1 for check in path_checks if check["status"] == "ok"),
            "missing": sum(1 for check in path_checks if check["status"] == "missing"),
            "declared_missing_allowed": sum(
                1 for check in path_checks if check["status"] == "declared_missing_allowed"
            ),
            "planned_create": sum(
                1 for check in path_checks if check["status"] == "planned_create"
            ),
        },
        "missing_required_paths": validation_report["missing_required_paths"],
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validation_report = validate_event_log_config(config)
    run_name = config.get("run_name", "event_log_config_validation")
    artifact_base_dir = config.get("artifacts", {}).get("base_dir", "artifacts/runs")
    run_dir = create_run_dir(run_name, base_dir=artifact_base_dir)
    write_yaml(run_dir / "config.yaml", config)
    write_json(run_dir / "config_validation_report.json", validation_report)
    write_json(run_dir / "config_validation_summary.json", build_summary(validation_report))
    print(f"Wrote event-log config validation bundle to {run_dir}")


if __name__ == "__main__":
    main()
