from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from personalization_platform.data.event_log_schema import build_event_log_schema_contract
from personalization_platform.utils.artifacts import create_run_dir, write_json, write_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the event-log schema contract run bundle.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    return parser.parse_args()


def build_summary(contract: dict[str, Any]) -> dict[str, Any]:
    tables = contract["tables"]
    return {
        "contract_version": contract["contract_version"],
        "table_count": len(tables),
        "table_names": [table["name"] for table in tables],
        "required_field_counts": {
            table["name"]: len(table["required_fields"]) for table in tables
        },
        "assumption_count": len(contract["assumptions"]),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = config.get("run_name", "event_log_schema")
    run_dir = create_run_dir(run_name, base_dir=config.get("artifact_base_dir", "artifacts/runs"))
    contract = build_event_log_schema_contract()
    summary = build_summary(contract)
    write_yaml(run_dir / "config.yaml", config)
    write_json(run_dir / "schema_contract.json", contract)
    write_json(run_dir / "schema_summary.json", summary)
    print(f"Wrote event-log schema bundle to {run_dir}")


if __name__ == "__main__":
    main()
