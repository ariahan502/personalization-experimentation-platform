from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from personalization_platform.utils.artifacts import create_run_dir, write_json, write_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def build_summary(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "project_name": config["project"]["name"],
        "dataset": config["project"]["base_dataset"],
        "story": config["project"]["product_story"],
        "modules": config["modules"],
        "constraints": config["real_world_constraints"],
        "phases": config["phases"],
        "status": "scaffold_ready",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the scaffold summary run bundle.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = config.get("run_name", "project_scaffold")
    run_dir = create_run_dir(run_name)
    summary = build_summary(config)
    write_yaml(run_dir / "config.yaml", config)
    write_json(run_dir / "project_summary.json", summary)
    print(f"Wrote scaffold run bundle to {run_dir}")


if __name__ == "__main__":
    main()
