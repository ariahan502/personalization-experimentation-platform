from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from personalization_platform.experiments.assignment import assign_experiment
from personalization_platform.utils.artifacts import create_run_dir, write_json, write_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign a deterministic control/treatment experiment.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = config.get("run_name", "experiment_smoke")
    run_dir = create_run_dir(run_name, base_dir=config["artifacts"]["base_dir"])
    output_dir = Path(config["output"]["base_dir"]) / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=False)

    assignments, assigned_exposures, metrics, manifest = assign_experiment(config)
    assignments.to_csv(output_dir / "assignments.csv", index=False)
    assigned_exposures.to_csv(output_dir / "assigned_exposures.csv", index=False)

    write_yaml(run_dir / "config.yaml", config)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "manifest.json", manifest)
    print(f"Wrote experiment assignment bundle to {run_dir}")
    print(f"Wrote assignment outputs to {output_dir}")


if __name__ == "__main__":
    main()
