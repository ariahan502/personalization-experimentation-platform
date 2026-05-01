from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from personalization_platform.ranking.dataset import build_ranking_dataset
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
    parser = argparse.ArgumentParser(description="Build the first ranking dataset from candidates and event logs.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = config.get("run_name", "ranking_dataset")
    run_dir = create_run_dir(run_name, base_dir=config["artifacts"]["base_dir"])
    output_dir = Path(config["output"]["base_dir"]) / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=False)

    dataset, metrics, manifest = build_ranking_dataset(config)
    dataset.to_csv(output_dir / "ranking_dataset.csv", index=False)
    manifest = attach_lineage(
        manifest,
        run_dir=run_dir,
        output_dir=output_dir,
        config=config,
        upstream_runs=[
            build_upstream_run_entry(label="event_log", path=metrics["event_log_input_dir"]),
            build_upstream_run_entry(label="candidates", path=metrics["candidate_input_dir"]),
        ],
    )

    write_yaml(run_dir / "config.yaml", config)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "manifest.json", manifest)
    print(f"Wrote ranking dataset bundle to {run_dir}")
    print(f"Wrote ranking dataset to {output_dir}")


if __name__ == "__main__":
    main()
