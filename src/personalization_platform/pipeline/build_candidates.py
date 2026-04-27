from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from personalization_platform.retrieval.trending import (
    build_trending_candidates,
    build_trending_manifest,
)
from personalization_platform.utils.artifacts import create_run_dir, write_json, write_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate sets from event-log outputs.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = config.get("run_name", "candidate_build")
    artifact_base_dir = config["artifacts"]["base_dir"]
    run_dir = create_run_dir(run_name, base_dir=artifact_base_dir)
    output_dir = Path(config["output"]["base_dir"]) / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=False)

    candidates, metrics = build_trending_candidates(config)
    candidates.to_csv(output_dir / "candidates.csv", index=False)
    manifest = build_trending_manifest(config=config, metrics=metrics, output_dir=output_dir)

    write_yaml(run_dir / "config.yaml", config)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "manifest.json", manifest)
    print(f"Wrote candidate bundle to {run_dir}")
    print(f"Wrote candidate outputs to {output_dir}")


if __name__ == "__main__":
    main()
