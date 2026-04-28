from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from personalization_platform.ranking.logistic_baseline import (
    sanitize_metrics_for_json,
    train_logistic_baseline,
    write_model_pickle,
)
from personalization_platform.utils.artifacts import create_run_dir, write_json, write_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first baseline ranker from the ranking dataset.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = config.get("run_name", "ranker_smoke")
    run_dir = create_run_dir(run_name, base_dir=config["artifacts"]["base_dir"])
    output_dir = Path(config["output"]["base_dir"]) / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=False)

    metrics, scored_rows, manifest = train_logistic_baseline(config)
    model_artifacts = metrics["model_artifacts"]
    write_model_pickle(output_dir / "model.pkl", model_artifacts)
    scored_rows.to_csv(output_dir / "scored_rows.csv", index=False)

    write_yaml(run_dir / "config.yaml", config)
    write_json(run_dir / "metrics.json", sanitize_metrics_for_json(metrics))
    write_json(run_dir / "manifest.json", manifest)
    print(f"Wrote ranker bundle to {run_dir}")
    print(f"Wrote ranker outputs to {output_dir}")


if __name__ == "__main__":
    main()
