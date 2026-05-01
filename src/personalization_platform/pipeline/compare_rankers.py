from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from personalization_platform.ranking.comparison import compare_rankers
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
    parser = argparse.ArgumentParser(description="Compare baseline rankers on the smoke ranking dataset.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = config.get("run_name", "ranker_compare_smoke")
    run_dir = create_run_dir(run_name, base_dir=config["artifacts"]["base_dir"])

    metrics, diagnostics = compare_rankers(config)
    manifest = attach_lineage(
        {
            "comparison_name": metrics["comparison_name"],
            "primary_variant_name": metrics["primary_variant_name"],
            "variant_names": sorted(metrics["variants"].keys()),
            "ranking_dataset_input_dir": metrics["ranking_dataset_input_dir"],
            "assumptions": [
                "All compared variants consume the same ranking dataset input dir for reproducible offline comparison.",
                "The retrieval-order baseline is materialized alongside trained variants so deltas are easy to audit.",
            ],
        },
        run_dir=run_dir,
        config=config,
        upstream_runs=[
            build_upstream_run_entry(label="ranking_dataset", path=metrics["ranking_dataset_input_dir"])
        ],
    )
    write_yaml(run_dir / "config.yaml", config)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "diagnostics.json", diagnostics)
    write_json(run_dir / "manifest.json", manifest)
    print(f"Wrote ranker comparison bundle to {run_dir}")


if __name__ == "__main__":
    main()
