from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from personalization_platform.delivery.simulation import simulate_serving_bundle
from personalization_platform.utils.artifacts import create_run_dir, write_json, write_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic serving-log simulation from reranked outputs.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = config.get("run_name", "simulated_serving_smoke")
    run_dir = create_run_dir(run_name, base_dir=config["artifacts"]["base_dir"])
    summary, log_frames, manifest = simulate_serving_bundle(config)
    write_yaml(run_dir / "config.yaml", config)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "manifest.json", manifest)
    log_frames["request_events"].to_csv(run_dir / "request_events.csv", index=False)
    log_frames["exposure_events"].to_csv(run_dir / "exposure_events.csv", index=False)
    log_frames["response_events"].to_csv(run_dir / "response_events.csv", index=False)
    log_frames["click_events"].to_csv(run_dir / "click_events.csv", index=False)
    print(f"Wrote simulated serving bundle to {run_dir}")


if __name__ == "__main__":
    main()
