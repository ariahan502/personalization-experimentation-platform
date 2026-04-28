from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from personalization_platform.reporting.bundle import build_reporting_bundle
from personalization_platform.utils.artifacts import create_run_dir, write_json, write_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a portfolio-facing reporting bundle from smoke artifacts.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = config.get("run_name", "portfolio_report_smoke")
    run_dir = create_run_dir(run_name, base_dir=config["artifacts"]["base_dir"])

    executive_summary, report_payload, report_markdown, architecture_markdown = build_reporting_bundle(config)
    write_yaml(run_dir / "config.yaml", config)
    write_json(run_dir / "executive_summary.json", executive_summary)
    write_json(run_dir / "report_payload.json", report_payload)
    (run_dir / "portfolio_report.md").write_text(report_markdown, encoding="utf-8")
    (run_dir / "architecture_note.md").write_text(architecture_markdown, encoding="utf-8")
    print(f"Wrote portfolio reporting bundle to {run_dir}")


if __name__ == "__main__":
    main()
