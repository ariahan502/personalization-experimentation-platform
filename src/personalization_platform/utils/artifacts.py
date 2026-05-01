from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def create_run_dir(run_name: str, base_dir: str = "artifacts/runs") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(base_dir) / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_run_dir_metadata(run_dir: Path) -> dict[str, str]:
    name = run_dir.name
    parts = name.split("_", maxsplit=3)
    if len(parts) < 4:
        return {
            "run_id": name,
            "timestamp": "",
            "run_name": name,
            "path": str(run_dir),
        }
    timestamp = "_".join(parts[:3])
    run_name = parts[3]
    return {
        "run_id": name,
        "timestamp": timestamp,
        "run_name": run_name,
        "path": str(run_dir),
    }


def build_run_manifest_metadata(
    *,
    run_dir: Path,
    output_dir: Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = parse_run_dir_metadata(run_dir)
    if output_dir is not None:
        metadata["output_dir"] = str(output_dir)
    if config is not None:
        metadata["artifact_base_dir"] = str(config.get("artifacts", {}).get("base_dir", run_dir.parent))
    return metadata


def build_upstream_run_entry(*, label: str, path: str | Path) -> dict[str, str]:
    path_obj = Path(path)
    return {"label": label, **parse_run_dir_metadata(path_obj)}


def attach_lineage(
    manifest: dict[str, Any],
    *,
    run_dir: Path,
    output_dir: Path | None = None,
    config: dict[str, Any] | None = None,
    upstream_runs: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    enriched = dict(manifest)
    enriched["run_metadata"] = build_run_manifest_metadata(run_dir=run_dir, output_dir=output_dir, config=config)
    enriched["upstream_runs"] = upstream_runs or []
    return enriched


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
