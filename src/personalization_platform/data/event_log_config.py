from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


VALID_SOURCE_MODES = {"smoke_fixture", "raw_mind"}
VALID_SPLITS = {"train", "valid", "test"}
REQUIRED_TABLES = ("requests", "impressions", "user_state", "item_state")
SUPPORTED_FILE_KEYS = ("behaviors", "news")


@dataclass(frozen=True)
class PathCheck:
    label: str
    path: str
    exists: bool
    required: bool
    kind: str
    status: str


def _get_required_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping.")
    return value


def _normalize_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Config field '{key}' must be a non-empty list.")
    normalized = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(f"Config field '{key}' must contain only non-empty strings.")
        normalized.append(entry)
    return normalized


def _normalize_file_map(section: dict[str, Any], key: str) -> dict[str, str]:
    value = section.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config field '{key}' must be a mapping of logical names to filenames.")
    normalized: dict[str, str] = {}
    for file_key in SUPPORTED_FILE_KEYS:
        filename = value.get(file_key)
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"Config field '{key}.{file_key}' must be a non-empty string.")
        normalized[file_key] = filename
    return normalized


def validate_event_log_config(config: dict[str, Any]) -> dict[str, Any]:
    input_config = _get_required_section(config, "input")
    output_config = _get_required_section(config, "output")
    artifact_config = _get_required_section(config, "artifacts")
    validation_config = _get_required_section(config, "validation")

    dataset = input_config.get("dataset")
    if dataset != "mind":
        raise ValueError("Config field 'input.dataset' must be 'mind' for the current event-log workflow.")

    source_mode = input_config.get("source_mode")
    if source_mode not in VALID_SOURCE_MODES:
        raise ValueError(
            "Config field 'input.source_mode' must be one of: smoke_fixture, raw_mind."
        )

    split = input_config.get("split")
    if split not in VALID_SPLITS:
        raise ValueError("Config field 'input.split' must be one of: train, valid, test.")

    selected_tables = _normalize_list(output_config.get("tables"), "output.tables")
    missing_tables = [table for table in REQUIRED_TABLES if table not in selected_tables]
    if missing_tables:
        raise ValueError(
            "Config field 'output.tables' must include the first event-log tables: "
            + ", ".join(missing_tables)
        )

    output_base_dir = output_config.get("base_dir")
    if not isinstance(output_base_dir, str) or not output_base_dir.strip():
        raise ValueError("Config field 'output.base_dir' must be a non-empty string.")

    artifact_base_dir = artifact_config.get("base_dir")
    if not isinstance(artifact_base_dir, str) or not artifact_base_dir.strip():
        raise ValueError("Config field 'artifacts.base_dir' must be a non-empty string.")

    require_existing_inputs = validation_config.get("require_existing_inputs")
    if not isinstance(require_existing_inputs, bool):
        raise ValueError("Config field 'validation.require_existing_inputs' must be a boolean.")

    if source_mode == "smoke_fixture":
        source_section_key = "smoke_fixture"
        source_description = "Local smoke fixture input rooted under data/fixtures."
    else:
        source_section_key = "raw_input"
        source_description = "Raw MIND input rooted under data/raw."

    source_section = _get_required_section(config, source_section_key)
    root_dir = source_section.get("root_dir")
    if not isinstance(root_dir, str) or not root_dir.strip():
        raise ValueError(f"Config field '{source_section_key}.root_dir' must be a non-empty string.")

    filenames = _normalize_file_map(source_section, "files")

    row_limit = input_config.get("row_limit")
    if row_limit is not None and (not isinstance(row_limit, int) or row_limit <= 0):
        raise ValueError("Config field 'input.row_limit' must be a positive integer when provided.")

    path_checks = build_path_checks(
        source_mode=source_mode,
        root_dir=root_dir,
        filenames=filenames,
        output_base_dir=output_base_dir,
        artifact_base_dir=artifact_base_dir,
        require_existing_inputs=require_existing_inputs,
    )

    return {
        "contract_version": "v1",
        "dataset": dataset,
        "source_mode": source_mode,
        "source_description": source_description,
        "split": split,
        "row_limit": row_limit,
        "output_tables": selected_tables,
        "path_checks": [asdict(check) for check in path_checks],
        "missing_required_paths": [
            check.path
            for check in path_checks
            if check.required and check.status == "missing"
        ],
        "expected_failure_behavior": (
            "The future build_event_log command should fail fast with a clear missing-path error "
            "when validation.require_existing_inputs is true and any required input path is absent."
        ),
    }


def build_path_checks(
    *,
    source_mode: str,
    root_dir: str,
    filenames: dict[str, str],
    output_base_dir: str,
    artifact_base_dir: str,
    require_existing_inputs: bool,
) -> list[PathCheck]:
    root_path = Path(root_dir)
    input_checks = [
        PathCheck(
            label=f"{source_mode}_root",
            path=str(root_path),
            exists=root_path.exists(),
            required=require_existing_inputs,
            kind="directory",
            status=_path_status(root_path.exists(), require_existing_inputs),
        )
    ]

    for file_key, filename in filenames.items():
        file_path = root_path / filename
        input_checks.append(
            PathCheck(
                label=f"{source_mode}_{file_key}_file",
                path=str(file_path),
                exists=file_path.exists(),
                required=require_existing_inputs,
                kind="file",
                status=_path_status(file_path.exists(), require_existing_inputs),
            )
        )

    output_checks = [
        PathCheck(
            label="output_base_dir",
            path=output_base_dir,
            exists=Path(output_base_dir).exists(),
            required=False,
            kind="directory",
            status="ok" if Path(output_base_dir).exists() else "planned_create",
        ),
        PathCheck(
            label="artifact_base_dir",
            path=artifact_base_dir,
            exists=Path(artifact_base_dir).exists(),
            required=False,
            kind="directory",
            status="ok" if Path(artifact_base_dir).exists() else "planned_create",
        ),
    ]

    return input_checks + output_checks


def _path_status(exists: bool, required: bool) -> str:
    if exists:
        return "ok"
    if required:
        return "missing"
    return "declared_missing_allowed"
