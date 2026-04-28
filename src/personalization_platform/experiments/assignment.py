from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


def assign_experiment(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    rerank_dir = resolve_rerank_dir(config)
    reranked_rows = pd.read_csv(rerank_dir / "reranked_rows.csv")
    experiment = validate_experiment_config(config)

    assignment_unit = experiment["assignment_unit"]
    request_rows = (
        reranked_rows[["request_id", "user_id", "dataset_split"]]
        .drop_duplicates()
        .sort_values(["request_id"])
        .reset_index(drop=True)
    )
    request_rows["assignment_unit_id"] = request_rows[assignment_unit].astype(str)
    request_rows["hash_bucket"] = request_rows["assignment_unit_id"].map(
        lambda unit_id: compute_hash_bucket(
            experiment_id=experiment["experiment_id"],
            salt=experiment["salt"],
            assignment_unit_id=unit_id,
        )
    )
    request_rows["treatment_id"] = request_rows["hash_bucket"].map(
        lambda bucket: choose_treatment(bucket=bucket, treatments=experiment["treatments"])
    )
    treatment_lookup = {
        treatment["treatment_id"]: treatment["treatment_name"] for treatment in experiment["treatments"]
    }
    request_rows["treatment_name"] = request_rows["treatment_id"].map(treatment_lookup)
    request_rows["experiment_id"] = experiment["experiment_id"]
    request_rows["assignment_unit"] = assignment_unit
    request_rows["is_control"] = request_rows["treatment_id"].map(
        lambda treatment_id: int(treatment_id == experiment["control_treatment_id"])
    )

    assignment_table = request_rows[
        [
            "experiment_id",
            "assignment_unit",
            "assignment_unit_id",
            "request_id",
            "user_id",
            "dataset_split",
            "hash_bucket",
            "treatment_id",
            "treatment_name",
            "is_control",
        ]
    ]

    assigned_exposures = reranked_rows.merge(
        assignment_table[
            ["request_id", "assignment_unit_id", "hash_bucket", "treatment_id", "treatment_name", "is_control"]
        ],
        on="request_id",
        how="left",
    )

    metrics = build_assignment_metrics(
        assignment_table=assignment_table,
        assigned_exposures=assigned_exposures,
        rerank_dir=rerank_dir,
        experiment=experiment,
    )
    manifest = build_assignment_manifest(experiment=experiment, metrics=metrics)
    return assignment_table, assigned_exposures, metrics, manifest


def resolve_rerank_dir(config: dict[str, Any]) -> Path:
    experiment_input = config["input"]
    base_dir = Path(experiment_input["rerank_base_dir"])
    run_name = experiment_input["rerank_run_name"]
    matches = sorted(base_dir.glob(f"*_{run_name}"))
    if not matches:
        raise FileNotFoundError(f"No rerank outputs found under {base_dir} matching '*_{run_name}'.")
    return matches[-1]


def validate_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("Config section 'experiment' must be a mapping.")
    required_fields = ["experiment_id", "assignment_unit", "salt", "treatments"]
    for field in required_fields:
        if field not in experiment:
            raise ValueError(f"Config field 'experiment.{field}' is required.")

    assignment_unit = experiment["assignment_unit"]
    if assignment_unit not in {"user_id", "request_id"}:
        raise ValueError("Config field 'experiment.assignment_unit' must be 'user_id' or 'request_id'.")

    treatments = experiment["treatments"]
    if not isinstance(treatments, list) or len(treatments) < 2:
        raise ValueError("Config field 'experiment.treatments' must contain at least two treatments.")

    total_weight = 0.0
    control_treatment_id: str | None = None
    normalized_treatments: list[dict[str, Any]] = []
    for treatment in treatments:
        if not isinstance(treatment, dict):
            raise ValueError("Each treatment entry must be a mapping.")
        for field in ("treatment_id", "treatment_name", "weight"):
            if field not in treatment:
                raise ValueError(f"Each treatment must include '{field}'.")
        weight = float(treatment["weight"])
        if weight <= 0:
            raise ValueError("Treatment weights must be positive.")
        total_weight += weight
        is_control = bool(treatment.get("is_control", False))
        if is_control:
            if control_treatment_id is not None:
                raise ValueError("Exactly one treatment may be marked as control.")
            control_treatment_id = str(treatment["treatment_id"])
        normalized_treatments.append(
            {
                "treatment_id": str(treatment["treatment_id"]),
                "treatment_name": str(treatment["treatment_name"]),
                "weight": weight,
                "is_control": is_control,
            }
        )

    if abs(total_weight - 1.0) > 1e-9:
        raise ValueError("Treatment weights must sum to 1.0.")
    if control_treatment_id is None:
        raise ValueError("One treatment must be marked with is_control: true.")

    return {
        "experiment_id": str(experiment["experiment_id"]),
        "assignment_unit": assignment_unit,
        "salt": str(experiment["salt"]),
        "treatments": normalized_treatments,
        "control_treatment_id": control_treatment_id,
    }


def compute_hash_bucket(*, experiment_id: str, salt: str, assignment_unit_id: str) -> float:
    payload = f"{experiment_id}|{salt}|{assignment_unit_id}".encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    integer = int(digest[:12], 16)
    return integer / float(16**12)


def choose_treatment(*, bucket: float, treatments: list[dict[str, Any]]) -> str:
    cumulative = 0.0
    for treatment in treatments:
        cumulative += float(treatment["weight"])
        if bucket < cumulative:
            return treatment["treatment_id"]
    return treatments[-1]["treatment_id"]


def build_assignment_metrics(
    *,
    assignment_table: pd.DataFrame,
    assigned_exposures: pd.DataFrame,
    rerank_dir: Path,
    experiment: dict[str, Any],
) -> dict[str, Any]:
    assignment_counts = assignment_table["treatment_id"].value_counts().to_dict()
    exposure_counts = assigned_exposures["treatment_id"].value_counts().to_dict()
    unique_assignments = assignment_table[
        ["experiment_id", "assignment_unit_id", "treatment_id"]
    ].drop_duplicates()
    unique_units = assignment_table["assignment_unit_id"].nunique()
    return {
        "experiment_id": experiment["experiment_id"],
        "assignment_unit": experiment["assignment_unit"],
        "rerank_input_dir": str(rerank_dir),
        "request_count": int(assignment_table["request_id"].nunique()),
        "assignment_unit_count": int(assignment_table["assignment_unit_id"].nunique()),
        "exposure_row_count": int(len(assigned_exposures)),
        "treatment_assignment_counts": {key: int(value) for key, value in assignment_counts.items()},
        "treatment_exposure_counts": {key: int(value) for key, value in exposure_counts.items()},
        "control_treatment_id": experiment["control_treatment_id"],
        "determinism_check": {
            "unique_assignment_rows": int(len(unique_assignments)),
            "unique_assignment_units": int(unique_units),
            "inconsistent_assignment_units": int(unique_units - len(unique_assignments)),
        },
    }


def build_assignment_manifest(*, experiment: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": experiment["experiment_id"],
        "assignment_unit": experiment["assignment_unit"],
        "control_treatment_id": experiment["control_treatment_id"],
        "treatments": experiment["treatments"],
        "hashing_strategy": "md5(experiment_id | salt | assignment_unit_id) mapped onto cumulative treatment weights",
        "assumptions": [
            "Assignments are deterministic for repeated runs with the same experiment_id, salt, assignment unit, and treatment weights.",
            "The smoke setup assigns on a single experiment only and does not manage cross-experiment interference.",
            "Assigned exposures inherit the request-level treatment from the chosen assignment unit so downstream analysis can audit what each treatment saw.",
        ],
        "metrics_snapshot": metrics,
    }
