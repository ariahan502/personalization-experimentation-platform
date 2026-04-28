from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd


def analyze_experiment(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    assignment_dir = resolve_assignment_dir(config)
    assignments = pd.read_csv(assignment_dir / "assignments.csv")
    assigned_exposures = pd.read_csv(assignment_dir / "assigned_exposures.csv")
    experiment = config["experiment"]

    treatment_summaries = build_treatment_summaries(assignments, assigned_exposures)
    primary_metrics = build_primary_metrics(treatment_summaries, experiment)
    guardrails = build_guardrail_metrics(treatment_summaries, experiment)
    srm_check = build_srm_check(assignments, experiment)
    diagnostics = build_treatment_diagnostics(assigned_exposures)

    summary = {
        "experiment_id": experiment["experiment_id"],
        "analysis_name": "experiment_readout_smoke",
        "assignment_input_dir": str(assignment_dir),
        "primary_metrics": primary_metrics,
        "guardrails": guardrails,
        "srm_check": srm_check,
        "treatment_summaries": treatment_summaries,
    }
    readout_bundle = {
        "experiment_id": experiment["experiment_id"],
        "assignment_input_dir": str(assignment_dir),
        "analysis_scope": "offline_smoke_readout",
        "metric_definitions": {
            "primary": "Top-1 click-through rate computed from the reranked top item per assigned request.",
            "guardrail.average_rank_shift": "Average absolute reranking shift within each treatment's assigned exposures.",
            "guardrail.creator_repeat_rate": "Share of requests whose top-2 exposures repeat a creator within the treatment.",
            "guardrail.mean_prediction": "Average pre-rerank model prediction over assigned exposures.",
            "srm": "Chi-square sample-ratio mismatch check against configured treatment weights on assignment units.",
        },
        "diagnostics": diagnostics,
        "caveats": [
            "This readout is based on offline smoke fixtures and simulated treatment assignment, not online experiment traffic.",
            "Primary metrics are useful for pipeline validation and structure only; they are not statistically reliable decision evidence at this scale.",
            "SRM is included as an integrity check even though the smoke sample is intentionally tiny.",
        ],
    }
    return summary, readout_bundle


def resolve_assignment_dir(config: dict[str, Any]) -> Path:
    experiment_input = config["input"]
    base_dir = Path(experiment_input["assignment_base_dir"])
    run_name = experiment_input["assignment_run_name"]
    matches = sorted(base_dir.glob(f"*_{run_name}"))
    if not matches:
        raise FileNotFoundError(
            f"No experiment assignment outputs found under {base_dir} matching '*_{run_name}'."
        )
    return matches[-1]


def build_treatment_summaries(
    assignments: pd.DataFrame,
    assigned_exposures: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    assignment_lookup = assignments.groupby("treatment_id")
    exposure_lookup = assigned_exposures.groupby("treatment_id")

    for treatment_id, assignment_rows in assignment_lookup:
        exposure_rows = exposure_lookup.get_group(treatment_id)
        summaries[treatment_id] = {
            "request_count": int(assignment_rows["request_id"].nunique()),
            "assignment_unit_count": int(assignment_rows["assignment_unit_id"].nunique()),
            "exposure_row_count": int(len(exposure_rows)),
            "top1_ctr": top1_ctr(exposure_rows),
            "mean_label": float(exposure_rows["label"].mean()) if len(exposure_rows) else 0.0,
            "mean_prediction": float(exposure_rows["prediction"].mean()) if len(exposure_rows) else 0.0,
            "average_rank_shift": float(exposure_rows["rank_shift"].abs().mean()) if len(exposure_rows) else 0.0,
            "top1_topic_mix": {
                str(key): int(value)
                for key, value in exposure_rows.loc[exposure_rows["post_rank"] == 1, "topic"].value_counts().to_dict().items()
            },
            "top1_creator_mix": {
                str(key): int(value)
                for key, value in exposure_rows.loc[exposure_rows["post_rank"] == 1, "creator_id"].value_counts().to_dict().items()
            },
        }
    return summaries


def top1_ctr(rows: pd.DataFrame) -> float:
    top_rows = rows.loc[rows["post_rank"] == 1]
    return float(top_rows["label"].mean()) if len(top_rows) else 0.0


def build_primary_metrics(
    treatment_summaries: dict[str, dict[str, Any]],
    experiment: dict[str, Any],
) -> dict[str, Any]:
    control_id = control_treatment_id(experiment)
    control_ctr = treatment_summaries[control_id]["top1_ctr"]
    metrics: dict[str, Any] = {"metric_name": "top1_ctr", "control_treatment_id": control_id}
    for treatment in experiment["treatments"]:
        treatment_id = treatment["treatment_id"]
        treatment_ctr = treatment_summaries[treatment_id]["top1_ctr"]
        metrics[treatment_id] = {
            "top1_ctr": treatment_ctr,
            "lift_vs_control": treatment_ctr - control_ctr,
        }
    return metrics


def build_guardrail_metrics(
    treatment_summaries: dict[str, dict[str, Any]],
    experiment: dict[str, Any],
) -> dict[str, Any]:
    guardrails: dict[str, Any] = {}
    for treatment in experiment["treatments"]:
        treatment_id = treatment["treatment_id"]
        summary = treatment_summaries[treatment_id]
        guardrails[treatment_id] = {
            "average_rank_shift": summary["average_rank_shift"],
            "mean_prediction": summary["mean_prediction"],
            "top1_creator_concentration": max(summary["top1_creator_mix"].values()) if summary["top1_creator_mix"] else 0,
        }
    return guardrails


def build_srm_check(assignments: pd.DataFrame, experiment: dict[str, Any]) -> dict[str, Any]:
    observed_counts = assignments["treatment_id"].value_counts().to_dict()
    total = len(assignments)
    chi_square = 0.0
    expected_counts: dict[str, float] = {}
    for treatment in experiment["treatments"]:
        treatment_id = treatment["treatment_id"]
        expected = total * float(treatment["weight"])
        observed = observed_counts.get(treatment_id, 0)
        expected_counts[treatment_id] = expected
        if expected > 0:
            chi_square += ((observed - expected) ** 2) / expected
    degrees_of_freedom = max(len(experiment["treatments"]) - 1, 1)
    p_value = chi_square_p_value_approx(chi_square=chi_square, dof=degrees_of_freedom)
    return {
        "observed_counts": {key: int(value) for key, value in observed_counts.items()},
        "expected_counts": expected_counts,
        "chi_square": chi_square,
        "degrees_of_freedom": degrees_of_freedom,
        "approx_p_value": p_value,
        "flagged": bool(p_value < 0.05),
    }


def chi_square_p_value_approx(*, chi_square: float, dof: int) -> float:
    if dof == 1:
        return math.erfc(math.sqrt(chi_square / 2.0))
    return math.exp(-0.5 * chi_square)


def build_treatment_diagnostics(assigned_exposures: pd.DataFrame) -> dict[str, Any]:
    by_treatment_topic = (
        assigned_exposures.groupby(["treatment_id", "topic"])
        .agg(
            exposure_rows=("item_id", "size"),
            positive_rate=("label", "mean"),
            avg_prediction=("prediction", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    valid_only = assigned_exposures.loc[assigned_exposures["dataset_split"] == "valid"]
    by_treatment_valid = (
        valid_only.groupby("treatment_id")
        .agg(
            exposure_rows=("item_id", "size"),
            top1_ctr=("label", lambda s: float(s[valid_only.loc[s.index, "post_rank"] == 1].mean()) if len(s) else 0.0),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "by_treatment_topic": by_treatment_topic,
        "valid_only_summary": by_treatment_valid,
    }


def control_treatment_id(experiment: dict[str, Any]) -> str:
    for treatment in experiment["treatments"]:
        if treatment.get("is_control"):
            return treatment["treatment_id"]
    raise ValueError("No control treatment configured.")
