from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_HISTORY_SEGMENTS = [
    {"name": "cold_start", "min_history_length": 0, "max_history_length": 0},
    {"name": "short_history", "min_history_length": 1, "max_history_length": 2},
    {"name": "medium_history", "min_history_length": 3, "max_history_length": 5},
    {"name": "long_history", "min_history_length": 6, "max_history_length": None},
]


def analyze_experiment(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    assignment_dir = resolve_assignment_dir(config)
    assignments = pd.read_csv(assignment_dir / "assignments.csv")
    assigned_exposures = pd.read_csv(assignment_dir / "assigned_exposures.csv")
    experiment = config["experiment"]

    treatment_summaries = build_treatment_summaries(assignments, assigned_exposures)
    primary_metrics = build_primary_metrics(treatment_summaries, experiment)
    guardrails = build_guardrail_metrics(treatment_summaries, experiment)
    srm_check = build_srm_check(assignments, experiment)
    diagnostics = build_treatment_diagnostics(assigned_exposures, config=config)

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
            "primary.top2_ctr": "Average click-through rate across the top-2 reranked exposures within each treatment.",
            "primary.mean_label": "Average click label across all assigned exposures within each treatment.",
            "guardrail.average_rank_shift": "Average absolute reranking shift within each treatment's assigned exposures.",
            "guardrail.creator_repeat_rate": "Share of requests whose top-2 exposures repeat a creator within the treatment.",
            "guardrail.topic_repeat_rate": "Share of requests whose top-2 exposures repeat a topic within the treatment.",
            "guardrail.top1_topic_concentration": "Largest single-topic share among top-1 exposures within the treatment.",
            "guardrail.mean_prediction": "Average pre-rerank model prediction over assigned exposures.",
            "guardrail.mean_rerank_delta": "Average difference between rerank_score and model prediction over assigned exposures.",
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
            "top2_ctr": topk_ctr(exposure_rows, k=2),
            "mean_label": float(exposure_rows["label"].mean()) if len(exposure_rows) else 0.0,
            "mean_prediction": float(exposure_rows["prediction"].mean()) if len(exposure_rows) else 0.0,
            "mean_rerank_delta": (
                float((exposure_rows["rerank_score"] - exposure_rows["prediction"]).mean())
                if len(exposure_rows)
                else 0.0
            ),
            "average_rank_shift": float(exposure_rows["rank_shift"].abs().mean()) if len(exposure_rows) else 0.0,
            "top2_creator_repeat_rate": repeat_rate(exposure_rows, rank_limit=2, value_column="creator_id"),
            "top2_topic_repeat_rate": repeat_rate(exposure_rows, rank_limit=2, value_column="topic"),
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


def topk_ctr(rows: pd.DataFrame, *, k: int) -> float:
    top_rows = rows.loc[rows["post_rank"] <= k]
    return float(top_rows["label"].mean()) if len(top_rows) else 0.0


def build_primary_metrics(
    treatment_summaries: dict[str, dict[str, Any]],
    experiment: dict[str, Any],
) -> dict[str, Any]:
    control_id = control_treatment_id(experiment)
    control_summary = treatment_summaries[control_id]
    metrics: dict[str, Any] = {"metric_name": "top1_ctr", "control_treatment_id": control_id}
    for treatment in experiment["treatments"]:
        treatment_id = treatment["treatment_id"]
        treatment_summary = treatment_summaries[treatment_id]
        metrics[treatment_id] = {
            "top1_ctr": treatment_summary["top1_ctr"],
            "lift_vs_control": treatment_summary["top1_ctr"] - control_summary["top1_ctr"],
            "top2_ctr": treatment_summary["top2_ctr"],
            "top2_ctr_lift_vs_control": treatment_summary["top2_ctr"] - control_summary["top2_ctr"],
            "mean_label": treatment_summary["mean_label"],
            "mean_label_lift_vs_control": treatment_summary["mean_label"] - control_summary["mean_label"],
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
            "mean_rerank_delta": summary["mean_rerank_delta"],
            "top2_creator_repeat_rate": summary["top2_creator_repeat_rate"],
            "top2_topic_repeat_rate": summary["top2_topic_repeat_rate"],
            "top1_creator_concentration": concentration(summary["top1_creator_mix"]),
            "top1_topic_concentration": concentration(summary["top1_topic_mix"]),
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


def build_treatment_diagnostics(assigned_exposures: pd.DataFrame, *, config: dict[str, Any]) -> dict[str, Any]:
    prepared = add_history_segments(assigned_exposures, config=config)
    by_treatment_topic = (
        prepared.groupby(["treatment_id", "topic"])
        .agg(
            exposure_rows=("item_id", "size"),
            positive_rate=("label", "mean"),
            avg_prediction=("prediction", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    valid_only = prepared.loc[prepared["dataset_split"] == "valid"]
    by_treatment_valid = (
        valid_only.groupby("treatment_id")
        .agg(
            exposure_rows=("item_id", "size"),
            top1_ctr=("label", lambda s: float(s[valid_only.loc[s.index, "post_rank"] == 1].mean()) if len(s) else 0.0),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    by_dataset_split = build_treatment_slice(prepared, group_column="dataset_split")
    by_candidate_source = build_treatment_slice(prepared, group_column="candidate_source")
    by_cold_start = build_treatment_slice(prepared, group_column="is_cold_start", request_level=True)
    by_history_depth = build_treatment_slice(
        prepared,
        group_column="history_segment",
        request_level=True,
        ordered_labels=history_segment_names(config),
    )
    return {
        "by_treatment_topic": by_treatment_topic,
        "valid_only_summary": by_treatment_valid,
        "treatment_slices": {
            "by_dataset_split": by_dataset_split,
            "by_candidate_source": by_candidate_source,
            "by_cold_start_request": by_cold_start,
            "by_history_depth_request": by_history_depth,
        },
    }


def control_treatment_id(experiment: dict[str, Any]) -> str:
    for treatment in experiment["treatments"]:
        if treatment.get("is_control"):
            return treatment["treatment_id"]
    raise ValueError("No control treatment configured.")


def concentration(value_counts: dict[str, int]) -> float:
    total = sum(value_counts.values())
    if total == 0:
        return 0.0
    return max(value_counts.values()) / total


def repeat_rate(rows: pd.DataFrame, *, rank_limit: int, value_column: str) -> float:
    if rows.empty:
        return 0.0
    repeated_request_count = 0
    request_count = 0
    for _, request_rows in rows.groupby("request_id"):
        top_rows = request_rows.loc[request_rows["post_rank"] <= rank_limit, value_column].astype(str).tolist()
        if not top_rows:
            continue
        request_count += 1
        if len(set(top_rows)) < len(top_rows):
            repeated_request_count += 1
    return repeated_request_count / request_count if request_count else 0.0


def add_history_segments(rows: pd.DataFrame, *, config: dict[str, Any]) -> pd.DataFrame:
    prepared = rows.copy()
    if "history_length" not in prepared.columns:
        prepared["history_length"] = 0
    if "is_cold_start" not in prepared.columns:
        prepared["is_cold_start"] = prepared["history_length"].fillna(0).astype(float).eq(0)
    prepared["is_cold_start"] = prepared["is_cold_start"].map(bool)
    prepared["history_segment"] = prepared["history_length"].fillna(0).astype(int).map(
        lambda history_length: assign_history_segment(history_length, config=config)
    )
    return prepared


def assign_history_segment(history_length: int, *, config: dict[str, Any]) -> str:
    for segment in resolve_history_segments(config):
        max_history_length = segment.get("max_history_length")
        if history_length < int(segment["min_history_length"]):
            continue
        if max_history_length is None or history_length <= int(max_history_length):
            return str(segment["name"])
    return "unmapped_history"


def resolve_history_segments(config: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics_config = config.get("diagnostics", {})
    return diagnostics_config.get("history_segments", DEFAULT_HISTORY_SEGMENTS)


def history_segment_names(config: dict[str, Any]) -> list[str]:
    return [str(segment["name"]) for segment in resolve_history_segments(config)]


def build_treatment_slice(
    rows: pd.DataFrame,
    *,
    group_column: str,
    request_level: bool = False,
    ordered_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for (treatment_id, segment_value), segment_rows in rows.groupby(["treatment_id", group_column], dropna=False):
        record = {
            "treatment_id": treatment_id,
            "segment": normalize_segment_value(segment_value),
            "request_count": int(segment_rows["request_id"].nunique()),
            "exposure_row_count": int(len(segment_rows)),
            "top1_ctr": top1_ctr(segment_rows),
            "top2_ctr": topk_ctr(segment_rows, k=2),
            "mean_label": float(segment_rows["label"].mean()) if len(segment_rows) else 0.0,
            "mean_prediction": float(segment_rows["prediction"].mean()) if len(segment_rows) else 0.0,
        }
        if request_level:
            record["repeat_rate_top2_topic"] = repeat_rate(segment_rows, rank_limit=2, value_column="topic")
            record["repeat_rate_top2_creator"] = repeat_rate(segment_rows, rank_limit=2, value_column="creator_id")
        records.append(record)
    if ordered_labels is None:
        return records
    ordering = {label: index for index, label in enumerate(ordered_labels)}
    return sorted(records, key=lambda row: (row["treatment_id"], ordering.get(str(row["segment"]), len(ordering))))


def normalize_segment_value(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return "cold_start" if value else "warm"
    return value
