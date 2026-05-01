from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from personalization_platform.experiments.readout import (
    build_exposure_outcome_series,
    build_guardrail_metrics,
    build_primary_metrics,
    build_request_outcome_series,
    build_sample_size_summary,
    build_scale_caveat,
    build_srm_check,
    build_treatment_uncertainty,
    build_treatment_diagnostics,
    build_treatment_summaries,
)


def analyze_live_experiment(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    serving_dir = resolve_serving_run_dir(config)
    request_events = pd.read_csv(serving_dir / "request_events.csv")
    exposure_events = pd.read_csv(serving_dir / "exposure_events.csv")
    click_events = pd.read_csv(serving_dir / "click_events.csv")
    response_events = pd.read_csv(serving_dir / "response_events.csv")
    experiment = config["experiment"]

    assignments = build_live_assignment_table(request_events)
    assigned_exposures = build_live_assigned_exposures(
        exposure_events=exposure_events,
        click_events=click_events,
    )

    treatment_summaries = build_treatment_summaries(assignments, assigned_exposures, config=config)
    treatment_summaries = ensure_all_treatments_present(
        treatment_summaries=treatment_summaries,
        experiment=experiment,
        config=config,
    )
    primary_metrics = build_primary_metrics(treatment_summaries, experiment, config=config)
    guardrails = build_guardrail_metrics(treatment_summaries, experiment)
    srm_check = build_srm_check(assignments, experiment)
    diagnostics = build_treatment_diagnostics(assigned_exposures, config=config)
    live_guardrails = build_live_guardrails(response_events=response_events, request_events=request_events)

    summary = {
        "experiment_id": experiment["experiment_id"],
        "analysis_name": "live_experiment_readout_local",
        "serving_input_dir": str(serving_dir),
        "primary_metrics": primary_metrics,
        "guardrails": guardrails,
        "live_guardrails": live_guardrails,
        "srm_check": srm_check,
        "treatment_summaries": treatment_summaries,
        "sample_size_summary": build_sample_size_summary(treatment_summaries),
    }
    readout_bundle = {
        "experiment_id": experiment["experiment_id"],
        "serving_input_dir": str(serving_dir),
        "analysis_scope": "local_serving_log_readout",
        "metric_definitions": {
            "primary": "Top-1 and top-2 CTR are computed from request-time exposure and click logs joined by request_id and item_id.",
            "guardrail.fallback_rate": "Share of served requests that used the trending-only fallback path.",
            "guardrail.degraded_rate": "Share of served requests that reported one or more degraded modes.",
            "guardrail.srm": "Sample-ratio mismatch check over request-time assignments.",
        },
        "diagnostics": diagnostics | {"live_guardrails": live_guardrails},
        "caveats": [
            "This readout is built from local serving smoke logs, not external traffic or production telemetry.",
            build_scale_caveat(treatment_summaries),
            "The current readout is useful for structure, assignment integrity, and guardrail wiring; it is not decision-quality live evidence.",
        ],
    }
    return summary, readout_bundle


def resolve_serving_run_dir(config: dict[str, Any]) -> Path:
    serving_input = config["input"]
    base_dir = Path(
        serving_input.get(
            "serving_base_dir",
            serving_input.get("local_api_base_dir"),
        )
    )
    run_name = str(
        serving_input.get(
            "serving_run_name",
            serving_input.get("local_api_run_name"),
        )
    )
    required_files = [
        "request_events.csv",
        "exposure_events.csv",
        "click_events.csv",
        "response_events.csv",
    ]
    matches = [
        path
        for path in sorted(base_dir.glob(f"*_{run_name}"))
        if all((path / filename).exists() for filename in required_files)
    ]
    if not matches:
        raise FileNotFoundError(
            f"No completed serving-log bundles found under {base_dir} matching '*_{run_name}'."
        )
    return matches[-1]


def build_live_assignment_table(request_events: pd.DataFrame) -> pd.DataFrame:
    assignments = request_events[
        [
            "experiment_id",
            "assignment_unit",
            "assignment_unit_id",
            "request_id",
            "user_id",
            "hash_bucket",
            "treatment_id",
            "treatment_name",
            "is_control",
        ]
    ].copy()
    assignments["dataset_split"] = "serving_live"
    assignments["is_control"] = assignments["is_control"].fillna(0).astype(int)
    return assignments


def build_live_assigned_exposures(
    *,
    exposure_events: pd.DataFrame,
    click_events: pd.DataFrame,
) -> pd.DataFrame:
    click_labels = (
        click_events[["request_id", "item_id"]]
        .drop_duplicates()
        .assign(label=1)
    )
    exposures = exposure_events.merge(click_labels, on=["request_id", "item_id"], how="left")
    exposures["label"] = exposures["label"].fillna(0).astype(int)
    exposures["dataset_split"] = "serving_live"
    exposures["history_length"] = 0
    exposures["is_cold_start"] = False
    if "topic" not in exposures.columns:
        exposures["topic"] = "unknown_topic"
    if "creator_id" not in exposures.columns:
        exposures["creator_id"] = "creator_unknown"
    return exposures


def build_live_guardrails(
    *,
    response_events: pd.DataFrame,
    request_events: pd.DataFrame,
) -> dict[str, Any]:
    prepared = response_events.copy()
    if "degraded_modes" in prepared.columns:
        prepared["degraded_modes_parsed"] = prepared["degraded_modes"].map(
            lambda value: json.loads(value) if isinstance(value, str) and value else []
        )
        prepared["degraded_request"] = prepared["degraded_modes_parsed"].map(lambda values: int(bool(values)))
    else:
        prepared["degraded_request"] = 0
        prepared["degraded_modes_parsed"] = [[] for _ in range(len(prepared))]
    if "fallback_used" not in prepared.columns:
        prepared["fallback_used"] = 0

    by_treatment: dict[str, dict[str, Any]] = {}
    request_counts = request_events["treatment_id"].value_counts().to_dict()
    for treatment_id, rows in prepared.groupby("treatment_id", dropna=False):
        by_treatment[str(treatment_id)] = {
            "request_count": int(request_counts.get(treatment_id, len(rows))),
            "degraded_request_rate": float(rows["degraded_request"].mean()) if len(rows) else 0.0,
            "fallback_request_rate": float(rows["fallback_used"].astype(float).mean()) if len(rows) else 0.0,
            "degraded_mode_counts": count_degraded_modes(rows["degraded_modes_parsed"].tolist()),
        }
    return by_treatment


def ensure_all_treatments_present(
    *,
    treatment_summaries: dict[str, dict[str, Any]],
    experiment: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    completed = dict(treatment_summaries)
    for treatment in experiment["treatments"]:
        treatment_id = str(treatment["treatment_id"])
        if treatment_id in completed:
            continue
        empty_rows = pd.DataFrame(
            columns=[
                "request_id",
                "assignment_unit_id",
                "item_id",
                "label",
                "prediction",
                "rerank_score",
                "rank_shift",
                "post_rank",
                "creator_id",
                "topic",
            ]
        )
        uncertainty_inputs = build_request_outcome_series(empty_rows) | build_exposure_outcome_series(empty_rows)
        completed[treatment_id] = {
            "request_count": 0,
            "assignment_unit_count": 0,
            "exposure_row_count": 0,
            "top1_ctr": 0.0,
            "top2_ctr": 0.0,
            "mean_label": 0.0,
            "mean_prediction": 0.0,
            "mean_rerank_delta": 0.0,
            "average_rank_shift": 0.0,
            "top2_creator_repeat_rate": 0.0,
            "top2_topic_repeat_rate": 0.0,
            "top1_topic_mix": {},
            "top1_creator_mix": {},
            "uncertainty": build_treatment_uncertainty(uncertainty_inputs, config=config),
            "uncertainty_inputs": uncertainty_inputs,
        }
    return completed


def count_degraded_modes(rows: list[list[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for degraded_modes in rows:
        for degraded_mode in degraded_modes:
            counts[str(degraded_mode)] = counts.get(str(degraded_mode), 0) + 1
    return counts
