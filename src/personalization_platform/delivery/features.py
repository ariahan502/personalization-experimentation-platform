from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from personalization_platform.reranking.policy import compute_freshness_minutes, freshness_bonus


def build_serving_feature_state(
    *,
    config: dict[str, Any],
    contextual_state: dict[str, Any],
) -> dict[str, Any]:
    log_config = config.get("input", {})
    serving_log_base_dir = log_config.get("serving_log_base_dir")
    serving_log_run_name = log_config.get("serving_log_run_name")
    if not serving_log_base_dir or not serving_log_run_name:
        return {
            "logs_available": False,
            "source_run_dir": None,
            "request_events": pd.DataFrame(),
            "exposure_events": pd.DataFrame(),
            "click_events": pd.DataFrame(),
        }

    run_dir = resolve_completed_serving_log_dir(
        base_dir=serving_log_base_dir,
        run_name=serving_log_run_name,
    )
    if run_dir is None:
        return {
            "logs_available": False,
            "source_run_dir": None,
            "request_events": pd.DataFrame(),
            "exposure_events": pd.DataFrame(),
            "click_events": pd.DataFrame(),
        }

    request_events = pd.read_csv(run_dir / "request_events.csv")
    exposure_events = pd.read_csv(run_dir / "exposure_events.csv")
    click_events = pd.read_csv(run_dir / "click_events.csv")
    return {
        "logs_available": True,
        "source_run_dir": str(run_dir),
        "request_events": request_events,
        "exposure_events": exposure_events,
        "click_events": click_events,
    }


def hydrate_request_time_features(
    *,
    candidate_rows: pd.DataFrame,
    history_context: dict[str, Any],
    contextual_state: dict[str, Any],
    serving_feature_state: dict[str, Any],
    scoring_weights: dict[str, Any],
    user_id: str,
    request_time: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    hydrated = candidate_rows.copy()
    hydrated["pre_rank"] = hydrated.index + 1
    hydrated["topic_affinity"] = hydrated["topic"].map(
        lambda topic: float(history_context["topic_counts"].get(str(topic), 0))
    )
    max_topic_count = max(history_context["topic_counts"].values(), default=0)
    hydrated["normalized_topic_affinity"] = hydrated["topic_affinity"].map(
        lambda value: value / max(max_topic_count, 1)
    )
    hydrated["seen_history_penalty"] = hydrated["item_id"].map(
        lambda item_id: float(item_id in history_context["history_item_ids"])
    )
    hydrated["click_prior"] = hydrated["item_id"].map(
        lambda item_id: contextual_state["item_click_priors"].get(item_id, 0.0)
    )
    hydrated["impression_prior"] = hydrated["item_impression_priors"].map(
        lambda value: float(value)
    ) if "item_impression_priors" in hydrated.columns else hydrated["item_id"].map(
        lambda item_id: contextual_state["item_impression_priors"].get(item_id, 0.0)
    )
    hydrated["freshness_minutes_since_last_seen"] = hydrated["item_id"].map(
        lambda item_id: compute_freshness_minutes(
            impressions=contextual_state["impressions"],
            item_id=item_id,
            request_time=request_time,
        )
    )
    hydrated["freshness_bonus"] = hydrated["freshness_minutes_since_last_seen"].map(
        lambda value: freshness_bonus(value=value, weight=float(scoring_weights.get("freshness_weight", 0.35)))
    )

    log_features = build_log_derived_features(
        user_id=user_id,
        candidate_rows=hydrated,
        contextual_state=contextual_state,
        serving_feature_state=serving_feature_state,
    )
    for column, values in log_features["candidate_feature_columns"].items():
        hydrated[column] = values

    topic_weight = float(scoring_weights.get("topic_affinity_weight", 1.0))
    click_prior_weight = float(scoring_weights.get("click_prior_weight", 0.5))
    impression_prior_weight = float(scoring_weights.get("impression_prior_weight", 0.2))
    history_penalty_weight = float(scoring_weights.get("history_seen_penalty", 0.75))
    recent_topic_click_weight = float(scoring_weights.get("recent_topic_click_weight", 0.35))
    recent_item_ctr_weight = float(scoring_weights.get("recent_item_ctr_weight", 0.25))
    recent_user_activity_weight = float(scoring_weights.get("recent_user_activity_weight", 0.15))

    hydrated["prediction"] = (
        topic_weight * hydrated["normalized_topic_affinity"].astype(float)
        + click_prior_weight * hydrated["click_prior"].astype(float)
        + impression_prior_weight * hydrated["impression_prior"].astype(float)
        + recent_topic_click_weight * hydrated["recent_topic_click_share"].astype(float)
        + recent_item_ctr_weight * hydrated["recent_item_ctr"].astype(float)
        + recent_user_activity_weight * hydrated["recent_user_click_rate"].astype(float)
        - history_penalty_weight * hydrated["seen_history_penalty"].astype(float)
    )

    feature_state_summary = {
        "serving_logs_available": bool(serving_feature_state.get("logs_available", False)),
        "serving_log_source_run_dir": serving_feature_state.get("source_run_dir"),
        "fallbacks": log_features["fallbacks"],
        "recent_request_event_count": int(log_features["recent_request_event_count"]),
        "recent_click_event_count": int(log_features["recent_click_event_count"]),
    }
    return hydrated, feature_state_summary


def build_log_derived_features(
    *,
    user_id: str,
    candidate_rows: pd.DataFrame,
    contextual_state: dict[str, Any],
    serving_feature_state: dict[str, Any],
) -> dict[str, Any]:
    request_events = serving_feature_state.get("request_events", pd.DataFrame())
    exposure_events = serving_feature_state.get("exposure_events", pd.DataFrame())
    click_events = serving_feature_state.get("click_events", pd.DataFrame())
    item_metadata_lookup = contextual_state["item_metadata_lookup"]

    if request_events.empty or exposure_events.empty:
        return {
            "candidate_feature_columns": {
                "recent_item_exposure_count": pd.Series([0.0] * len(candidate_rows)),
                "recent_item_click_count": pd.Series([0.0] * len(candidate_rows)),
                "recent_item_ctr": pd.Series([0.0] * len(candidate_rows)),
                "recent_topic_click_share": pd.Series([0.0] * len(candidate_rows)),
                "recent_user_request_count": pd.Series([0.0] * len(candidate_rows)),
                "recent_user_click_rate": pd.Series([0.0] * len(candidate_rows)),
            },
            "fallbacks": [
                "No prior serving log bundle was available, so recent serving-derived features defaulted to 0.0.",
            ],
            "recent_request_event_count": 0,
            "recent_click_event_count": 0,
        }

    user_request_count = int((request_events["user_id"].astype(str) == str(user_id)).sum())
    user_click_request_ids = set(click_events.loc[click_events["user_id"].astype(str) == str(user_id), "request_id"].astype(str).tolist())
    user_click_rate = user_request_count and (len(user_click_request_ids) / max(user_request_count, 1)) or 0.0

    exposure_counts = exposure_events["item_id"].astype(str).value_counts().to_dict()
    click_counts = click_events["item_id"].astype(str).value_counts().to_dict() if not click_events.empty else {}
    topic_click_counts = Counter(
        str(item_metadata_lookup.get(item_id, {}).get("topic", "unknown_topic"))
        for item_id in click_events["item_id"].astype(str).tolist()
    ) if not click_events.empty else Counter()
    total_topic_clicks = sum(topic_click_counts.values())

    item_exposure_features = candidate_rows["item_id"].map(lambda item_id: float(exposure_counts.get(str(item_id), 0)))
    item_click_features = candidate_rows["item_id"].map(lambda item_id: float(click_counts.get(str(item_id), 0)))
    item_ctr_features = candidate_rows["item_id"].map(
        lambda item_id: float(click_counts.get(str(item_id), 0)) / max(float(exposure_counts.get(str(item_id), 0)), 1.0)
    )
    topic_click_share = candidate_rows["topic"].map(
        lambda topic: float(topic_click_counts.get(str(topic), 0)) / max(total_topic_clicks, 1)
    )

    return {
        "candidate_feature_columns": {
            "recent_item_exposure_count": item_exposure_features,
            "recent_item_click_count": item_click_features,
            "recent_item_ctr": item_ctr_features,
            "recent_topic_click_share": topic_click_share,
            "recent_user_request_count": pd.Series([float(user_request_count)] * len(candidate_rows)),
            "recent_user_click_rate": pd.Series([float(user_click_rate)] * len(candidate_rows)),
        },
        "fallbacks": [],
        "recent_request_event_count": int(len(request_events)),
        "recent_click_event_count": int(len(click_events)),
    }


def resolve_completed_serving_log_dir(*, base_dir: str, run_name: str) -> Path | None:
    matches = sorted(Path(base_dir).glob(f"*_{run_name}"))
    required_files = [
        "request_events.csv",
        "exposure_events.csv",
        "click_events.csv",
    ]
    for candidate in reversed(matches):
        if all((candidate / filename).exists() for filename in required_files):
            return candidate
    return None
