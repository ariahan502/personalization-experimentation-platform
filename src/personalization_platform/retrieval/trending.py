from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from personalization_platform.retrieval.common import load_event_log_inputs


def build_trending_candidates(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    event_log_inputs = load_event_log_inputs(config)
    candidate_count = config["retrieval"]["candidate_count"]
    candidates = build_trending_source_candidates(
        event_log_inputs=event_log_inputs,
        candidate_count=candidate_count,
    )
    metrics = build_trending_metrics(
        candidates=candidates,
        requests=event_log_inputs["requests"],
        clicked_lookup=event_log_inputs["clicked_lookup"],
        event_log_dir=event_log_inputs["event_log_dir"],
        candidate_count=candidate_count,
    )
    return candidates, metrics


def build_trending_source_candidates(
    *,
    event_log_inputs: dict[str, Any],
    candidate_count: int,
) -> pd.DataFrame:
    requests = event_log_inputs["requests"]
    impressions = event_log_inputs["impressions"]
    history_lookup = event_log_inputs["history_lookup"]
    candidate_rows: list[dict[str, Any]] = []

    requests_sorted = requests.sort_values("request_ts")
    for request in requests_sorted.itertuples(index=False):
        prior_impressions = impressions.loc[impressions["request_ts_lookup"] < request.request_ts].copy()
        if prior_impressions.empty:
            continue

        ranking = (
            prior_impressions.groupby("item_id")
            .agg(
                click_count=("clicked", "sum"),
                impression_count=("clicked", "size"),
                last_seen_ts=("request_ts_lookup", "max"),
                topic=("topic", "last"),
            )
            .reset_index()
        )
        ranking["trending_score"] = ranking["click_count"] * 1000 + ranking["impression_count"]
        ranking = ranking.sort_values(
            ["trending_score", "click_count", "impression_count", "last_seen_ts", "item_id"],
            ascending=[False, False, False, False, True],
        )

        history_items = history_lookup.get(request.request_id, set())
        ranking = ranking.loc[~ranking["item_id"].isin(history_items)].head(candidate_count)

        for rank_position, row in enumerate(ranking.itertuples(index=False), start=1):
            candidate_rows.append(
                {
                    "request_id": request.request_id,
                    "user_id": request.user_id,
                    "item_id": row.item_id,
                    "candidate_source": "trending",
                    "source_rank": rank_position,
                    "source_score": float(row.trending_score),
                    "source_click_count": int(row.click_count),
                    "source_impression_count": int(row.impression_count),
                    "source_last_seen_ts": row.last_seen_ts.strftime("%Y-%m-%dT%H:%M:%S"),
                    "topic": row.topic,
                }
            )

    return pd.DataFrame(candidate_rows)


def build_trending_metrics(
    *,
    candidates: pd.DataFrame,
    requests: pd.DataFrame,
    clicked_lookup: dict[str, list[str]],
    event_log_dir: Path,
    candidate_count: int,
) -> dict[str, Any]:
    candidate_requests = set(candidates["request_id"].unique()) if not candidates.empty else set()
    hit_requests = 0
    for request_id, clicked_items in clicked_lookup.items():
        request_candidates = set(
            candidates.loc[candidates["request_id"] == request_id, "item_id"].tolist()
        )
        if request_candidates.intersection(clicked_items):
            hit_requests += 1

    requests_with_click = sum(1 for items in clicked_lookup.values() if items)
    coverage = hit_requests / requests_with_click if requests_with_click else 0.0

    return {
        "source_name": "trending",
        "candidate_count_requested": candidate_count,
        "event_log_input_dir": str(event_log_dir),
        "row_counts": {
            "requests": int(len(requests)),
            "candidates": int(len(candidates)),
        },
        "requests_with_candidates": int(len(candidate_requests)),
        "requests_without_candidates": int(len(requests) - len(candidate_requests)),
        "average_candidates_per_scored_request": (
            float(len(candidates) / len(candidate_requests)) if candidate_requests else 0.0
        ),
        "distinct_candidate_items": (
            int(candidates["item_id"].nunique()) if not candidates.empty else 0
        ),
        "requests_with_click": int(requests_with_click),
        "clicked_item_hit_rate": coverage,
    }


def build_trending_manifest(
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "source_name": "trending",
        "source_type": "global_popularity",
        "event_log_input_dir": metrics["event_log_input_dir"],
        "candidate_output_path": str(output_dir / "candidates.csv"),
        "candidate_count_requested": config["retrieval"]["candidate_count"],
        "candidate_columns": [
            "request_id",
            "user_id",
            "item_id",
            "candidate_source",
            "source_rank",
            "source_score",
            "source_click_count",
            "source_impression_count",
            "source_last_seen_ts",
            "topic",
        ],
        "assumptions": [
            "Trending candidates are scored from prior impressions only, using clicks first and total prior impressions as a tie-breaker.",
            "Items already present in the user's visible history are excluded from the request-level candidate list.",
            "The first request for a smoke run may have no prior-trending candidates because there is no earlier interaction history.",
        ],
    }
