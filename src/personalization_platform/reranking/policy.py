from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from personalization_platform.retrieval.common import load_event_log_inputs


def rerank_feed(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    ranker_dir = resolve_ranker_dir(config)
    scored_rows = pd.read_csv(ranker_dir / "scored_rows.csv")
    event_log_inputs = load_event_log_inputs(config)
    requests = event_log_inputs["requests"].copy()
    impressions = event_log_inputs["impressions"].copy()
    item_state = event_log_inputs["item_state"].copy()

    requests["request_ts"] = pd.to_datetime(requests["request_ts"])
    request_ts_lookup = dict(zip(requests["request_id"], requests["request_ts"], strict=False))
    impressions["request_ts_lookup"] = pd.to_datetime(impressions["request_ts_lookup"])

    creator_map = config["reranking"].get("creator_map", {})
    item_creator_lookup = dict(zip(item_state["item_id"], item_state["creator_id"], strict=False))
    topic_penalty = float(config["reranking"]["topic_repeat_penalty"])
    creator_penalty = float(config["reranking"]["creator_repeat_penalty"])
    freshness_weight = float(config["reranking"]["freshness_weight"])

    ranked_rows: list[dict[str, Any]] = []
    changed_requests = 0

    for request_id, request_frame in scored_rows.groupby("request_id", sort=False):
        request_time = request_ts_lookup[request_id]
        working = request_frame.copy()
        working["request_ts_dt"] = pd.to_datetime(working["request_ts"])
        working["creator_id"] = working["item_id"].map(
            lambda item_id: item_creator_lookup.get(item_id, creator_map.get(item_id, "creator_unknown"))
        )
        working["freshness_minutes_since_last_seen"] = working["item_id"].map(
            lambda item_id: compute_freshness_minutes(
                impressions=impressions,
                item_id=item_id,
                request_time=request_time,
            )
        )
        working["freshness_bonus"] = working["freshness_minutes_since_last_seen"].map(
            lambda value: freshness_bonus(value=value, weight=freshness_weight)
        )

        selected_rows: list[dict[str, Any]] = []
        used_topics: set[str] = set()
        used_creators: set[str] = set()
        available = working.copy()
        while not available.empty:
            available = available.copy()
            available["diversity_penalty"] = available["topic"].map(
                lambda topic: topic_penalty if topic in used_topics else 0.0
            )
            available["creator_penalty"] = available["creator_id"].map(
                lambda creator_id: creator_penalty if creator_id in used_creators else 0.0
            )
            available["rerank_score"] = (
                available["prediction"].astype(float)
                + available["freshness_bonus"].astype(float)
                - available["diversity_penalty"].astype(float)
                - available["creator_penalty"].astype(float)
            )
            next_row = (
                available.sort_values(
                    ["rerank_score", "prediction", "merged_rank", "item_id"],
                    ascending=[False, False, True, True],
                )
                .iloc[0]
                .to_dict()
            )
            selected_rows.append(next_row)
            used_topics.add(str(next_row["topic"]))
            used_creators.add(str(next_row["creator_id"]))
            available = available.loc[available["item_id"] != next_row["item_id"]]

        before_order = request_frame.sort_values("prediction", ascending=False)["item_id"].tolist()
        after_order = [row["item_id"] for row in selected_rows]
        if before_order != after_order:
            changed_requests += 1

        for rerank_position, row in enumerate(selected_rows, start=1):
            row["pre_rank"] = before_order.index(row["item_id"]) + 1
            row["post_rank"] = rerank_position
            row["rank_shift"] = row["pre_rank"] - row["post_rank"]
            ranked_rows.append(row)

    reranked = pd.DataFrame(ranked_rows)
    reranked = reranked[
        [
            "request_id",
            "user_id",
            "item_id",
            "topic",
            "creator_id",
            "label",
            "dataset_split",
            "prediction",
            "merged_rank",
            "pre_rank",
            "post_rank",
            "rank_shift",
            "freshness_minutes_since_last_seen",
            "freshness_bonus",
            "diversity_penalty",
            "creator_penalty",
            "rerank_score",
            "candidate_source",
            "source_list",
            "source_details",
        ]
    ].sort_values(["request_id", "post_rank"]).reset_index(drop=True)

    metrics = build_rerank_metrics(
        reranked=reranked,
        request_count=int(scored_rows["request_id"].nunique()),
        changed_requests=changed_requests,
    )
    manifest = build_rerank_manifest(config=config, metrics=metrics, ranker_dir=ranker_dir)
    return reranked, metrics, manifest


def resolve_ranker_dir(config: dict[str, Any]) -> Path:
    rerank_input = config["input"]
    base_dir = Path(rerank_input["ranker_base_dir"])
    run_name = rerank_input["ranker_run_name"]
    matches = sorted(base_dir.glob(f"*_{run_name}"))
    if not matches:
        raise FileNotFoundError(
            f"No ranker outputs found under {base_dir} matching '*_{run_name}'."
        )
    return matches[-1]


def compute_freshness_minutes(*, impressions: pd.DataFrame, item_id: str, request_time: pd.Timestamp) -> float:
    prior = impressions.loc[
        (impressions["item_id"] == item_id) & (impressions["request_ts_lookup"] < request_time),
        "request_ts_lookup",
    ]
    if prior.empty:
        return 1_000_000.0
    delta = request_time - prior.max()
    return float(delta.total_seconds() / 60.0)


def freshness_bonus(*, value: float, weight: float) -> float:
    capped = min(value, 1_000_000.0)
    return weight * (60.0 / (capped + 60.0))


def build_rerank_metrics(*, reranked: pd.DataFrame, request_count: int, changed_requests: int) -> dict[str, Any]:
    before_mrr = request_mrr(reranked, rank_column="pre_rank")
    after_mrr = request_mrr(reranked, rank_column="post_rank")
    return {
        "workflow_name": "constraint_aware_reranking",
        "request_count": request_count,
        "reranked_row_count": int(len(reranked)),
        "changed_request_count": int(changed_requests),
        "changed_request_rate": float(changed_requests / request_count) if request_count else 0.0,
        "average_absolute_rank_shift": float(reranked["rank_shift"].abs().mean()) if not reranked.empty else 0.0,
        "before_mean_reciprocal_rank": before_mrr,
        "after_mean_reciprocal_rank": after_mrr,
        "top1_topic_counts_before": top1_counts(reranked, "pre_rank", "topic"),
        "top1_topic_counts_after": top1_counts(reranked, "post_rank", "topic"),
        "top1_creator_counts_before": top1_counts(reranked, "pre_rank", "creator_id"),
        "top1_creator_counts_after": top1_counts(reranked, "post_rank", "creator_id"),
    }


def request_mrr(rows: pd.DataFrame, *, rank_column: str) -> float:
    reciprocal_ranks: list[float] = []
    for _, request_rows in rows.groupby("request_id"):
        positives = request_rows.loc[request_rows["label"] == 1, rank_column].tolist()
        if positives:
            reciprocal_ranks.append(1.0 / min(positives))
        else:
            reciprocal_ranks.append(0.0)
    return float(sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0


def top1_counts(rows: pd.DataFrame, rank_column: str, value_column: str) -> dict[str, int]:
    top_rows = rows.loc[rows[rank_column] == 1, value_column]
    return {str(key): int(value) for key, value in top_rows.value_counts().to_dict().items()}


def build_rerank_manifest(
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    ranker_dir: Path,
) -> dict[str, Any]:
    return {
        "workflow_name": "constraint_aware_reranking",
        "ranker_input_dir": str(ranker_dir),
        "rules": {
            "freshness_proxy": {
                "description": "Boost items that have not been seen recently in prior event-log impressions.",
                "weight": config["reranking"]["freshness_weight"],
            },
            "topic_diversity": {
                "description": "Penalize repeated topics within the reranked request list.",
                "penalty": config["reranking"]["topic_repeat_penalty"],
            },
            "creator_spread": {
                "description": "Penalize repeated creator exposure within the reranked request list using creator identifiers from item_state.",
                "penalty": config["reranking"]["creator_repeat_penalty"],
            },
        },
        "assumptions": [
            "Published timestamps are not available in the smoke fixture, so freshness is approximated by time since prior item exposure in the event log.",
            "Creator spread uses creator identifiers derived into item_state, with config-backed fallback only for compatibility.",
            "Reranking is applied greedily within each request so rule effects remain inspectable.",
        ],
        "metrics_snapshot": metrics,
    }
