from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from personalization_platform.reranking.policy import compute_freshness_minutes, freshness_bonus
from personalization_platform.retrieval.common import load_event_log_inputs


class CandidateItemInput(BaseModel):
    item_id: str = Field(..., description="Candidate item identifier to score.")
    topic: str | None = Field(default=None, description="Optional topic override for unseen items.")
    creator_id: str | None = Field(default=None, description="Optional creator override for unseen items.")
    publisher: str | None = Field(default=None, description="Optional publisher override for unseen items.")
    title: str | None = Field(default=None, description="Optional title override for unseen items.")
    candidate_source: str | None = Field(default="request_payload", description="Optional upstream source label.")


class ScoreFeedRequest(BaseModel):
    request_id: str | None = Field(
        default=None,
        description="Fixture request identifier for replay mode.",
    )
    top_k: int = Field(default=10, ge=1, le=50, description="Maximum number of ranked items to return.")
    user_id: str | None = Field(default=None, description="User identifier for replay validation or contextual scoring.")
    history_item_ids: list[str] | None = Field(
        default=None,
        description="Optional request-time history items for contextual scoring.",
    )
    history_topics: list[str] | None = Field(
        default=None,
        description="Optional request-time history topics for contextual scoring.",
    )
    candidate_items: list[CandidateItemInput] | None = Field(
        default=None,
        description="Optional candidate set for contextual scoring.",
    )


class RankedItem(BaseModel):
    item_id: str
    topic: str
    creator_id: str
    candidate_source: str
    pre_rank: int
    post_rank: int
    rank_shift: int
    prediction: float
    rerank_score: float
    freshness_bonus: float
    score_components: dict[str, float]


class ScoreFeedResponse(BaseModel):
    api_name: str
    mode: str
    request_id: str
    user_id: str
    dataset_split: str
    source_rerank_dir: str
    available_item_count: int
    returned_item_count: int
    items: list[RankedItem]
    assumptions: list[str]


def create_local_api_app(config: dict[str, Any]) -> FastAPI:
    rerank_dir = resolve_run_dir(
        base_dir=config["input"]["rerank_base_dir"],
        run_name=config["input"]["rerank_run_name"],
    )
    reranked_rows = pd.read_csv(rerank_dir / "reranked_rows.csv")
    api_config = config["api"]
    request_index = build_request_index(reranked_rows)
    contextual_state = build_contextual_state(config)

    app = FastAPI(
        title=api_config.get("title", "Personalization Local API"),
        description="Local artifact-backed ranked feed demo API.",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "api_name": api_config.get("api_name", "local_ranked_feed_api"),
            "supported_modes": ["fixture_replay", "contextual_scoring"],
            "source_rerank_dir": str(rerank_dir),
            "request_count": len(request_index),
            "contextual_scoring_enabled": contextual_state is not None,
        }

    @app.post("/score/feed", response_model=ScoreFeedResponse)
    def score_feed(payload: ScoreFeedRequest) -> ScoreFeedResponse:
        if payload.request_id is not None and not payload.candidate_items:
            return build_replay_response(
                payload=payload,
                request_index=request_index,
                rerank_dir=rerank_dir,
                api_name=api_config.get("api_name", "local_ranked_feed_api"),
            )

        if not payload.candidate_items:
            raise HTTPException(
                status_code=400,
                detail="Contextual scoring requires candidate_items when request_id replay is not used.",
            )
        if contextual_state is None:
            raise HTTPException(
                status_code=400,
                detail="Contextual scoring is unavailable because event-log inputs were not configured for the API.",
            )

        return build_contextual_response(
            payload=payload,
            contextual_state=contextual_state,
            rerank_dir=rerank_dir,
            api_name=api_config.get("api_name", "local_ranked_feed_api"),
            api_config=api_config,
        )

    return app


def build_replay_response(
    *,
    payload: ScoreFeedRequest,
    request_index: dict[str, pd.DataFrame],
    rerank_dir: Path,
    api_name: str,
) -> ScoreFeedResponse:
    if payload.request_id is None or payload.request_id not in request_index:
        raise HTTPException(status_code=404, detail=f"Unknown request_id '{payload.request_id}'.")

    request_rows = request_index[payload.request_id].copy()
    request_user_id = str(request_rows.iloc[0]["user_id"])
    if payload.user_id is not None and payload.user_id != request_user_id:
        raise HTTPException(
            status_code=400,
            detail=f"Provided user_id '{payload.user_id}' does not match request owner '{request_user_id}'.",
        )

    ranked = request_rows.sort_values(["post_rank", "item_id"], ascending=[True, True]).head(payload.top_k)
    items = [
        RankedItem(
            item_id=str(row["item_id"]),
            topic=str(row["topic"]),
            creator_id=str(row["creator_id"]),
            candidate_source=str(row["candidate_source"]),
            pre_rank=int(row["pre_rank"]),
            post_rank=int(row["post_rank"]),
            rank_shift=int(row["rank_shift"]),
            prediction=float(row["prediction"]),
            rerank_score=float(row["rerank_score"]),
            freshness_bonus=float(row["freshness_bonus"]),
            score_components={
                "prediction": float(row["prediction"]),
                "freshness_bonus": float(row["freshness_bonus"]),
                "diversity_penalty": float(row.get("diversity_penalty", 0.0)),
                "creator_penalty": float(row.get("creator_penalty", 0.0)),
            },
        )
        for row in ranked.to_dict(orient="records")
    ]
    assumptions = [
        "This mode replays the latest local reranked outputs for a known fixture request identifier.",
        "Returned ranks and scores are intended for local demo and integration validation only.",
        "Replay mode does not perform live candidate generation or feature retrieval.",
    ]
    return ScoreFeedResponse(
        api_name=api_name,
        mode="fixture_replay",
        request_id=str(payload.request_id),
        user_id=request_user_id,
        dataset_split=str(request_rows.iloc[0]["dataset_split"]),
        source_rerank_dir=str(rerank_dir),
        available_item_count=int(len(request_rows)),
        returned_item_count=int(len(items)),
        items=items,
        assumptions=assumptions,
    )


def build_contextual_response(
    *,
    payload: ScoreFeedRequest,
    contextual_state: dict[str, Any],
    rerank_dir: Path,
    api_name: str,
    api_config: dict[str, Any],
) -> ScoreFeedResponse:
    effective_user_id = payload.user_id or "anonymous_user"
    history_context = resolve_history_context(payload=payload, contextual_state=contextual_state)
    candidate_rows = build_candidate_payload_rows(payload=payload, contextual_state=contextual_state)
    if candidate_rows.empty:
        raise HTTPException(status_code=400, detail="No contextual candidates could be resolved from the request.")

    scoring_weights = api_config.get("scoring_weights", {})
    request_time = contextual_state["serving_request_time"]
    candidate_rows["pre_rank"] = candidate_rows.index + 1
    candidate_rows["topic_affinity"] = candidate_rows["topic"].map(
        lambda topic: float(history_context["topic_counts"].get(str(topic), 0))
    )
    max_topic_count = max(history_context["topic_counts"].values(), default=0)
    candidate_rows["normalized_topic_affinity"] = candidate_rows["topic_affinity"].map(
        lambda value: value / max(max_topic_count, 1)
    )
    candidate_rows["seen_history_penalty"] = candidate_rows["item_id"].map(
        lambda item_id: float(item_id in history_context["history_item_ids"])
    )
    candidate_rows["click_prior"] = candidate_rows["item_id"].map(
        lambda item_id: contextual_state["item_click_priors"].get(item_id, 0.0)
    )
    candidate_rows["impression_prior"] = candidate_rows["item_impression_priors"].map(
        lambda value: float(value)
    ) if "item_impression_priors" in candidate_rows.columns else candidate_rows["item_id"].map(
        lambda item_id: contextual_state["item_impression_priors"].get(item_id, 0.0)
    )
    candidate_rows["freshness_minutes_since_last_seen"] = candidate_rows["item_id"].map(
        lambda item_id: compute_freshness_minutes(
            impressions=contextual_state["impressions"],
            item_id=item_id,
            request_time=request_time,
        )
    )
    freshness_weight = float(scoring_weights.get("freshness_weight", 0.35))
    topic_weight = float(scoring_weights.get("topic_affinity_weight", 1.0))
    click_prior_weight = float(scoring_weights.get("click_prior_weight", 0.5))
    impression_prior_weight = float(scoring_weights.get("impression_prior_weight", 0.2))
    history_penalty_weight = float(scoring_weights.get("history_seen_penalty", 0.75))
    topic_repeat_penalty = float(scoring_weights.get("topic_repeat_penalty", 0.2))
    creator_repeat_penalty = float(scoring_weights.get("creator_repeat_penalty", 0.15))

    candidate_rows["freshness_bonus"] = candidate_rows["freshness_minutes_since_last_seen"].map(
        lambda value: freshness_bonus(value=value, weight=freshness_weight)
    )
    candidate_rows["prediction"] = (
        topic_weight * candidate_rows["normalized_topic_affinity"].astype(float)
        + click_prior_weight * candidate_rows["click_prior"].astype(float)
        + impression_prior_weight * candidate_rows["impression_prior"].astype(float)
        - history_penalty_weight * candidate_rows["seen_history_penalty"].astype(float)
    )

    selected_rows: list[dict[str, Any]] = []
    used_topics: set[str] = set()
    used_creators: set[str] = set()
    available = candidate_rows.copy()
    while not available.empty and len(selected_rows) < int(payload.top_k):
        available = available.copy()
        available["diversity_penalty"] = available["topic"].map(
            lambda topic: topic_repeat_penalty if str(topic) in used_topics else 0.0
        )
        available["creator_penalty"] = available["creator_id"].map(
            lambda creator_id: creator_repeat_penalty if str(creator_id) in used_creators else 0.0
        )
        available["rerank_score"] = (
            available["prediction"].astype(float)
            + available["freshness_bonus"].astype(float)
            - available["diversity_penalty"].astype(float)
            - available["creator_penalty"].astype(float)
        )
        next_row = (
            available.sort_values(
                ["rerank_score", "prediction", "pre_rank", "item_id"],
                ascending=[False, False, True, True],
            )
            .iloc[0]
            .to_dict()
        )
        selected_rows.append(next_row)
        used_topics.add(str(next_row["topic"]))
        used_creators.add(str(next_row["creator_id"]))
        available = available.loc[available["item_id"] != next_row["item_id"]]

    items: list[RankedItem] = []
    for post_rank, row in enumerate(selected_rows, start=1):
        items.append(
            RankedItem(
                item_id=str(row["item_id"]),
                topic=str(row["topic"]),
                creator_id=str(row["creator_id"]),
                candidate_source=str(row["candidate_source"]),
                pre_rank=int(row["pre_rank"]),
                post_rank=post_rank,
                rank_shift=int(row["pre_rank"]) - post_rank,
                prediction=float(row["prediction"]),
                rerank_score=float(row["rerank_score"]),
                freshness_bonus=float(row["freshness_bonus"]),
                score_components={
                    "topic_affinity": float(row["normalized_topic_affinity"]),
                    "click_prior": float(row["click_prior"]),
                    "impression_prior": float(row["impression_prior"]),
                    "seen_history_penalty": float(row["seen_history_penalty"]),
                    "freshness_bonus": float(row["freshness_bonus"]),
                    "diversity_penalty": float(row["diversity_penalty"]),
                    "creator_penalty": float(row["creator_penalty"]),
                },
            )
        )

    assumptions = [
        "This mode performs a simplified online-like rescore from request payload candidates plus local event-log priors.",
        "If request history is omitted, the API falls back to the latest known local user_state snapshot for that user when available.",
        "Contextual scoring uses local item metadata, historical topic counts, item popularity priors, and rerank-style diversity penalties rather than a live feature service.",
    ]
    request_identifier = payload.request_id or f"contextual-{effective_user_id}"
    return ScoreFeedResponse(
        api_name=api_name,
        mode="contextual_scoring",
        request_id=request_identifier,
        user_id=effective_user_id,
        dataset_split="contextual_request",
        source_rerank_dir=str(rerank_dir),
        available_item_count=int(len(candidate_rows)),
        returned_item_count=int(len(items)),
        items=items,
        assumptions=assumptions,
    )


def build_request_index(reranked_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(request_id): frame.copy()
        for request_id, frame in reranked_rows.groupby("request_id", sort=False)
    }


def build_contextual_state(config: dict[str, Any]) -> dict[str, Any] | None:
    input_config = config.get("input", {})
    if "event_log_base_dir" not in input_config or "event_log_run_name" not in input_config:
        return None

    event_log_inputs = load_event_log_inputs(config)
    requests = event_log_inputs["requests"].copy()
    user_state = event_log_inputs["user_state"].copy()
    item_state = event_log_inputs["item_state"].copy()
    impressions = event_log_inputs["impressions"].copy()

    latest_user_state = build_latest_user_state_lookup(requests=requests, user_state=user_state)
    item_metadata_lookup = {
        str(row.item_id): {
            "item_id": str(row.item_id),
            "topic": str(row.topic),
            "creator_id": str(row.creator_id),
            "publisher": str(row.publisher),
            "title": str(row.title),
        }
        for row in item_state.itertuples(index=False)
    }
    click_counts = (
        impressions.loc[impressions["clicked"] == 1, "item_id"].value_counts().to_dict()
    )
    impression_counts = impressions["item_id"].value_counts().to_dict()
    max_click_count = max(click_counts.values(), default=0)
    max_impression_count = max(impression_counts.values(), default=0)

    return {
        "requests": requests,
        "impressions": impressions,
        "latest_user_state": latest_user_state,
        "item_metadata_lookup": item_metadata_lookup,
        "item_click_priors": {
            str(item_id): count / max(max_click_count, 1) for item_id, count in click_counts.items()
        },
        "item_impression_priors": {
            str(item_id): count / max(max_impression_count, 1) for item_id, count in impression_counts.items()
        },
        "serving_request_time": requests["request_ts"].max() + pd.Timedelta(minutes=5),
        "event_log_dir": str(event_log_inputs["event_log_dir"]),
    }


def build_latest_user_state_lookup(*, requests: pd.DataFrame, user_state: pd.DataFrame) -> dict[str, dict[str, Any]]:
    request_times = requests[["request_id", "request_ts"]].rename(columns={"request_ts": "request_ts_lookup"})
    prepared = user_state.merge(request_times, on="request_id", how="left").sort_values(
        ["user_id", "request_ts_lookup", "request_id"]
    )
    latest_lookup: dict[str, dict[str, Any]] = {}
    for row in prepared.itertuples(index=False):
        latest_lookup[str(row.user_id)] = {
            "history_item_ids": set(json.loads(row.history_item_ids)),
            "topic_counts": {str(key): int(value) for key, value in json.loads(row.recent_topic_counts).items()},
            "is_cold_start": bool(row.is_cold_start),
        }
    return latest_lookup


def resolve_history_context(
    *,
    payload: ScoreFeedRequest,
    contextual_state: dict[str, Any],
) -> dict[str, Any]:
    latest_known = contextual_state["latest_user_state"].get(str(payload.user_id), {})
    history_item_ids = set(payload.history_item_ids or latest_known.get("history_item_ids", set()))
    if payload.history_topics is not None:
        topic_counts: dict[str, int] = {}
        for topic in payload.history_topics:
            topic_counts[str(topic)] = topic_counts.get(str(topic), 0) + 1
    else:
        topic_counts = dict(latest_known.get("topic_counts", {}))
    return {
        "history_item_ids": history_item_ids,
        "topic_counts": topic_counts,
    }


def build_candidate_payload_rows(
    *,
    payload: ScoreFeedRequest,
    contextual_state: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in payload.candidate_items or []:
        metadata = dict(contextual_state["item_metadata_lookup"].get(candidate.item_id, {}))
        resolved = {
            "item_id": candidate.item_id,
            "topic": candidate.topic or metadata.get("topic", "unknown_topic"),
            "creator_id": candidate.creator_id or metadata.get("creator_id", "creator_unknown"),
            "publisher": candidate.publisher or metadata.get("publisher", "publisher_unknown"),
            "title": candidate.title or metadata.get("title", ""),
            "candidate_source": candidate.candidate_source or "request_payload",
            "item_impression_priors": contextual_state["item_impression_priors"].get(candidate.item_id, 0.0),
        }
        rows.append(resolved)
    return pd.DataFrame(rows)


def resolve_run_dir(*, base_dir: str, run_name: str) -> Path:
    matches = sorted(Path(base_dir).glob(f"*_{run_name}"))
    if not matches:
        raise FileNotFoundError(f"No outputs found under {base_dir} matching '*_{run_name}'.")
    return matches[-1]
