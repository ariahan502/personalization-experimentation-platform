from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from personalization_platform.delivery.features import (
    build_serving_feature_state,
    hydrate_request_time_features,
)
from personalization_platform.delivery.request_time import assemble_request_time_candidates
from personalization_platform.experiments.assignment import (
    choose_treatment,
    compute_hash_bucket,
    validate_experiment_config,
)
from personalization_platform.features.contracts import build_training_serving_feature_contract
from personalization_platform.retrieval.common import load_event_log_inputs


class CandidateItemInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(..., description="Candidate item identifier to score.")
    topic: str | None = Field(default=None, description="Optional topic override for unseen items.")
    creator_id: str | None = Field(default=None, description="Optional creator override for unseen items.")
    publisher: str | None = Field(default=None, description="Optional publisher override for unseen items.")
    title: str | None = Field(default=None, description="Optional title override for unseen items.")
    candidate_source: str | None = Field(default="request_payload", description="Optional upstream source label.")


class ScoreFeedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    experiment_id: str | None = None
    assignment_unit: str | None = None
    assignment_unit_id: str | None = None
    hash_bucket: float | None = None
    treatment_id: str | None = None
    treatment_name: str | None = None
    is_control: int | None = None
    dataset_split: str
    source_rerank_dir: str
    available_item_count: int
    returned_item_count: int
    items: list[RankedItem]
    degraded_modes: list[str]
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
    experiment_context = build_serving_experiment_context(config)
    feature_contract = build_training_serving_feature_contract()
    health_contract = build_api_health_contract(
        rerank_dir=rerank_dir,
        reranked_rows=reranked_rows,
        request_index=request_index,
        contextual_state=contextual_state,
        feature_contract=feature_contract,
        retrieval_config=config.get("request_time_retrieval", {}),
        experiment_context=experiment_context,
    )

    app = FastAPI(
        title=api_config.get("title", "Personalization Local API"),
        description="Local artifact-backed ranked feed demo API.",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "overall_status": health_contract["overall_status"],
            "api_name": api_config.get("api_name", "local_ranked_feed_api"),
            "supported_modes": ["fixture_replay", "contextual_scoring", "request_time_assembly"],
            "source_rerank_dir": str(rerank_dir),
            "request_count": len(request_index),
            "contextual_scoring_enabled": contextual_state is not None,
            "health_checks": health_contract["checks"],
            "degraded_modes": health_contract["degraded_modes"],
            "serving_state": health_contract["serving_state"],
            "feature_contract": health_contract["feature_contract"],
        }

    @app.post("/score/feed", response_model=ScoreFeedResponse)
    def score_feed(payload: ScoreFeedRequest) -> ScoreFeedResponse:
        if should_use_replay_mode(payload=payload, request_index=request_index):
            return build_replay_response(
                payload=payload,
                request_index=request_index,
                rerank_dir=rerank_dir,
                api_name=api_config.get("api_name", "local_ranked_feed_api"),
                experiment_context=experiment_context,
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
            retrieval_config=config.get("request_time_retrieval", {}),
            experiment_context=experiment_context,
        )

    return app


def build_replay_response(
    *,
    payload: ScoreFeedRequest,
    request_index: dict[str, pd.DataFrame],
    rerank_dir: Path,
    api_name: str,
    experiment_context: dict[str, Any] | None,
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
    treatment_context = assign_serving_treatment(
        request_id=str(payload.request_id),
        user_id=request_user_id,
        experiment_context=experiment_context,
    )
    return ScoreFeedResponse(
        api_name=api_name,
        mode="fixture_replay",
        request_id=str(payload.request_id),
        user_id=request_user_id,
        experiment_id=treatment_context.get("experiment_id"),
        assignment_unit=treatment_context.get("assignment_unit"),
        assignment_unit_id=treatment_context.get("assignment_unit_id"),
        hash_bucket=treatment_context.get("hash_bucket"),
        treatment_id=treatment_context.get("treatment_id"),
        treatment_name=treatment_context.get("treatment_name"),
        is_control=treatment_context.get("is_control"),
        dataset_split=str(request_rows.iloc[0]["dataset_split"]),
        source_rerank_dir=str(rerank_dir),
        available_item_count=int(len(request_rows)),
        returned_item_count=int(len(items)),
        items=items,
        degraded_modes=[],
        assumptions=assumptions,
    )


def build_contextual_response(
    *,
    payload: ScoreFeedRequest,
    contextual_state: dict[str, Any],
    rerank_dir: Path,
    api_name: str,
    api_config: dict[str, Any],
    retrieval_config: dict[str, Any],
    experiment_context: dict[str, Any] | None,
) -> ScoreFeedResponse:
    effective_user_id = payload.user_id or "anonymous_user"
    history_context = resolve_history_context(payload=payload, contextual_state=contextual_state)
    request_identifier = payload.request_id or f"contextual-{effective_user_id}"
    assembly_diagnostics = {"degraded_modes": [], "source_summaries": [], "fallback_used": False}
    if payload.candidate_items:
        candidate_rows = build_candidate_payload_rows(payload=payload, contextual_state=contextual_state)
        response_mode = "contextual_scoring"
        assumptions = [
            "This mode performs a simplified online-like rescore from caller-provided candidates plus local event-log priors.",
            "If request history is omitted, the API falls back to the latest known local user_state snapshot for that user when available.",
            "Contextual scoring uses explicit request-time feature hydration from local state, plus prior serving logs when available, rather than a full online feature store.",
        ]
    else:
        candidate_rows, assembly_diagnostics = assemble_request_time_candidates(
            request_id=request_identifier,
            user_id=effective_user_id,
            history_context=history_context,
            contextual_state=contextual_state,
            retrieval_config=resolve_request_time_retrieval_config(retrieval_config, top_k=int(payload.top_k)),
        )
        response_mode = "request_time_assembly"
        assumptions = [
            "This mode assembles request-time candidates from local affinity, content, and trending retrieval heuristics before scoring.",
            "Request-time assembly is still backed by local event-log state and does not represent a distributed online retrieval system.",
            "The assembled candidate surface preserves source provenance and hydrates fresh serving-style features, but still only approximates a future production retrieval layer.",
        ]
    candidate_rows = hydrate_candidate_rows(candidate_rows=candidate_rows, contextual_state=contextual_state)
    if candidate_rows.empty:
        raise HTTPException(status_code=400, detail="No request-time candidates could be resolved from the request.")

    scoring_weights = api_config.get("scoring_weights", {})
    request_time = contextual_state["serving_request_time"]
    candidate_rows, feature_state_summary = hydrate_request_time_features(
        candidate_rows=candidate_rows,
        history_context=history_context,
        contextual_state=contextual_state,
        serving_feature_state=contextual_state["serving_feature_state"],
        scoring_weights=scoring_weights,
        user_id=effective_user_id,
        request_time=request_time,
    )
    assumptions.extend(feature_state_summary["fallbacks"])
    if assembly_diagnostics["degraded_modes"]:
        assumptions.append(
            "Request-time assembly degraded modes: " + ", ".join(assembly_diagnostics["degraded_modes"]) + "."
        )
    topic_repeat_penalty = float(scoring_weights.get("topic_repeat_penalty", 0.2))
    creator_repeat_penalty = float(scoring_weights.get("creator_repeat_penalty", 0.15))

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
                    "recent_item_ctr": float(row["recent_item_ctr"]),
                    "recent_topic_click_share": float(row["recent_topic_click_share"]),
                    "recent_user_click_rate": float(row["recent_user_click_rate"]),
                    "seen_history_penalty": float(row["seen_history_penalty"]),
                    "freshness_bonus": float(row["freshness_bonus"]),
                    "diversity_penalty": float(row["diversity_penalty"]),
                    "creator_penalty": float(row["creator_penalty"]),
                },
            )
        )

    treatment_context = assign_serving_treatment(
        request_id=request_identifier,
        user_id=effective_user_id,
        experiment_context=experiment_context,
    )
    return ScoreFeedResponse(
        api_name=api_name,
        mode=response_mode,
        request_id=request_identifier,
        user_id=effective_user_id,
        experiment_id=treatment_context.get("experiment_id"),
        assignment_unit=treatment_context.get("assignment_unit"),
        assignment_unit_id=treatment_context.get("assignment_unit_id"),
        hash_bucket=treatment_context.get("hash_bucket"),
        treatment_id=treatment_context.get("treatment_id"),
        treatment_name=treatment_context.get("treatment_name"),
        is_control=treatment_context.get("is_control"),
        dataset_split="contextual_request",
        source_rerank_dir=str(rerank_dir),
        available_item_count=int(len(candidate_rows)),
        returned_item_count=int(len(items)),
        items=items,
        degraded_modes=assembly_diagnostics["degraded_modes"],
        assumptions=assumptions,
    )


def build_request_index(reranked_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(request_id): frame.copy()
        for request_id, frame in reranked_rows.groupby("request_id", sort=False)
    }


def build_api_health_contract(
    *,
    rerank_dir: Path,
    reranked_rows: pd.DataFrame,
    request_index: dict[str, pd.DataFrame],
    contextual_state: dict[str, Any] | None,
    feature_contract: dict[str, Any],
    retrieval_config: dict[str, Any],
    experiment_context: dict[str, Any] | None,
) -> dict[str, Any]:
    degraded_modes: list[str] = []
    if contextual_state is None:
        degraded_modes.append("contextual_scoring_unavailable")
        degraded_modes.append("request_time_assembly_unavailable")
    checks = [
        build_health_check(
            name="rerank_rows_present",
            passed=not reranked_rows.empty,
            description="Local reranked rows are loaded and available for replay mode.",
        ),
        build_health_check(
            name="replay_requests_present",
            passed=bool(request_index),
            description="At least one replayable request exists in the latest rerank bundle.",
        ),
        build_health_check(
            name="contextual_state_available",
            passed=contextual_state is not None,
            description="Contextual scoring inputs are available from local event-log state.",
        ),
        build_health_check(
            name="request_time_assembly_available",
            passed=contextual_state is not None,
            description="Request-time candidate assembly can run against local retrieval state.",
        ),
    ]
    return {
        "overall_status": "warn" if degraded_modes else "pass",
        "checks": checks,
        "degraded_modes": degraded_modes,
        "serving_state": {
            "source_rerank_dir": str(rerank_dir),
            "rerank_row_count": int(len(reranked_rows)),
            "replay_request_count": int(len(request_index)),
            "contextual_scoring_enabled": contextual_state is not None,
            "request_time_assembly_enabled": contextual_state is not None,
            "request_time_controls": {
                "max_sources_per_request": retrieval_config.get("max_sources_per_request"),
                "fallback_to_trending_only": bool(retrieval_config.get("fallback_to_trending_only", True)),
                "configured_source_count": len(retrieval_config.get("sources", [])),
            },
            "experiment_assignment_enabled": experiment_context is not None,
            "experiment_id": experiment_context.get("experiment_id") if experiment_context is not None else None,
            "assignment_unit": experiment_context.get("assignment_unit") if experiment_context is not None else None,
            "serving_log_features_available": (
                bool(contextual_state["serving_feature_state"]["logs_available"]) if contextual_state is not None else False
            ),
            "serving_log_source_run_dir": (
                contextual_state["serving_feature_state"]["source_run_dir"] if contextual_state is not None else None
            ),
            "known_user_count": int(len(contextual_state["latest_user_state"])) if contextual_state is not None else 0,
            "known_item_count": int(len(contextual_state["item_metadata_lookup"])) if contextual_state is not None else 0,
        },
        "feature_contract": feature_contract,
    }


def should_use_replay_mode(*, payload: ScoreFeedRequest, request_index: dict[str, pd.DataFrame]) -> bool:
    return (
        payload.request_id is not None
        and payload.request_id in request_index
        and not payload.candidate_items
        and payload.history_item_ids is None
        and payload.history_topics is None
    )


def resolve_request_time_retrieval_config(retrieval_config: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    if retrieval_config:
        return retrieval_config
    candidate_count = max(top_k * 2, top_k)
    return {
        "candidate_count": candidate_count,
        "sources": [
            {"name": "affinity", "candidate_count": candidate_count, "priority": 1},
            {"name": "content", "candidate_count": candidate_count, "priority": 2},
            {"name": "trending", "candidate_count": candidate_count, "priority": 3},
        ],
    }


def build_health_check(*, name: str, passed: bool, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "warn",
        "description": description,
    }


def build_serving_experiment_context(config: dict[str, Any]) -> dict[str, Any] | None:
    experiment = config.get("experiment")
    if experiment is None:
        return None
    return validate_experiment_config(config)


def assign_serving_treatment(
    *,
    request_id: str,
    user_id: str,
    experiment_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if experiment_context is None:
        return {}
    assignment_unit = str(experiment_context["assignment_unit"])
    assignment_unit_id = str(user_id if assignment_unit == "user_id" else request_id)
    hash_bucket = compute_hash_bucket(
        experiment_id=experiment_context["experiment_id"],
        salt=experiment_context["salt"],
        assignment_unit_id=assignment_unit_id,
    )
    treatment_id = choose_treatment(bucket=hash_bucket, treatments=experiment_context["treatments"])
    treatment_lookup = {
        treatment["treatment_id"]: treatment for treatment in experiment_context["treatments"]
    }
    treatment = treatment_lookup[treatment_id]
    return {
        "experiment_id": experiment_context["experiment_id"],
        "assignment_unit": assignment_unit,
        "assignment_unit_id": assignment_unit_id,
        "hash_bucket": hash_bucket,
        "treatment_id": treatment_id,
        "treatment_name": treatment["treatment_name"],
        "is_control": int(bool(treatment["is_control"])),
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
        "serving_feature_state": build_serving_feature_state(
            config=config,
            contextual_state={
                "impressions": impressions,
                "item_metadata_lookup": item_metadata_lookup,
            },
        ),
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


def hydrate_candidate_rows(*, candidate_rows: pd.DataFrame, contextual_state: dict[str, Any]) -> pd.DataFrame:
    if candidate_rows.empty:
        return candidate_rows
    hydrated = candidate_rows.copy()
    metadata_lookup = contextual_state["item_metadata_lookup"]
    for column, default_value in [
        ("topic", "unknown_topic"),
        ("creator_id", "creator_unknown"),
        ("publisher", "publisher_unknown"),
        ("title", ""),
    ]:
        if column not in hydrated.columns:
            hydrated[column] = hydrated["item_id"].map(
                lambda item_id: metadata_lookup.get(str(item_id), {}).get(column, default_value)
            )
        else:
            hydrated[column] = hydrated.apply(
                lambda row: row[column]
                if pd.notna(row[column]) and str(row[column]) != ""
                else metadata_lookup.get(str(row["item_id"]), {}).get(column, default_value),
                axis=1,
            )
    if "item_impression_priors" not in hydrated.columns:
        hydrated["item_impression_priors"] = hydrated["item_id"].map(
            lambda item_id: contextual_state["item_impression_priors"].get(str(item_id), 0.0)
        )
    return hydrated


def resolve_run_dir(*, base_dir: str, run_name: str) -> Path:
    matches = sorted(Path(base_dir).glob(f"*_{run_name}"))
    if not matches:
        raise FileNotFoundError(f"No outputs found under {base_dir} matching '*_{run_name}'.")
    return matches[-1]
