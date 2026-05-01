from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from personalization_platform.pipeline.build_candidates import merge_candidates
from personalization_platform.retrieval.content import max_title_overlap, tokenize


def assemble_request_time_candidates(
    *,
    request_id: str,
    user_id: str,
    history_context: dict[str, Any],
    contextual_state: dict[str, Any],
    retrieval_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    diagnostics = {
        "degraded_modes": [],
        "source_summaries": [],
        "fallback_used": False,
    }
    configured_sources = list(retrieval_config.get("sources", []))
    max_sources_per_request = int(retrieval_config.get("max_sources_per_request", len(configured_sources) or 1))
    source_configs = configured_sources[:max_sources_per_request]
    if len(source_configs) < len(configured_sources):
        diagnostics["degraded_modes"].append("source_budget_truncated")

    source_frames: list[pd.DataFrame] = []
    for source_config in source_configs:
        source_name = str(source_config["name"])
        candidate_count = int(source_config["candidate_count"])
        try:
            if source_name == "affinity":
                frame = build_request_time_affinity_candidates(
                    request_id=request_id,
                    user_id=user_id,
                    history_context=history_context,
                    contextual_state=contextual_state,
                    candidate_count=candidate_count,
                )
            elif source_name == "content":
                frame = build_request_time_content_candidates(
                    request_id=request_id,
                    user_id=user_id,
                    history_context=history_context,
                    contextual_state=contextual_state,
                    candidate_count=candidate_count,
                )
            elif source_name == "trending":
                frame = build_request_time_trending_candidates(
                    request_id=request_id,
                    user_id=user_id,
                    history_context=history_context,
                    contextual_state=contextual_state,
                    candidate_count=candidate_count,
                )
            else:
                raise ValueError(f"Unsupported request-time retrieval source '{source_name}'.")
        except Exception as exc:
            diagnostics["degraded_modes"].append(f"{source_name}_source_failed")
            diagnostics["source_summaries"].append(
                {"source": source_name, "candidate_rows": 0, "status": "failed", "reason": str(exc)}
            )
            continue
        if frame.empty:
            diagnostics["source_summaries"].append(
                {"source": source_name, "candidate_rows": 0, "status": "empty"}
            )
            continue
        frame = frame.copy()
        frame["source_priority"] = int(source_config["priority"])
        source_frames.append(frame)
        diagnostics["source_summaries"].append(
            {"source": source_name, "candidate_rows": int(len(frame)), "status": "used"}
        )

    merged = merge_candidates(
        source_frames=source_frames,
        final_candidate_count=int(retrieval_config["candidate_count"]),
    )
    if not merged.empty:
        return merged, diagnostics

    if retrieval_config.get("fallback_to_trending_only", True):
        trending_count = int(retrieval_config.get("trending_fallback_candidate_count", retrieval_config["candidate_count"]))
        fallback_frame = build_request_time_trending_candidates(
            request_id=request_id,
            user_id=user_id,
            history_context=history_context,
            contextual_state=contextual_state,
            candidate_count=trending_count,
        )
        if not fallback_frame.empty:
            fallback_frame = fallback_frame.copy()
            fallback_frame["source_priority"] = 999
            diagnostics["degraded_modes"].append("trending_only_fallback")
            diagnostics["fallback_used"] = True
            diagnostics["source_summaries"].append(
                {"source": "trending_fallback", "candidate_rows": int(len(fallback_frame)), "status": "fallback_used"}
            )
            merged = merge_candidates(
                source_frames=[fallback_frame],
                final_candidate_count=int(retrieval_config["candidate_count"]),
            )
            return merged, diagnostics

    diagnostics["degraded_modes"].append("empty_candidate_set")
    return merged, diagnostics


def build_request_time_affinity_candidates(
    *,
    request_id: str,
    user_id: str,
    history_context: dict[str, Any],
    contextual_state: dict[str, Any],
    candidate_count: int,
) -> pd.DataFrame:
    topic_counts = history_context["topic_counts"]
    history_items = history_context["history_item_ids"]
    if not topic_counts and not history_items:
        return pd.DataFrame()

    item_metadata_lookup = contextual_state["item_metadata_lookup"]
    co_impression_counts = build_co_impression_counts(
        history_items=history_items,
        impressions=contextual_state["impressions"],
    )
    ranking_rows: list[dict[str, Any]] = []
    for item_id, metadata in item_metadata_lookup.items():
        if item_id in history_items:
            continue
        topic_affinity = int(topic_counts.get(str(metadata.get("topic", "")), 0))
        co_impression_affinity = int(co_impression_counts.get(item_id, 0))
        if topic_affinity <= 0 and co_impression_affinity <= 0:
            continue
        click_prior = float(contextual_state["item_click_priors"].get(item_id, 0.0))
        impression_prior = float(contextual_state["item_impression_priors"].get(item_id, 0.0))
        affinity_score = float(topic_affinity * 1000 + co_impression_affinity * 100 + click_prior * 10 + impression_prior)
        ranking_rows.append(
            {
                "request_id": request_id,
                "user_id": user_id,
                "item_id": item_id,
                "candidate_source": "affinity",
                "source_score": affinity_score,
                "topic": str(metadata.get("topic", "unknown_topic")),
                "_topic_affinity_count": topic_affinity,
                "_co_impression_affinity_count": co_impression_affinity,
            }
        )
    if not ranking_rows:
        return pd.DataFrame()
    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["source_score", "_topic_affinity_count", "_co_impression_affinity_count", "item_id"],
        ascending=[False, False, False, True],
    ).head(candidate_count).reset_index(drop=True)
    ranking["source_rank"] = ranking.index + 1
    return ranking[
        ["request_id", "user_id", "item_id", "candidate_source", "source_rank", "source_score", "topic"]
    ]


def build_request_time_content_candidates(
    *,
    request_id: str,
    user_id: str,
    history_context: dict[str, Any],
    contextual_state: dict[str, Any],
    candidate_count: int,
) -> pd.DataFrame:
    history_items = history_context["history_item_ids"]
    if not history_items:
        return pd.DataFrame()

    item_metadata_lookup = contextual_state["item_metadata_lookup"]
    topic_counts = Counter(history_context["topic_counts"])
    history_publishers = Counter()
    history_creators = Counter()
    history_title_tokens: dict[str, set[str]] = {}
    for item_id in history_items:
        metadata = item_metadata_lookup.get(item_id)
        if not metadata:
            continue
        if metadata.get("publisher"):
            history_publishers[str(metadata["publisher"])] += 1
        if metadata.get("creator_id"):
            history_creators[str(metadata["creator_id"])] += 1
        history_title_tokens[item_id] = tokenize(str(metadata.get("title", "")))

    ranking_rows: list[dict[str, Any]] = []
    for item_id, metadata in item_metadata_lookup.items():
        if item_id in history_items:
            continue
        topic_affinity = int(topic_counts.get(str(metadata.get("topic", "")), 0))
        publisher_affinity = int(history_publishers.get(str(metadata.get("publisher", "")), 0))
        creator_affinity = int(history_creators.get(str(metadata.get("creator_id", "")), 0))
        lexical_overlap = max_title_overlap(
            candidate_tokens=tokenize(str(metadata.get("title", ""))),
            history_title_tokens=history_title_tokens,
        )
        if topic_affinity <= 0 and publisher_affinity <= 0 and creator_affinity <= 0 and lexical_overlap <= 0:
            continue
        content_score = float(
            creator_affinity * 1000
            + publisher_affinity * 300
            + topic_affinity * 30
            + lexical_overlap
        )
        ranking_rows.append(
            {
                "request_id": request_id,
                "user_id": user_id,
                "item_id": item_id,
                "candidate_source": "content",
                "source_score": content_score,
                "topic": str(metadata.get("topic", "unknown_topic")),
                "_creator_affinity_count": creator_affinity,
                "_publisher_affinity_count": publisher_affinity,
                "_topic_affinity_count": topic_affinity,
            }
        )
    if not ranking_rows:
        return pd.DataFrame()
    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["source_score", "_creator_affinity_count", "_publisher_affinity_count", "_topic_affinity_count", "item_id"],
        ascending=[False, False, False, False, True],
    ).head(candidate_count).reset_index(drop=True)
    ranking["source_rank"] = ranking.index + 1
    return ranking[
        ["request_id", "user_id", "item_id", "candidate_source", "source_rank", "source_score", "topic"]
    ]


def build_request_time_trending_candidates(
    *,
    request_id: str,
    user_id: str,
    history_context: dict[str, Any],
    contextual_state: dict[str, Any],
    candidate_count: int,
) -> pd.DataFrame:
    history_items = history_context["history_item_ids"]
    ranking_rows: list[dict[str, Any]] = []
    for item_id, metadata in contextual_state["item_metadata_lookup"].items():
        if item_id in history_items:
            continue
        click_prior = float(contextual_state["item_click_priors"].get(item_id, 0.0))
        impression_prior = float(contextual_state["item_impression_priors"].get(item_id, 0.0))
        if click_prior <= 0.0 and impression_prior <= 0.0:
            continue
        ranking_rows.append(
            {
                "request_id": request_id,
                "user_id": user_id,
                "item_id": item_id,
                "candidate_source": "trending",
                "source_score": float(click_prior * 1000 + impression_prior),
                "topic": str(metadata.get("topic", "unknown_topic")),
                "_click_prior": click_prior,
                "_impression_prior": impression_prior,
            }
        )
    if not ranking_rows:
        return pd.DataFrame()
    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["source_score", "_click_prior", "_impression_prior", "item_id"],
        ascending=[False, False, False, True],
    ).head(candidate_count).reset_index(drop=True)
    ranking["source_rank"] = ranking.index + 1
    return ranking[
        ["request_id", "user_id", "item_id", "candidate_source", "source_rank", "source_score", "topic"]
    ]


def build_co_impression_counts(*, history_items: set[str], impressions: pd.DataFrame) -> Counter:
    if not history_items:
        return Counter()
    seed_request_ids = set(impressions.loc[impressions["item_id"].isin(history_items), "request_id"].tolist())
    if not seed_request_ids:
        return Counter()
    co_impressions = impressions.loc[impressions["request_id"].isin(seed_request_ids)].copy()
    if co_impressions.empty:
        return Counter()
    return Counter(co_impressions["item_id"].tolist())
