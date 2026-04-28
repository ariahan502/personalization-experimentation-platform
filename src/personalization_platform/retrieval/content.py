from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def build_content_source_candidates(
    *,
    event_log_inputs: dict[str, Any],
    candidate_count: int,
) -> pd.DataFrame:
    requests = event_log_inputs["requests"]
    user_state = event_log_inputs["user_state"]
    item_state = event_log_inputs["item_state"]
    history_lookup = event_log_inputs["history_lookup"]

    topic_lookup = dict(zip(item_state["item_id"], item_state["topic"], strict=False))
    subcategory_lookup = dict(zip(item_state["item_id"], item_state["subcategory"], strict=False))
    publisher_lookup = dict(zip(item_state["item_id"], item_state["publisher"], strict=False))
    creator_lookup = dict(zip(item_state["item_id"], item_state["creator_id"], strict=False))
    title_lookup = dict(zip(item_state["item_id"], item_state["title"], strict=False))
    recent_topic_counts_lookup = {
        row.request_id: Counter(_load_topic_counts(row.recent_topic_counts))
        for row in user_state.itertuples(index=False)
    }

    candidate_rows: list[dict[str, Any]] = []
    requests_sorted = requests.sort_values("request_ts")
    for request in requests_sorted.itertuples(index=False):
        history_items = history_lookup.get(request.request_id, set())
        if not history_items:
            continue

        topic_counts = recent_topic_counts_lookup.get(request.request_id, Counter())
        history_subcategory_counts = Counter(
            subcategory_lookup[item_id]
            for item_id in history_items
            if item_id in subcategory_lookup and subcategory_lookup[item_id]
        )
        history_publisher_counts = Counter(
            publisher_lookup[item_id]
            for item_id in history_items
            if item_id in publisher_lookup and publisher_lookup[item_id]
        )
        history_creator_counts = Counter(
            creator_lookup[item_id]
            for item_id in history_items
            if item_id in creator_lookup and creator_lookup[item_id]
        )
        history_title_tokens = {
            item_id: tokenize(title_lookup.get(item_id, ""))
            for item_id in history_items
        }

        ranking_rows: list[dict[str, Any]] = []
        for item in item_state.itertuples(index=False):
            if item.item_id in history_items:
                continue
            topic_affinity = int(topic_counts.get(item.topic, 0))
            subcategory_affinity = int(history_subcategory_counts.get(item.subcategory, 0))
            publisher_affinity = int(history_publisher_counts.get(item.publisher, 0))
            creator_affinity = int(history_creator_counts.get(item.creator_id, 0))
            lexical_overlap = max_title_overlap(
                candidate_tokens=tokenize(item.title),
                history_title_tokens=history_title_tokens,
            )
            if (
                topic_affinity <= 0
                and subcategory_affinity <= 0
                and publisher_affinity <= 0
                and creator_affinity <= 0
                and lexical_overlap <= 0
            ):
                continue
            content_score = float(
                creator_affinity * 1000
                + publisher_affinity * 300
                + subcategory_affinity * 100
                + topic_affinity * 30
                + lexical_overlap
            )
            ranking_rows.append(
                {
                    "item_id": item.item_id,
                    "topic": item.topic,
                    "subcategory": item.subcategory,
                    "publisher": item.publisher,
                    "creator_id": item.creator_id,
                    "content_score": content_score,
                    "topic_affinity_count": topic_affinity,
                    "subcategory_affinity_count": subcategory_affinity,
                    "publisher_affinity_count": publisher_affinity,
                    "creator_affinity_count": creator_affinity,
                    "title_overlap_count": lexical_overlap,
                }
            )

        if not ranking_rows:
            continue

        ranking = pd.DataFrame(ranking_rows).sort_values(
            [
                "content_score",
                "creator_affinity_count",
                "publisher_affinity_count",
                "subcategory_affinity_count",
                "topic_affinity_count",
                "title_overlap_count",
                "item_id",
            ],
            ascending=[False, False, False, False, False, False, True],
        )
        ranking = ranking.head(candidate_count)
        for rank_position, row in enumerate(ranking.itertuples(index=False), start=1):
            candidate_rows.append(
                {
                    "request_id": request.request_id,
                    "user_id": request.user_id,
                    "item_id": row.item_id,
                    "candidate_source": "content",
                    "source_rank": rank_position,
                    "source_score": float(row.content_score),
                    "source_topic_affinity_count": int(row.topic_affinity_count),
                    "source_subcategory_affinity_count": int(row.subcategory_affinity_count),
                    "source_publisher_affinity_count": int(row.publisher_affinity_count),
                    "source_creator_affinity_count": int(row.creator_affinity_count),
                    "source_title_overlap_count": int(row.title_overlap_count),
                    "topic": row.topic,
                }
            )

    return pd.DataFrame(candidate_rows)


def _load_topic_counts(raw_value: str) -> dict[str, int]:
    import json

    if not raw_value:
        return {}
    return {str(key): int(value) for key, value in json.loads(raw_value).items()}


def tokenize(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(str(text).lower()))


def max_title_overlap(*, candidate_tokens: set[str], history_title_tokens: dict[str, set[str]]) -> int:
    if not candidate_tokens:
        return 0
    overlap = 0
    for tokens in history_title_tokens.values():
        overlap = max(overlap, len(candidate_tokens.intersection(tokens)))
    return overlap
