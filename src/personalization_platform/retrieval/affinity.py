from __future__ import annotations

from typing import Any

import pandas as pd


def build_affinity_source_candidates(
    *,
    event_log_inputs: dict[str, Any],
    candidate_count: int,
) -> pd.DataFrame:
    requests = event_log_inputs["requests"]
    impressions = event_log_inputs["impressions"]
    item_state = event_log_inputs["item_state"]
    history_lookup = event_log_inputs["history_lookup"]
    topic_count_lookup = event_log_inputs["topic_count_lookup"]

    item_topic_lookup = dict(zip(item_state["item_id"], item_state["topic"], strict=False))
    item_title_lookup = dict(zip(item_state["item_id"], item_state["title"], strict=False))
    candidate_rows: list[dict[str, Any]] = []

    requests_sorted = requests.sort_values("request_ts")
    for request in requests_sorted.itertuples(index=False):
        topic_counts = topic_count_lookup.get(request.request_id, {})
        history_items = history_lookup.get(request.request_id, set())
        if not topic_counts and not history_items:
            continue

        prior_impressions = impressions.loc[impressions["request_ts_lookup"] < request.request_ts].copy()
        if prior_impressions.empty:
            continue

        item_clicks = prior_impressions.groupby("item_id")["clicked"].sum().to_dict()
        item_request_counts = prior_impressions.groupby("item_id").size().to_dict()

        seed_request_ids = set(
            prior_impressions.loc[prior_impressions["item_id"].isin(history_items), "request_id"].tolist()
        )
        co_impressions = prior_impressions.loc[prior_impressions["request_id"].isin(seed_request_ids)].copy()
        co_impression_counts = (
            co_impressions.groupby("item_id").size().to_dict() if not co_impressions.empty else {}
        )

        ranking_rows: list[dict[str, Any]] = []
        for item_id, topic in item_topic_lookup.items():
            topic_affinity = int(topic_counts.get(topic, 0))
            co_impression_affinity = int(co_impression_counts.get(item_id, 0))
            if item_id in history_items:
                continue
            if topic_affinity <= 0 and co_impression_affinity <= 0:
                continue
            ranking_rows.append(
                {
                    "item_id": item_id,
                    "topic": topic,
                    "title": item_title_lookup.get(item_id, ""),
                    "topic_affinity_count": topic_affinity,
                    "co_impression_affinity_count": co_impression_affinity,
                    "prior_click_count": int(item_clicks.get(item_id, 0)),
                    "prior_impression_count": int(item_request_counts.get(item_id, 0)),
                    "affinity_score": float(
                        topic_affinity * 1000
                        + co_impression_affinity * 100
                        + item_clicks.get(item_id, 0) * 10
                        + item_request_counts.get(item_id, 0)
                    ),
                }
            )

        if not ranking_rows:
            continue

        ranking = pd.DataFrame(ranking_rows).sort_values(
            [
                "affinity_score",
                "topic_affinity_count",
                "co_impression_affinity_count",
                "prior_click_count",
                "item_id",
            ],
            ascending=[False, False, False, False, True],
        )
        ranking = ranking.head(candidate_count)
        for rank_position, row in enumerate(ranking.itertuples(index=False), start=1):
            candidate_rows.append(
                {
                    "request_id": request.request_id,
                    "user_id": request.user_id,
                    "item_id": row.item_id,
                    "candidate_source": "affinity",
                    "source_rank": rank_position,
                    "source_score": float(row.affinity_score),
                    "source_topic_affinity_count": int(row.topic_affinity_count),
                    "source_co_impression_affinity_count": int(row.co_impression_affinity_count),
                    "source_prior_click_count": int(row.prior_click_count),
                    "source_prior_impression_count": int(row.prior_impression_count),
                    "topic": row.topic,
                }
            )

    return pd.DataFrame(candidate_rows)
