import pandas as pd

from personalization_platform.reranking.policy import compute_freshness_minutes, freshness_bonus, request_mrr


def test_compute_freshness_minutes_uses_latest_prior_impression():
    impressions = pd.DataFrame(
        {
            "item_id": ["i1", "i1", "i2"],
            "request_ts_lookup": pd.to_datetime(
                ["2026-04-28T10:00:00", "2026-04-28T10:45:00", "2026-04-28T10:30:00"]
            ),
        }
    )

    freshness = compute_freshness_minutes(
        impressions=impressions,
        item_id="i1",
        request_time=pd.Timestamp("2026-04-28T11:00:00"),
    )

    assert freshness == 15.0


def test_freshness_bonus_is_higher_for_less_recent_items():
    recent_bonus = freshness_bonus(value=5.0, weight=0.5)
    stale_bonus = freshness_bonus(value=240.0, weight=0.5)

    assert recent_bonus > stale_bonus


def test_request_mrr_uses_best_positive_rank_per_request():
    rows = pd.DataFrame(
        [
            {"request_id": "r1", "label": 0, "post_rank": 1},
            {"request_id": "r1", "label": 1, "post_rank": 2},
            {"request_id": "r2", "label": 1, "post_rank": 1},
            {"request_id": "r2", "label": 0, "post_rank": 2},
        ]
    )

    mrr = request_mrr(rows, rank_column="post_rank")

    assert mrr == 0.75
