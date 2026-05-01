from pathlib import Path

import pandas as pd

from personalization_platform.delivery.features import (
    build_serving_feature_state,
    hydrate_request_time_features,
)


def test_hydrate_request_time_features_uses_prior_serving_logs_when_available(tmp_path):
    run_dir = tmp_path / "artifacts" / "20260430_010101_local_api_smoke"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_id": "r1-request-1",
                "event_ts": "2026-04-30T10:00:00Z",
                "api_name": "test_api",
                "request_id": "r1",
                "mode": "contextual_scoring",
                "user_id": "u1",
                "top_k": 2,
                "candidate_input_count": 2,
                "returned_item_count": 2,
                "treatment_id": "unassigned",
                "source_rerank_dir": "rerank_dir",
            }
        ]
    ).to_csv(run_dir / "request_events.csv", index=False)
    pd.DataFrame(
        [
            {
                "event_id": "r1-exposure-1",
                "event_ts": "2026-04-30T10:00:01Z",
                "request_event_id": "r1-request-1",
                "request_id": "r1",
                "api_name": "test_api",
                "mode": "contextual_scoring",
                "user_id": "u1",
                "treatment_id": "unassigned",
                "item_id": "i2",
                "candidate_source": "content",
                "pre_rank": 1,
                "post_rank": 1,
                "rank_shift": 0,
                "prediction": 0.7,
                "rerank_score": 0.8,
                "freshness_bonus": 0.1,
            }
        ]
    ).to_csv(run_dir / "exposure_events.csv", index=False)
    pd.DataFrame(
        [
            {
                "event_id": "r1-click-1",
                "event_ts": "2026-04-30T10:00:02Z",
                "request_event_id": "r1-request-1",
                "request_id": "r1",
                "api_name": "test_api",
                "mode": "contextual_scoring",
                "user_id": "u1",
                "item_id": "i2",
                "click_label": 1,
            }
        ]
    ).to_csv(run_dir / "click_events.csv", index=False)

    contextual_state = {
        "impressions": pd.DataFrame(
            [
                {"item_id": "i2", "request_ts_lookup": pd.Timestamp("2026-04-30T09:45:00")},
                {"item_id": "i3", "request_ts_lookup": pd.Timestamp("2026-04-30T09:30:00")},
            ]
        ),
        "item_metadata_lookup": {
            "i2": {"topic": "Tech", "creator_id": "c2", "publisher": "pub2", "title": "Tech story"},
            "i3": {"topic": "World", "creator_id": "c3", "publisher": "pub3", "title": "World story"},
        },
        "item_click_priors": {"i2": 1.0, "i3": 0.0},
        "item_impression_priors": {"i2": 1.0, "i3": 0.5},
    }
    serving_feature_state = build_serving_feature_state(
        config={
            "input": {
                "serving_log_base_dir": str(tmp_path / "artifacts"),
                "serving_log_run_name": "local_api_smoke",
            }
        },
        contextual_state=contextual_state,
    )

    candidate_rows = pd.DataFrame(
        [
            {"item_id": "i2", "topic": "Tech", "creator_id": "c2", "item_impression_priors": 1.0},
            {"item_id": "i3", "topic": "World", "creator_id": "c3", "item_impression_priors": 0.5},
        ]
    )
    hydrated, summary = hydrate_request_time_features(
        candidate_rows=candidate_rows,
        history_context={"history_item_ids": {"i1"}, "topic_counts": {"Tech": 2}},
        contextual_state=contextual_state,
        serving_feature_state=serving_feature_state,
        scoring_weights={},
        user_id="u1",
        request_time=pd.Timestamp("2026-04-30T10:05:00"),
    )

    tech_row = hydrated.loc[hydrated["item_id"] == "i2"].iloc[0]
    assert summary["serving_logs_available"] is True
    assert summary["recent_request_event_count"] == 1
    assert tech_row["recent_item_ctr"] == 1.0
    assert tech_row["recent_topic_click_share"] == 1.0
    assert tech_row["recent_user_click_rate"] == 1.0


def test_hydrate_request_time_features_falls_back_cleanly_without_serving_logs():
    contextual_state = {
        "impressions": pd.DataFrame(
            [{"item_id": "i2", "request_ts_lookup": pd.Timestamp("2026-04-30T09:45:00")}]
        ),
        "item_metadata_lookup": {"i2": {"topic": "Tech", "creator_id": "c2", "publisher": "pub2", "title": "Tech"}},
        "item_click_priors": {"i2": 1.0},
        "item_impression_priors": {"i2": 1.0},
    }
    candidate_rows = pd.DataFrame([{"item_id": "i2", "topic": "Tech", "creator_id": "c2", "item_impression_priors": 1.0}])
    hydrated, summary = hydrate_request_time_features(
        candidate_rows=candidate_rows,
        history_context={"history_item_ids": set(), "topic_counts": {}},
        contextual_state=contextual_state,
        serving_feature_state={"logs_available": False, "source_run_dir": None},
        scoring_weights={},
        user_id="u1",
        request_time=pd.Timestamp("2026-04-30T10:05:00"),
    )

    row = hydrated.iloc[0]
    assert summary["serving_logs_available"] is False
    assert summary["fallbacks"]
    assert row["recent_item_ctr"] == 0.0
    assert row["recent_topic_click_share"] == 0.0
    assert row["recent_user_click_rate"] == 0.0
