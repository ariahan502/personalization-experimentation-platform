from personalization_platform.delivery.event_logging import (
    build_event_log_summary,
    build_serving_interaction_logs,
)


def test_build_serving_interaction_logs_records_request_exposure_response_and_clicks():
    logs = build_serving_interaction_logs(
        api_name="test_api",
        request_payloads=[
            {"request_id": "r1", "user_id": "u1", "top_k": 2},
            {"request_id": "r2", "user_id": "u2", "top_k": 1, "candidate_items": [{"item_id": "i3"}]},
        ],
        response_payloads=[
            {
                "api_name": "test_api",
                "mode": "fixture_replay",
                "request_id": "r1",
                "user_id": "u1",
                "source_rerank_dir": "rerank_dir",
                "returned_item_count": 2,
                "items": [
                    {
                        "item_id": "i1",
                        "candidate_source": "affinity",
                        "pre_rank": 1,
                        "post_rank": 1,
                        "rank_shift": 0,
                        "prediction": 0.8,
                        "rerank_score": 0.9,
                        "freshness_bonus": 0.1,
                    },
                    {
                        "item_id": "i2",
                        "candidate_source": "trending",
                        "pre_rank": 2,
                        "post_rank": 2,
                        "rank_shift": 0,
                        "prediction": 0.4,
                        "rerank_score": 0.5,
                        "freshness_bonus": 0.1,
                    },
                ],
            },
            {
                "api_name": "test_api",
                "mode": "contextual_scoring",
                "request_id": "r2",
                "user_id": "u2",
                "source_rerank_dir": "rerank_dir",
                "returned_item_count": 1,
                "items": [
                    {
                        "item_id": "i3",
                        "candidate_source": "content",
                        "pre_rank": 1,
                        "post_rank": 1,
                        "rank_shift": 0,
                        "prediction": 0.7,
                        "rerank_score": 0.8,
                        "freshness_bonus": 0.05,
                    }
                ],
            },
        ],
        simulated_clicked_item_ids=[["i1"], []],
    )

    assert len(logs["request_events"]) == 2
    assert len(logs["exposure_events"]) == 3
    assert len(logs["response_events"]) == 2
    assert len(logs["click_events"]) == 1
    assert logs["request_events"]["treatment_id"].tolist() == ["unassigned", "unassigned"]
    assert logs["exposure_events"]["request_event_id"].nunique() == 2

    summary = build_event_log_summary(logs)
    assert summary["request_event_count"] == 2
    assert summary["click_event_count"] == 1
    assert summary["logged_modes"] == ["contextual_scoring", "fixture_replay"]


def test_build_serving_interaction_logs_persists_serving_treatment_context():
    logs = build_serving_interaction_logs(
        api_name="test_api",
        request_payloads=[{"request_id": "r1", "user_id": "u1", "top_k": 1}],
        response_payloads=[
            {
                "api_name": "test_api",
                "mode": "fixture_replay",
                "request_id": "r1",
                "user_id": "u1",
                "experiment_id": "exp-1",
                "assignment_unit": "user_id",
                "assignment_unit_id": "u1",
                "hash_bucket": 0.12,
                "treatment_id": "control",
                "treatment_name": "control_feed",
                "is_control": 1,
                "source_rerank_dir": "rerank_dir",
                "returned_item_count": 1,
                "items": [
                    {
                        "item_id": "i1",
                        "candidate_source": "affinity",
                        "pre_rank": 1,
                        "post_rank": 1,
                        "rank_shift": 0,
                        "prediction": 0.8,
                        "rerank_score": 0.9,
                        "freshness_bonus": 0.1,
                    }
                ],
            }
        ],
        simulated_clicked_item_ids=[["i1"]],
    )

    request_row = logs["request_events"].iloc[0]
    exposure_row = logs["exposure_events"].iloc[0]
    click_row = logs["click_events"].iloc[0]
    assert request_row["experiment_id"] == "exp-1"
    assert request_row["treatment_id"] == "control"
    assert exposure_row["assignment_unit_id"] == "u1"
    assert exposure_row["treatment_name"] == "control_feed"
    assert click_row["treatment_id"] == "control"
