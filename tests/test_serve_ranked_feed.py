from personalization_platform.pipeline.serve_ranked_feed import (
    build_response_degraded_modes,
    build_smoke_request_results,
    build_smoke_request_specs,
)


def test_build_smoke_request_specs_supports_plural_and_legacy_keys():
    specs = build_smoke_request_specs(
        {
            "smoke_requests": [{"request_id": "r1", "user_id": "u1"}],
            "contextual_smoke_request": {"request_id": "r2", "user_id": "u2"},
            "assembled_smoke_requests": [{"request_id": "r3", "user_id": "u3"}],
        }
    )

    assert [spec["label"] for spec in specs] == ["replay_1", "contextual_1", "assembled_1"]
    assert [spec["payload"]["request_id"] for spec in specs] == ["r1", "r2", "r3"]


def test_build_smoke_request_results_and_degraded_modes_capture_all_requests():
    response_groups = {
        "fixture_replay": [
            {
                "request_id": "r1",
                "mode": "fixture_replay",
                "user_id": "u1",
                "treatment_id": "control",
                "returned_item_count": 2,
                "items": [{"item_id": "i1"}],
                "degraded_modes": [],
            }
        ],
        "contextual_scoring": [
            {
                "request_id": "r2",
                "mode": "contextual_scoring",
                "user_id": "u2",
                "treatment_id": "reranked_policy",
                "returned_item_count": 3,
                "items": [{"item_id": "i2"}],
                "degraded_modes": ["source_budget_truncated"],
            }
        ],
        "request_time_assembly": [],
    }

    results = build_smoke_request_results(response_groups)
    degraded_modes = build_response_degraded_modes(response_groups)

    assert [row["request_id"] for row in results] == ["r1", "r2"]
    assert results[1]["treatment_id"] == "reranked_policy"
    assert degraded_modes["contextual_scoring"] == [["source_budget_truncated"]]
