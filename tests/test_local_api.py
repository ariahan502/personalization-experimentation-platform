import pandas as pd
from fastapi.testclient import TestClient

from personalization_platform.delivery.local_api import create_local_api_app


def test_local_api_replay_and_contextual_modes(tmp_path):
    rerank_base_dir = tmp_path / "reranked"
    rerank_dir = rerank_base_dir / "20260428_000000_rerank_smoke"
    rerank_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "request_id": "r1",
                "user_id": "u1",
                "item_id": "i1",
                "topic": "Tech",
                "creator_id": "c1",
                "label": 1,
                "dataset_split": "valid",
                "history_length": 2,
                "is_cold_start": False,
                "prediction": 0.7,
                "merged_rank": 1,
                "pre_rank": 1,
                "post_rank": 1,
                "rank_shift": 0,
                "freshness_minutes_since_last_seen": 15.0,
                "freshness_bonus": 0.28,
                "diversity_penalty": 0.0,
                "creator_penalty": 0.0,
                "rerank_score": 0.98,
                "candidate_source": "affinity",
                "source_list": '["affinity"]',
                "source_details": '[{"source": "affinity"}]',
            },
            {
                "request_id": "r1",
                "user_id": "u1",
                "item_id": "i2",
                "topic": "World",
                "creator_id": "c2",
                "label": 0,
                "dataset_split": "valid",
                "history_length": 2,
                "is_cold_start": False,
                "prediction": 0.4,
                "merged_rank": 2,
                "pre_rank": 2,
                "post_rank": 2,
                "rank_shift": 0,
                "freshness_minutes_since_last_seen": 40.0,
                "freshness_bonus": 0.21,
                "diversity_penalty": 0.0,
                "creator_penalty": 0.0,
                "rerank_score": 0.61,
                "candidate_source": "content",
                "source_list": '["content"]',
                "source_details": '[{"source": "content"}]',
            },
        ]
    ).to_csv(rerank_dir / "reranked_rows.csv", index=False)

    event_log_base_dir = tmp_path / "event_logs"
    event_log_dir = event_log_base_dir / "20260428_000000_mind_smoke_event_log"
    event_log_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "request_id": "r0",
                "user_id": "u1",
                "session_id": "s1",
                "request_ts": "2026-04-28T10:00:00",
                "split": "train",
                "candidate_count": 2,
                "history_length": 2,
                "request_index_in_session": 1,
            }
        ]
    ).to_csv(event_log_dir / "requests.csv", index=False)
    pd.DataFrame(
        [
            {
                "impression_id": "r0-i1",
                "request_id": "r0",
                "user_id": "u1",
                "item_id": "i1",
                "position": 1,
                "clicked": 1,
                "topic": "Tech",
            },
            {
                "impression_id": "r0-i2",
                "request_id": "r0",
                "user_id": "u1",
                "item_id": "i2",
                "position": 2,
                "clicked": 0,
                "topic": "World",
            },
        ]
    ).to_csv(event_log_dir / "impressions.csv", index=False)
    pd.DataFrame(
        [
            {
                "request_id": "r0",
                "user_id": "u1",
                "history_item_ids": '["i1"]',
                "history_click_count": 2,
                "is_cold_start": False,
                "recent_topic_counts": '{"Tech": 2, "World": 1}',
            }
        ]
    ).to_csv(event_log_dir / "user_state.csv", index=False)
    pd.DataFrame(
        [
            {
                "item_id": "i1",
                "topic": "Tech",
                "subcategory": "AI",
                "title": "AI chip launch",
                "publisher": "techwire",
                "creator_id": "c1",
                "published_ts": "",
                "abstract": "",
                "entity_ids": "[]",
            },
            {
                "item_id": "i2",
                "topic": "World",
                "subcategory": "Politics",
                "title": "Policy update",
                "publisher": "worlddesk",
                "creator_id": "c2",
                "published_ts": "",
                "abstract": "",
                "entity_ids": "[]",
            },
            {
                "item_id": "i3",
                "topic": "Tech",
                "subcategory": "AI",
                "title": "Model release",
                "publisher": "techwire",
                "creator_id": "c1",
                "published_ts": "",
                "abstract": "",
                "entity_ids": "[]",
            },
        ]
    ).to_csv(event_log_dir / "item_state.csv", index=False)

    app = create_local_api_app(
        {
            "input": {
                "rerank_base_dir": str(rerank_base_dir),
                "rerank_run_name": "rerank_smoke",
                "event_log_base_dir": str(event_log_base_dir),
                "event_log_run_name": "mind_smoke_event_log",
            },
            "api": {
                "api_name": "test_api",
                "title": "Test API",
            },
            "request_time_retrieval": {
                "candidate_count": 4,
                "sources": [
                    {"name": "affinity", "candidate_count": 3, "priority": 1},
                    {"name": "content", "candidate_count": 3, "priority": 2},
                    {"name": "trending", "candidate_count": 3, "priority": 3},
                ],
            },
            "experiment": {
                "experiment_id": "serving-exp",
                "assignment_unit": "user_id",
                "salt": "serving-seed",
                "treatments": [
                    {"treatment_id": "control", "treatment_name": "control_feed", "weight": 0.5, "is_control": True},
                    {"treatment_id": "treatment", "treatment_name": "treatment_feed", "weight": 0.5, "is_control": False},
                ],
            },
        }
    )

    with TestClient(app) as client:
        health = client.get("/health")
        replay = client.post("/score/feed", json={"request_id": "r1", "user_id": "u1", "top_k": 2})
        contextual = client.post(
            "/score/feed",
            json={
                "request_id": "contextual-u1",
                "user_id": "u1",
                "top_k": 2,
                "candidate_items": [
                    {"item_id": "i2"},
                    {"item_id": "i3"},
                ],
            },
        )
        assembled = client.post(
            "/score/feed",
            json={
                "request_id": "assembled-u1",
                "user_id": "u1",
                "top_k": 2,
                "history_item_ids": ["i1"],
                "history_topics": ["Tech", "Tech", "World"],
            },
        )

    assert health.status_code == 200
    health_payload = health.json()
    assert health_payload["overall_status"] == "pass"
    assert health_payload["serving_state"]["rerank_row_count"] == 2
    assert health_payload["serving_state"]["request_time_assembly_enabled"] is True
    assert health_payload["serving_state"]["serving_log_features_available"] is False
    assert health_payload["serving_state"]["experiment_assignment_enabled"] is True
    assert health_payload["serving_state"]["experiment_id"] == "serving-exp"
    assert any(check["name"] == "contextual_state_available" for check in health_payload["health_checks"])
    assert any(check["name"] == "request_time_assembly_available" for check in health_payload["health_checks"])
    assert health_payload["feature_contract"]["contract_version"] == "v1"
    assert health_payload["feature_contract"]["serving_request_schema"]["extra_fields_policy"] == "forbid"
    unsupported_features = {
        row["feature"] for row in health_payload["feature_contract"]["unsupported_online_training_features"]
    }
    assert "merged_rank" in unsupported_features
    assert "candidate_seen_in_impressions" in unsupported_features

    assert replay.status_code == 200
    assert replay.json()["mode"] == "fixture_replay"
    assert replay.json()["items"][0]["item_id"] == "i1"
    assert replay.json()["experiment_id"] == "serving-exp"

    assert contextual.status_code == 200
    contextual_payload = contextual.json()
    assert contextual_payload["mode"] == "contextual_scoring"
    assert contextual_payload["returned_item_count"] == 2
    assert {item["item_id"] for item in contextual_payload["items"]} == {"i2", "i3"}
    assert any(item["score_components"]["topic_affinity"] > 0.0 for item in contextual_payload["items"])
    assert "recent_item_ctr" in contextual_payload["items"][0]["score_components"]
    assert contextual_payload["degraded_modes"] == []
    assert contextual_payload["assignment_unit"] == "user_id"
    assert contextual_payload["assignment_unit_id"] == "u1"

    assert assembled.status_code == 200
    assembled_payload = assembled.json()
    assert assembled_payload["mode"] == "request_time_assembly"
    assert assembled_payload["returned_item_count"] == 2
    assert all(item["candidate_source"] in {"affinity", "content", "trending"} for item in assembled_payload["items"])
    assert assembled_payload["degraded_modes"] == []
    assert assembled_payload["treatment_id"] == contextual_payload["treatment_id"]

    invalid_payload = client.post(
        "/score/feed",
        json={
            "user_id": "u1",
            "candidate_items": [
                {"item_id": "i2", "merged_rank": 1},
            ],
        },
    )
    assert invalid_payload.status_code == 422


def test_local_api_assigns_different_users_to_different_treatments(tmp_path):
    rerank_base_dir = tmp_path / "reranked"
    rerank_dir = rerank_base_dir / "20260428_000000_rerank_smoke"
    rerank_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "request_id": "r1",
                "user_id": "u1",
                "item_id": "i1",
                "topic": "Tech",
                "creator_id": "c1",
                "label": 1,
                "dataset_split": "valid",
                "history_length": 2,
                "is_cold_start": False,
                "prediction": 0.7,
                "merged_rank": 1,
                "pre_rank": 1,
                "post_rank": 1,
                "rank_shift": 0,
                "freshness_minutes_since_last_seen": 15.0,
                "freshness_bonus": 0.28,
                "diversity_penalty": 0.0,
                "creator_penalty": 0.0,
                "rerank_score": 0.98,
                "candidate_source": "affinity",
                "source_list": '["affinity"]',
                "source_details": '[{"source": "affinity"}]',
            }
        ]
    ).to_csv(rerank_dir / "reranked_rows.csv", index=False)

    event_log_base_dir = tmp_path / "event_logs"
    event_log_dir = event_log_base_dir / "20260428_000000_mind_smoke_event_log"
    event_log_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "request_id": "r0",
                "user_id": "u1",
                "session_id": "s1",
                "request_ts": "2026-04-28T10:00:00",
                "split": "train",
                "candidate_count": 1,
                "history_length": 1,
                "request_index_in_session": 1,
            }
        ]
    ).to_csv(event_log_dir / "requests.csv", index=False)
    pd.DataFrame(
        [
            {
                "impression_id": "r0-i1",
                "request_id": "r0",
                "user_id": "u1",
                "item_id": "i1",
                "position": 1,
                "clicked": 1,
                "topic": "Tech",
            }
        ]
    ).to_csv(event_log_dir / "impressions.csv", index=False)
    pd.DataFrame(
        [
            {
                "request_id": "r0",
                "user_id": "u1",
                "history_item_ids": '["i1"]',
                "history_click_count": 1,
                "is_cold_start": False,
                "recent_topic_counts": '{"Tech": 1}',
            }
        ]
    ).to_csv(event_log_dir / "user_state.csv", index=False)
    pd.DataFrame(
        [
            {
                "item_id": "i1",
                "topic": "Tech",
                "subcategory": "AI",
                "title": "AI chip launch",
                "publisher": "techwire",
                "creator_id": "c1",
                "published_ts": "",
                "abstract": "",
                "entity_ids": "[]",
            }
        ]
    ).to_csv(event_log_dir / "item_state.csv", index=False)

    app = create_local_api_app(
        {
            "input": {
                "rerank_base_dir": str(rerank_base_dir),
                "rerank_run_name": "rerank_smoke",
                "event_log_base_dir": str(event_log_base_dir),
                "event_log_run_name": "mind_smoke_event_log",
            },
            "api": {
                "api_name": "test_api",
                "title": "Test API",
            },
            "experiment": {
                "experiment_id": "serving-exp",
                "assignment_unit": "user_id",
                "salt": "serving-seed",
                "treatments": [
                    {"treatment_id": "control", "treatment_name": "control_feed", "weight": 0.5, "is_control": True},
                    {"treatment_id": "treatment", "treatment_name": "treatment_feed", "weight": 0.5, "is_control": False},
                ],
            },
        }
    )

    with TestClient(app) as client:
        treatment_response = client.post("/score/feed", json={"request_id": "r1", "user_id": "u1", "top_k": 1})
        control_response = client.post("/score/feed", json={"request_id": "r2", "user_id": "u3", "top_k": 1})

    assert treatment_response.status_code == 200
    assert control_response.status_code == 200
    assert treatment_response.json()["treatment_id"] == "treatment"
    assert control_response.json()["treatment_id"] == "control"


def test_local_api_request_time_assembly_uses_trending_fallback_when_primary_sources_are_empty(tmp_path):
    rerank_base_dir = tmp_path / "reranked"
    rerank_dir = rerank_base_dir / "20260428_000000_rerank_smoke"
    rerank_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "request_id": "r1",
                "user_id": "u1",
                "item_id": "i1",
                "topic": "Tech",
                "creator_id": "c1",
                "dataset_split": "valid",
                "prediction": 0.7,
                "pre_rank": 1,
                "post_rank": 1,
                "rank_shift": 0,
                "freshness_bonus": 0.2,
                "rerank_score": 0.9,
                "candidate_source": "affinity",
            }
        ]
    ).to_csv(rerank_dir / "reranked_rows.csv", index=False)

    event_log_base_dir = tmp_path / "event_logs"
    event_log_dir = event_log_base_dir / "20260428_000000_mind_smoke_event_log"
    event_log_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "request_id": "r0",
                "user_id": "u1",
                "session_id": "s1",
                "request_ts": "2026-04-28T10:00:00",
                "split": "train",
                "candidate_count": 2,
                "history_length": 0,
                "request_index_in_session": 1,
            }
        ]
    ).to_csv(event_log_dir / "requests.csv", index=False)
    pd.DataFrame(
        [
            {"impression_id": "r0-i1", "request_id": "r0", "user_id": "u1", "item_id": "i1", "position": 1, "clicked": 1, "topic": "Tech"},
            {"impression_id": "r0-i2", "request_id": "r0", "user_id": "u1", "item_id": "i2", "position": 2, "clicked": 0, "topic": "World"},
        ]
    ).to_csv(event_log_dir / "impressions.csv", index=False)
    pd.DataFrame(
        [
            {
                "request_id": "r0",
                "user_id": "u1",
                "history_item_ids": "[]",
                "history_click_count": 0,
                "is_cold_start": True,
                "recent_topic_counts": "{}",
            }
        ]
    ).to_csv(event_log_dir / "user_state.csv", index=False)
    pd.DataFrame(
        [
            {"item_id": "i1", "topic": "Tech", "subcategory": "AI", "title": "AI chip launch", "publisher": "techwire", "creator_id": "c1", "published_ts": "", "abstract": "", "entity_ids": "[]"},
            {"item_id": "i2", "topic": "World", "subcategory": "Politics", "title": "Policy update", "publisher": "worlddesk", "creator_id": "c2", "published_ts": "", "abstract": "", "entity_ids": "[]"},
            {"item_id": "i3", "topic": "Sports", "subcategory": "Soccer", "title": "Game recap", "publisher": "sportsdesk", "creator_id": "c3", "published_ts": "", "abstract": "", "entity_ids": "[]"},
        ]
    ).to_csv(event_log_dir / "item_state.csv", index=False)

    app = create_local_api_app(
        {
            "input": {
                "rerank_base_dir": str(rerank_base_dir),
                "rerank_run_name": "rerank_smoke",
                "event_log_base_dir": str(event_log_base_dir),
                "event_log_run_name": "mind_smoke_event_log",
            },
            "api": {"api_name": "test_api", "title": "Test API"},
            "request_time_retrieval": {
                "candidate_count": 3,
                "max_sources_per_request": 1,
                "fallback_to_trending_only": True,
                "sources": [
                    {"name": "affinity", "candidate_count": 2, "priority": 1},
                    {"name": "content", "candidate_count": 2, "priority": 2},
                ],
            },
            "experiment": {
                "experiment_id": "serving-exp",
                "assignment_unit": "request_id",
                "salt": "serving-seed",
                "treatments": [
                    {"treatment_id": "control", "treatment_name": "control_feed", "weight": 0.5, "is_control": True},
                    {"treatment_id": "treatment", "treatment_name": "treatment_feed", "weight": 0.5, "is_control": False},
                ],
            },
        }
    )

    with TestClient(app) as client:
        assembled = client.post(
            "/score/feed",
            json={"request_id": "assembled-u1", "user_id": "u1", "top_k": 2},
        )

    assert assembled.status_code == 200
    payload = assembled.json()
    assert payload["mode"] == "request_time_assembly"
    assert "source_budget_truncated" in payload["degraded_modes"]
    assert "trending_only_fallback" in payload["degraded_modes"]
    assert payload["assignment_unit"] == "request_id"
    assert payload["assignment_unit_id"] == "assembled-u1"
    assert payload["items"]
    assert all(item["candidate_source"] == "trending" for item in payload["items"])
