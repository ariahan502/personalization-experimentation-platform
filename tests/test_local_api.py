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
        }
    )

    with TestClient(app) as client:
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

    assert replay.status_code == 200
    assert replay.json()["mode"] == "fixture_replay"
    assert replay.json()["items"][0]["item_id"] == "i1"

    assert contextual.status_code == 200
    contextual_payload = contextual.json()
    assert contextual_payload["mode"] == "contextual_scoring"
    assert contextual_payload["returned_item_count"] == 2
    assert {item["item_id"] for item in contextual_payload["items"]} == {"i2", "i3"}
    assert any(item["score_components"]["topic_affinity"] > 0.0 for item in contextual_payload["items"])
