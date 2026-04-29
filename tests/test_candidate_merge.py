import json

import pandas as pd

from personalization_platform.pipeline.build_candidates import merge_candidates


def test_merge_candidates_prefers_higher_priority_and_preserves_provenance():
    affinity = pd.DataFrame(
        [
            {
                "request_id": "r1",
                "user_id": "u1",
                "item_id": "i1",
                "candidate_source": "affinity",
                "source_rank": 1,
                "source_score": 0.8,
                "topic": "Tech",
                "source_priority": 1,
            },
            {
                "request_id": "r1",
                "user_id": "u1",
                "item_id": "i2",
                "candidate_source": "affinity",
                "source_rank": 2,
                "source_score": 0.6,
                "topic": "World",
                "source_priority": 1,
            },
        ]
    )
    trending = pd.DataFrame(
        [
            {
                "request_id": "r1",
                "user_id": "u1",
                "item_id": "i1",
                "candidate_source": "trending",
                "source_rank": 1,
                "source_score": 0.95,
                "topic": "Tech",
                "source_priority": 3,
            },
            {
                "request_id": "r1",
                "user_id": "u1",
                "item_id": "i3",
                "candidate_source": "trending",
                "source_rank": 2,
                "source_score": 0.7,
                "topic": "Sports",
                "source_priority": 3,
            },
        ]
    )

    merged = merge_candidates(source_frames=[affinity, trending], final_candidate_count=3)

    assert merged["item_id"].tolist() == ["i1", "i2", "i3"]
    first_row = merged.iloc[0].to_dict()
    assert first_row["candidate_source"] == "affinity"
    assert first_row["source_count"] == 2
    assert json.loads(first_row["source_list"]) == ["affinity", "trending"]
    assert [entry["source"] for entry in json.loads(first_row["source_details"])] == ["affinity", "trending"]
    assert merged["merged_rank"].tolist() == [1, 2, 3]


def test_merge_candidates_applies_final_candidate_count_after_dedup():
    content = pd.DataFrame(
        [
            {
                "request_id": "r2",
                "user_id": "u2",
                "item_id": "i1",
                "candidate_source": "content",
                "source_rank": 1,
                "source_score": 0.9,
                "topic": "Tech",
                "source_priority": 2,
            },
            {
                "request_id": "r2",
                "user_id": "u2",
                "item_id": "i2",
                "candidate_source": "content",
                "source_rank": 2,
                "source_score": 0.8,
                "topic": "World",
                "source_priority": 2,
            },
            {
                "request_id": "r2",
                "user_id": "u2",
                "item_id": "i3",
                "candidate_source": "content",
                "source_rank": 3,
                "source_score": 0.7,
                "topic": "Sports",
                "source_priority": 2,
            },
        ]
    )

    merged = merge_candidates(source_frames=[content], final_candidate_count=2)

    assert merged["item_id"].tolist() == ["i1", "i2"]
    assert merged["merged_rank"].tolist() == [1, 2]
