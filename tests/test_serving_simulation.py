import pandas as pd

from personalization_platform.delivery.simulation import (
    build_simulated_logs,
    simulate_click,
)


def test_simulate_click_is_deterministic():
    first = simulate_click(
        seed=17,
        request_id="r1",
        simulated_request_id="r1::sim_round_1::control",
        item_id="i1",
        treatment_id="control",
        assignment_strategy="paired_treatment_replay",
        position=1,
        label=1,
        round_index=1,
        positive_click_probability=0.8,
        negative_click_probability=0.05,
        position_decay=[1.0, 0.7],
    )
    second = simulate_click(
        seed=17,
        request_id="r1",
        simulated_request_id="r1::sim_round_1::control",
        item_id="i1",
        treatment_id="control",
        assignment_strategy="paired_treatment_replay",
        position=1,
        label=1,
        round_index=1,
        positive_click_probability=0.8,
        negative_click_probability=0.05,
        position_decay=[1.0, 0.7],
    )

    assert first == second


def test_build_simulated_logs_creates_request_and_treatment_rows(tmp_path):
    reranked_rows = pd.DataFrame(
        [
            {
                "request_id": "r1",
                "user_id": "u1",
                "dataset_split": "valid",
                "item_id": "i1",
                "label": 1,
                "pre_rank": 1,
                "post_rank": 1,
                "prediction": 0.8,
                "rerank_score": 0.9,
                "freshness_bonus": 0.1,
                "candidate_source": "affinity",
                "topic": "Tech",
                "creator_id": "c1",
            },
            {
                "request_id": "r1",
                "user_id": "u1",
                "dataset_split": "valid",
                "item_id": "i2",
                "label": 0,
                "pre_rank": 2,
                "post_rank": 2,
                "prediction": 0.2,
                "rerank_score": 0.3,
                "freshness_bonus": 0.0,
                "candidate_source": "content",
                "topic": "World",
                "creator_id": "c2",
            },
            {
                "request_id": "r2",
                "user_id": "u2",
                "dataset_split": "valid",
                "item_id": "i3",
                "label": 0,
                "pre_rank": 1,
                "post_rank": 2,
                "prediction": 0.6,
                "rerank_score": 0.4,
                "freshness_bonus": 0.0,
                "candidate_source": "trending",
                "topic": "Sports",
                "creator_id": "c3",
            },
            {
                "request_id": "r2",
                "user_id": "u2",
                "dataset_split": "valid",
                "item_id": "i4",
                "label": 1,
                "pre_rank": 2,
                "post_rank": 1,
                "prediction": 0.5,
                "rerank_score": 0.8,
                "freshness_bonus": 0.1,
                "candidate_source": "affinity",
                "topic": "Sports",
                "creator_id": "c4",
            },
        ]
    )
    request_rows = reranked_rows[["request_id", "user_id", "dataset_split"]].drop_duplicates()

    logs = build_simulated_logs(
        reranked_rows=reranked_rows,
        request_rows=request_rows,
        rerank_dir=tmp_path,
        experiment={
            "experiment_id": "sim-exp",
            "assignment_unit": "request_id",
            "salt": "sim-seed",
            "control_treatment_id": "control",
            "treatments": [
                {"treatment_id": "control", "treatment_name": "control_feed", "weight": 0.5, "is_control": True},
                {"treatment_id": "reranked_policy", "treatment_name": "reranked_feed", "weight": 0.5, "is_control": False},
            ],
        },
        simulation={
            "assignment_strategy": "paired_treatment_replay",
            "rounds": 2,
            "top_k": 2,
            "random_seed": 17,
            "base_timestamp": "2026-04-30T12:00:00Z",
            "positive_click_probability": 1.0,
            "negative_click_probability": 0.0,
            "position_decay": [1.0, 1.0],
        },
        api_name="simulator",
    )

    assert len(logs["request_events"]) == 8
    assert set(logs["request_events"]["mode"]) == {"deterministic_replay_simulation"}
    assert logs["request_events"]["treatment_id"].isin({"control", "reranked_policy"}).all()
    treatment_counts = logs["request_events"]["treatment_id"].value_counts().to_dict()
    assert treatment_counts == {"control": 4, "reranked_policy": 4}
    assert len(logs["response_events"]) == 8
    assert logs["exposure_events"]["post_rank"].max() == 2
    assert set(logs["click_events"]["item_id"]).issubset({"i1", "i4"})
