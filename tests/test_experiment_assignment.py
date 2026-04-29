import pandas as pd

from personalization_platform.experiments.assignment import assign_experiment, compute_hash_bucket


def test_compute_hash_bucket_is_deterministic():
    first = compute_hash_bucket(
        experiment_id="exp-1",
        salt="fixed-salt",
        assignment_unit_id="user-42",
    )
    second = compute_hash_bucket(
        experiment_id="exp-1",
        salt="fixed-salt",
        assignment_unit_id="user-42",
    )

    assert first == second
    assert 0.0 <= first < 1.0


def test_assign_experiment_keeps_same_user_on_same_treatment(tmp_path):
    rerank_dir = tmp_path / "reranked"
    output_dir = rerank_dir / "20260428_000000_rerank_smoke"
    output_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "request_id": "r1",
                "user_id": "u1",
                "item_id": "i1",
                "dataset_split": "valid",
                "post_rank": 1,
            },
            {
                "request_id": "r1",
                "user_id": "u1",
                "item_id": "i2",
                "dataset_split": "valid",
                "post_rank": 2,
            },
            {
                "request_id": "r2",
                "user_id": "u1",
                "item_id": "i3",
                "dataset_split": "valid",
                "post_rank": 1,
            },
            {
                "request_id": "r3",
                "user_id": "u2",
                "item_id": "i4",
                "dataset_split": "valid",
                "post_rank": 1,
            },
        ]
    ).to_csv(output_dir / "reranked_rows.csv", index=False)

    config = {
        "input": {
            "rerank_base_dir": str(rerank_dir),
            "rerank_run_name": "rerank_smoke",
        },
        "experiment": {
            "experiment_id": "exp-1",
            "assignment_unit": "user_id",
            "salt": "fixed-salt",
            "treatments": [
                {
                    "treatment_id": "control",
                    "treatment_name": "Control",
                    "weight": 0.5,
                    "is_control": True,
                },
                {
                    "treatment_id": "treatment",
                    "treatment_name": "Treatment",
                    "weight": 0.5,
                },
            ],
        },
    }

    assignment_table, assigned_exposures, metrics, _ = assign_experiment(config)

    u1_treatments = assignment_table.loc[assignment_table["user_id"] == "u1", "treatment_id"].unique().tolist()
    assert len(u1_treatments) == 1
    assert metrics["determinism_check"]["inconsistent_assignment_units"] == 0
    assert sorted(assigned_exposures["treatment_id"].unique().tolist()) == sorted(
        assignment_table["treatment_id"].unique().tolist()
    )
