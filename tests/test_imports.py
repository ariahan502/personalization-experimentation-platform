import pandas as pd

from personalization_platform.pipeline.show_blueprint import load_config
from personalization_platform.delivery.local_api import create_local_api_app
from personalization_platform.reporting.bundle import build_reporting_bundle


def test_load_config_reads_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_name: smoke\n", encoding="utf-8")

    loaded = load_config(config_path)

    assert loaded["run_name"] == "smoke"


def test_create_local_api_app_returns_fastapi_app(tmp_path):
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
                "history_length": 1,
                "is_cold_start": False,
                "prediction": 0.8,
                "merged_rank": 1,
                "pre_rank": 1,
                "post_rank": 1,
                "rank_shift": 0,
                "freshness_minutes_since_last_seen": 10.0,
                "freshness_bonus": 0.3,
                "diversity_penalty": 0.0,
                "creator_penalty": 0.0,
                "rerank_score": 1.1,
                "candidate_source": "affinity",
                "source_list": '["affinity"]',
                "source_details": '[{"source": "affinity"}]',
            }
        ]
    ).to_csv(rerank_dir / "reranked_rows.csv", index=False)

    app = create_local_api_app(
        {
            "input": {
                "rerank_base_dir": str(rerank_base_dir),
                "rerank_run_name": "rerank_smoke",
            },
            "api": {
                "api_name": "test_api",
                "title": "Test API",
            },
        }
    )

    assert app.title == "Test API"


def test_build_reporting_bundle_is_importable():
    assert callable(build_reporting_bundle)
