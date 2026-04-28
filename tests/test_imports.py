from personalization_platform.pipeline.show_blueprint import load_config
from personalization_platform.delivery.local_api import create_local_api_app
from personalization_platform.reporting.bundle import build_reporting_bundle


def test_load_config_reads_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_name: smoke\n", encoding="utf-8")

    loaded = load_config(config_path)

    assert loaded["run_name"] == "smoke"


def test_create_local_api_app_returns_fastapi_app():
    app = create_local_api_app(
        {
            "input": {
                "rerank_base_dir": "data/processed/reranked_feed",
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
