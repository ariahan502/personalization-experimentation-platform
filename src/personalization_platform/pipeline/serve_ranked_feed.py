from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from fastapi.testclient import TestClient

from personalization_platform.delivery.local_api import create_local_api_app
from personalization_platform.utils.artifacts import create_run_dir, write_json, write_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve or smoke-test the local ranked feed API.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    parser.add_argument("--serve", action="store_true", help="Launch the local FastAPI server instead of smoke-testing it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    app = create_local_api_app(config)

    if args.serve:
        uvicorn.run(
            app,
            host=config["api"].get("host", "127.0.0.1"),
            port=int(config["api"].get("port", 8000)),
        )
        return

    run_name = config.get("run_name", "local_api_smoke")
    run_dir = create_run_dir(run_name, base_dir=config["artifacts"]["base_dir"])
    summary, response_payload, openapi_snapshot = smoke_test_api(app=app, config=config)
    write_yaml(run_dir / "config.yaml", config)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "smoke_response.json", response_payload)
    write_json(run_dir / "openapi_snapshot.json", openapi_snapshot)
    print(f"Wrote local API smoke bundle to {run_dir}")


def smoke_test_api(*, app: Any, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    smoke_request = config["smoke_request"]
    with TestClient(app) as client:
        health_response = client.get("/health")
        score_response = client.post(
            "/score/feed",
            json={
                "request_id": smoke_request["request_id"],
                "top_k": int(smoke_request.get("top_k", 3)),
                "user_id": smoke_request.get("user_id"),
            },
        )
        if health_response.status_code != 200:
            raise RuntimeError(f"Health endpoint returned unexpected status {health_response.status_code}.")
        if score_response.status_code != 200:
            raise RuntimeError(
                f"Score endpoint returned unexpected status {score_response.status_code}: {score_response.text}"
            )

        health_payload = health_response.json()
        response_payload = score_response.json()
        summary = {
            "api_name": response_payload["api_name"],
            "mode": response_payload["mode"],
            "health_status": health_payload["status"],
            "request_id": response_payload["request_id"],
            "user_id": response_payload["user_id"],
            "dataset_split": response_payload["dataset_split"],
            "returned_item_count": response_payload["returned_item_count"],
            "top_item_id": response_payload["items"][0]["item_id"] if response_payload["items"] else None,
            "source_rerank_dir": response_payload["source_rerank_dir"],
        }
        return summary, response_payload, app.openapi()


if __name__ == "__main__":
    main()
