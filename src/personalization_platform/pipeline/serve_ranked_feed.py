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
    replay_request = config["smoke_request"]
    contextual_request = config["contextual_smoke_request"]
    with TestClient(app) as client:
        health_response = client.get("/health")
        replay_response = client.post(
            "/score/feed",
            json={
                "request_id": replay_request["request_id"],
                "top_k": int(replay_request.get("top_k", 3)),
                "user_id": replay_request.get("user_id"),
            },
        )
        contextual_response = client.post(
            "/score/feed",
            json=contextual_request,
        )
        if health_response.status_code != 200:
            raise RuntimeError(f"Health endpoint returned unexpected status {health_response.status_code}.")
        if replay_response.status_code != 200:
            raise RuntimeError(
                f"Replay score endpoint returned unexpected status {replay_response.status_code}: {replay_response.text}"
            )
        if contextual_response.status_code != 200:
            raise RuntimeError(
                f"Contextual score endpoint returned unexpected status {contextual_response.status_code}: {contextual_response.text}"
            )

        health_payload = health_response.json()
        replay_payload = replay_response.json()
        contextual_payload = contextual_response.json()
        summary = {
            "api_name": replay_payload["api_name"],
            "health_status": health_payload["status"],
            "supported_modes": health_payload["supported_modes"],
            "replay_request_id": replay_payload["request_id"],
            "replay_top_item_id": replay_payload["items"][0]["item_id"] if replay_payload["items"] else None,
            "contextual_request_id": contextual_payload["request_id"],
            "contextual_top_item_id": (
                contextual_payload["items"][0]["item_id"] if contextual_payload["items"] else None
            ),
            "top_item_id": replay_payload["items"][0]["item_id"] if replay_payload["items"] else None,
            "request_id": replay_payload["request_id"],
            "user_id": replay_payload["user_id"],
            "dataset_split": replay_payload["dataset_split"],
            "returned_item_count": replay_payload["returned_item_count"],
            "contextual_returned_item_count": contextual_payload["returned_item_count"],
            "source_rerank_dir": replay_payload["source_rerank_dir"],
        }
        return summary, {"replay_response": replay_payload, "contextual_response": contextual_payload}, app.openapi()


if __name__ == "__main__":
    main()
