from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from fastapi.testclient import TestClient

from personalization_platform.delivery.event_logging import (
    build_event_log_summary,
    build_serving_interaction_logs,
)
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
    summary, response_payload, openapi_snapshot, interaction_logs = smoke_test_api(app=app, config=config)
    write_yaml(run_dir / "config.yaml", config)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "health.json", summary["observability"])
    write_json(run_dir / "serving_contract.json", summary["serving_contract"])
    write_json(run_dir / "smoke_response.json", response_payload)
    write_json(run_dir / "openapi_snapshot.json", openapi_snapshot)
    interaction_logs["request_events"].to_csv(run_dir / "request_events.csv", index=False)
    interaction_logs["exposure_events"].to_csv(run_dir / "exposure_events.csv", index=False)
    interaction_logs["response_events"].to_csv(run_dir / "response_events.csv", index=False)
    interaction_logs["click_events"].to_csv(run_dir / "click_events.csv", index=False)
    print(f"Wrote local API smoke bundle to {run_dir}")


def smoke_test_api(
    *,
    app: Any,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    request_specs = build_smoke_request_specs(config)
    with TestClient(app) as client:
        started = time.perf_counter()
        health_response = client.get("/health")
        health_latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        if health_response.status_code != 200:
            raise RuntimeError(f"Health endpoint returned unexpected status {health_response.status_code}.")

        health_payload = health_response.json()
        response_groups: dict[str, list[dict[str, Any]]] = {
            "fixture_replay": [],
            "contextual_scoring": [],
            "request_time_assembly": [],
        }
        request_payloads: list[dict[str, Any]] = []
        response_payloads: list[dict[str, Any]] = []
        clicked_item_groups: list[list[str]] = []
        latency_by_label: dict[str, float] = {}
        for request_spec in request_specs:
            started = time.perf_counter()
            response = client.post(
                "/score/feed",
                json=sanitize_serving_request_payload(request_spec["payload"]),
            )
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            if response.status_code != 200:
                raise RuntimeError(
                    f"{request_spec['label']} score endpoint returned unexpected status "
                    f"{response.status_code}: {response.text}"
                )
            payload = response.json()
            latency_by_label[request_spec["label"]] = latency_ms
            response_groups[payload["mode"]].append(payload)
            request_payloads.append(request_spec["payload"])
            response_payloads.append(payload)
            clicked_item_groups.append(request_spec["payload"].get("simulated_clicked_item_ids", []))

        replay_payload = response_groups["fixture_replay"][0]
        contextual_payload = response_groups["contextual_scoring"][0]
        assembled_payload = response_groups["request_time_assembly"][0]
        interaction_logs = build_serving_interaction_logs(
            api_name=replay_payload["api_name"],
            request_payloads=request_payloads,
            response_payloads=response_payloads,
            simulated_clicked_item_ids=clicked_item_groups,
        )
        summary = {
            "api_name": replay_payload["api_name"],
            "health_status": health_payload["status"],
            "overall_status": health_payload.get("overall_status", "pass"),
            "supported_modes": health_payload["supported_modes"],
            "replay_request_id": replay_payload["request_id"],
            "replay_top_item_id": replay_payload["items"][0]["item_id"] if replay_payload["items"] else None,
            "experiment_id": replay_payload.get("experiment_id"),
            "replay_treatment_id": replay_payload.get("treatment_id"),
            "contextual_request_id": contextual_payload["request_id"],
            "contextual_top_item_id": (
                contextual_payload["items"][0]["item_id"] if contextual_payload["items"] else None
            ),
            "contextual_treatment_id": contextual_payload.get("treatment_id"),
            "contextual_degraded_modes": contextual_payload.get("degraded_modes", []),
            "assembled_request_id": assembled_payload["request_id"],
            "assembled_top_item_id": (
                assembled_payload["items"][0]["item_id"] if assembled_payload["items"] else None
            ),
            "assembled_treatment_id": assembled_payload.get("treatment_id"),
            "assembled_degraded_modes": assembled_payload.get("degraded_modes", []),
            "top_item_id": replay_payload["items"][0]["item_id"] if replay_payload["items"] else None,
            "request_id": replay_payload["request_id"],
            "user_id": replay_payload["user_id"],
            "dataset_split": replay_payload["dataset_split"],
            "returned_item_count": replay_payload["returned_item_count"],
            "contextual_returned_item_count": contextual_payload["returned_item_count"],
            "assembled_returned_item_count": assembled_payload["returned_item_count"],
            "source_rerank_dir": replay_payload["source_rerank_dir"],
            "smoke_request_count": len(request_specs),
            "smoke_request_results": build_smoke_request_results(response_groups),
            "observability": {
                "endpoint_latency_ms": {"health": health_latency_ms} | latency_by_label,
                "health_checks": health_payload.get("health_checks", []),
                "degraded_modes": health_payload.get("degraded_modes", []),
                "serving_state": health_payload.get("serving_state", {}),
                "response_degraded_modes": build_response_degraded_modes(response_groups),
            },
            "serving_contract": health_payload.get("feature_contract", {}),
            "online_event_log_summary": build_event_log_summary(interaction_logs),
        }
        return summary, {
            "replay_response": replay_payload,
            "contextual_response": contextual_payload,
            "assembled_response": assembled_payload,
            "responses_by_mode": response_groups,
        }, app.openapi(), interaction_logs


def sanitize_serving_request_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "simulated_clicked_item_ids"}


def build_smoke_request_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    request_specs: list[dict[str, Any]] = []
    request_groups = [
        ("replay", "smoke_request", "smoke_requests"),
        ("contextual", "contextual_smoke_request", "contextual_smoke_requests"),
        ("assembled", "assembled_smoke_request", "assembled_smoke_requests"),
    ]
    for label_prefix, singular_key, plural_key in request_groups:
        if plural_key in config:
            payloads = config[plural_key]
        elif singular_key in config:
            payloads = [config[singular_key]]
        else:
            payloads = []
        for index, payload in enumerate(payloads, start=1):
            request_specs.append(
                {
                    "label": f"{label_prefix}_{index}",
                    "payload": payload,
                }
            )
    if not request_specs:
        raise ValueError("Local API smoke config must provide at least one smoke request.")
    return request_specs


def build_smoke_request_results(response_groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for responses in response_groups.values():
        for payload in responses:
            rows.append(
                {
                    "request_id": payload["request_id"],
                    "mode": payload["mode"],
                    "user_id": payload["user_id"],
                    "treatment_id": payload.get("treatment_id"),
                    "returned_item_count": payload.get("returned_item_count", 0),
                    "top_item_id": payload["items"][0]["item_id"] if payload.get("items") else None,
                    "degraded_modes": payload.get("degraded_modes", []),
                }
            )
    return rows


def build_response_degraded_modes(response_groups: dict[str, list[dict[str, Any]]]) -> dict[str, list[list[str]]]:
    return {
        mode: [payload.get("degraded_modes", []) for payload in payloads]
        for mode, payloads in response_groups.items()
        if payloads
    }


if __name__ == "__main__":
    main()
