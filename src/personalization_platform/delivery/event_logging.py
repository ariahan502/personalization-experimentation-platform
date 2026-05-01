from __future__ import annotations

from datetime import UTC, datetime
import json
from typing import Any

import pandas as pd


REQUEST_EVENT_COLUMNS = [
    "event_id",
    "event_ts",
    "api_name",
    "request_id",
    "mode",
    "user_id",
    "top_k",
    "candidate_input_count",
    "returned_item_count",
    "experiment_id",
    "assignment_unit",
    "assignment_unit_id",
    "hash_bucket",
    "treatment_id",
    "treatment_name",
    "is_control",
    "source_rerank_dir",
]

EXPOSURE_EVENT_COLUMNS = [
    "event_id",
    "event_ts",
    "request_event_id",
    "request_id",
    "api_name",
    "mode",
    "user_id",
    "experiment_id",
    "assignment_unit",
    "assignment_unit_id",
    "hash_bucket",
    "treatment_id",
    "treatment_name",
    "is_control",
    "item_id",
    "candidate_source",
    "topic",
    "creator_id",
    "pre_rank",
    "post_rank",
    "rank_shift",
    "prediction",
    "rerank_score",
    "freshness_bonus",
]

RESPONSE_EVENT_COLUMNS = [
    "event_id",
    "event_ts",
    "request_event_id",
    "request_id",
    "api_name",
    "mode",
    "user_id",
    "experiment_id",
    "assignment_unit",
    "assignment_unit_id",
    "hash_bucket",
    "treatment_id",
    "treatment_name",
    "is_control",
    "degraded_modes",
    "fallback_used",
    "status",
    "returned_item_count",
    "top_item_id",
    "source_rerank_dir",
]

CLICK_EVENT_COLUMNS = [
    "event_id",
    "event_ts",
    "request_event_id",
    "request_id",
    "api_name",
    "mode",
    "user_id",
    "experiment_id",
    "assignment_unit",
    "assignment_unit_id",
    "hash_bucket",
    "treatment_id",
    "treatment_name",
    "is_control",
    "item_id",
    "click_label",
]


def build_serving_interaction_logs(
    *,
    api_name: str,
    request_payloads: list[dict[str, Any]],
    response_payloads: list[dict[str, Any]],
    simulated_clicked_item_ids: list[list[str]] | None = None,
) -> dict[str, pd.DataFrame]:
    request_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    click_rows: list[dict[str, Any]] = []
    click_groups = simulated_clicked_item_ids or [[] for _ in response_payloads]

    for index, (request_payload, response_payload, clicked_items) in enumerate(
        zip(request_payloads, response_payloads, click_groups, strict=True),
        start=1,
    ):
        request_event_id = f"{response_payload['request_id']}-request-{index}"
        request_ts = current_event_timestamp()
        request_rows.append(
            {
                "event_id": request_event_id,
                "event_ts": request_ts,
                "api_name": api_name,
                "request_id": str(response_payload["request_id"]),
                "mode": str(response_payload["mode"]),
                "user_id": str(response_payload["user_id"]),
                "top_k": int(request_payload.get("top_k", response_payload.get("returned_item_count", 0))),
                "candidate_input_count": int(len(request_payload.get("candidate_items", []) or [])),
                "returned_item_count": int(response_payload.get("returned_item_count", 0)),
                "experiment_id": response_payload.get("experiment_id"),
                "assignment_unit": response_payload.get("assignment_unit"),
                "assignment_unit_id": response_payload.get("assignment_unit_id"),
                "hash_bucket": response_payload.get("hash_bucket"),
                "treatment_id": response_payload.get("treatment_id", "unassigned"),
                "treatment_name": response_payload.get("treatment_name"),
                "is_control": response_payload.get("is_control"),
                "degraded_modes": json.dumps(response_payload.get("degraded_modes", [])),
                "fallback_used": int("trending_only_fallback" in response_payload.get("degraded_modes", [])),
                "source_rerank_dir": str(response_payload.get("source_rerank_dir", "")),
            }
        )

        response_rows.append(
            {
                "event_id": f"{response_payload['request_id']}-response-{index}",
                "event_ts": current_event_timestamp(),
                "request_event_id": request_event_id,
                "request_id": str(response_payload["request_id"]),
                "api_name": api_name,
                "mode": str(response_payload["mode"]),
                "user_id": str(response_payload["user_id"]),
                "experiment_id": response_payload.get("experiment_id"),
                "assignment_unit": response_payload.get("assignment_unit"),
                "assignment_unit_id": response_payload.get("assignment_unit_id"),
                "hash_bucket": response_payload.get("hash_bucket"),
                "treatment_id": response_payload.get("treatment_id", "unassigned"),
                "treatment_name": response_payload.get("treatment_name"),
                "is_control": response_payload.get("is_control"),
                "degraded_modes": json.dumps(response_payload.get("degraded_modes", [])),
                "fallback_used": int("trending_only_fallback" in response_payload.get("degraded_modes", [])),
                "status": "served",
                "returned_item_count": int(response_payload.get("returned_item_count", 0)),
                "top_item_id": (
                    str(response_payload["items"][0]["item_id"]) if response_payload.get("items") else ""
                ),
                "source_rerank_dir": str(response_payload.get("source_rerank_dir", "")),
            }
        )

        for post_rank, item in enumerate(response_payload.get("items", []), start=1):
            exposure_event_id = f"{response_payload['request_id']}-exposure-{post_rank}"
            exposure_rows.append(
                {
                    "event_id": exposure_event_id,
                    "event_ts": current_event_timestamp(),
                    "request_event_id": request_event_id,
                    "request_id": str(response_payload["request_id"]),
                    "api_name": api_name,
                    "mode": str(response_payload["mode"]),
                    "user_id": str(response_payload["user_id"]),
                    "experiment_id": response_payload.get("experiment_id"),
                    "assignment_unit": response_payload.get("assignment_unit"),
                    "assignment_unit_id": response_payload.get("assignment_unit_id"),
                    "hash_bucket": response_payload.get("hash_bucket"),
                    "treatment_id": response_payload.get("treatment_id", "unassigned"),
                    "treatment_name": response_payload.get("treatment_name"),
                    "is_control": response_payload.get("is_control"),
                    "item_id": str(item["item_id"]),
                    "candidate_source": str(item["candidate_source"]),
                    "topic": str(item.get("topic", "unknown_topic")),
                    "creator_id": str(item.get("creator_id", "creator_unknown")),
                    "pre_rank": int(item["pre_rank"]),
                    "post_rank": int(item["post_rank"]),
                    "rank_shift": int(item["rank_shift"]),
                    "prediction": float(item["prediction"]),
                    "rerank_score": float(item["rerank_score"]),
                    "freshness_bonus": float(item["freshness_bonus"]),
                }
            )

        for click_index, item_id in enumerate(clicked_items, start=1):
            click_rows.append(
                {
                    "event_id": f"{response_payload['request_id']}-click-{click_index}",
                    "event_ts": current_event_timestamp(),
                    "request_event_id": request_event_id,
                    "request_id": str(response_payload["request_id"]),
                    "api_name": api_name,
                    "mode": str(response_payload["mode"]),
                    "user_id": str(response_payload["user_id"]),
                    "experiment_id": response_payload.get("experiment_id"),
                    "assignment_unit": response_payload.get("assignment_unit"),
                    "assignment_unit_id": response_payload.get("assignment_unit_id"),
                    "hash_bucket": response_payload.get("hash_bucket"),
                    "treatment_id": response_payload.get("treatment_id", "unassigned"),
                    "treatment_name": response_payload.get("treatment_name"),
                    "is_control": response_payload.get("is_control"),
                    "item_id": str(item_id),
                    "click_label": 1,
                }
            )

    return {
        "request_events": pd.DataFrame(request_rows, columns=REQUEST_EVENT_COLUMNS),
        "exposure_events": pd.DataFrame(exposure_rows, columns=EXPOSURE_EVENT_COLUMNS),
        "response_events": pd.DataFrame(response_rows, columns=RESPONSE_EVENT_COLUMNS),
        "click_events": pd.DataFrame(click_rows, columns=CLICK_EVENT_COLUMNS),
    }


def build_event_log_summary(log_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    request_events = log_frames["request_events"]
    return {
        "request_event_count": int(len(request_events)),
        "exposure_event_count": int(len(log_frames["exposure_events"])),
        "response_event_count": int(len(log_frames["response_events"])),
        "click_event_count": int(len(log_frames["click_events"])),
        "logged_modes": sorted(request_events["mode"].unique().tolist())
        if not request_events.empty
        else [],
        "treatment_request_counts": {
            str(key): int(value)
            for key, value in request_events["treatment_id"].value_counts().to_dict().items()
        }
        if not request_events.empty
        else {},
        "control_request_count": int(request_events["is_control"].fillna(0).astype(int).sum())
        if not request_events.empty
        else 0,
    }


def current_event_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
