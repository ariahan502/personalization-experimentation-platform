from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def resolve_event_log_dir(config: dict[str, Any]) -> Path:
    retrieval_input = config["input"]
    base_dir = Path(retrieval_input["event_log_base_dir"])
    run_name = retrieval_input["event_log_run_name"]
    matches = sorted(base_dir.glob(f"*_{run_name}"))
    if not matches:
        raise FileNotFoundError(
            f"No event-log outputs found under {base_dir} matching '*_{run_name}'."
        )
    return matches[-1]


def load_event_log_inputs(config: dict[str, Any]) -> dict[str, Any]:
    event_log_dir = resolve_event_log_dir(config)
    requests = pd.read_csv(event_log_dir / "requests.csv")
    impressions = pd.read_csv(event_log_dir / "impressions.csv")
    user_state = pd.read_csv(event_log_dir / "user_state.csv")
    item_state = pd.read_csv(event_log_dir / "item_state.csv")

    requests["request_ts"] = pd.to_datetime(requests["request_ts"])
    impression_request_ts = requests[["request_id", "request_ts"]].rename(
        columns={"request_ts": "request_ts_lookup"}
    )
    impressions = impressions.merge(impression_request_ts, on="request_id", how="left")
    impressions["request_ts_lookup"] = pd.to_datetime(impressions["request_ts_lookup"])

    history_lookup = {
        row.request_id: set(json.loads(row.history_item_ids))
        for row in user_state.itertuples(index=False)
    }
    topic_count_lookup = {
        row.request_id: json.loads(row.recent_topic_counts)
        for row in user_state.itertuples(index=False)
    }
    clicked_lookup = (
        impressions.loc[impressions["clicked"] == 1, ["request_id", "item_id"]]
        .groupby("request_id")["item_id"]
        .agg(list)
        .to_dict()
    )

    return {
        "event_log_dir": event_log_dir,
        "requests": requests,
        "impressions": impressions,
        "user_state": user_state,
        "item_state": item_state,
        "history_lookup": history_lookup,
        "topic_count_lookup": topic_count_lookup,
        "clicked_lookup": clicked_lookup,
    }


def get_source_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    retrieval_config = config["retrieval"]
    if "sources" in retrieval_config:
        return retrieval_config["sources"]
    source_name = retrieval_config.get("source", "trending")
    return [
        {
            "name": source_name,
            "candidate_count": retrieval_config["candidate_count"],
            "priority": 1,
        }
    ]
