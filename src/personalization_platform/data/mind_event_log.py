from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from personalization_platform.data.event_log_config import validate_event_log_config
from personalization_platform.data.event_log_schema import SCHEMA_ASSUMPTIONS


BEHAVIORS_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]
NEWS_COLUMNS = [
    "item_id",
    "topic",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]
SESSION_GAP_MINUTES = 30


def build_event_log_tables(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    validation_report = validate_event_log_config(config)
    source_paths = resolve_source_paths(config)
    for label, path in source_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required input '{label}' was not found at {path}.")

    behaviors = load_behaviors(source_paths["behaviors"], validation_report["row_limit"])
    news = load_news(source_paths["news"])
    topic_lookup = dict(zip(news["item_id"], news["topic"], strict=False))

    requests = build_requests_table(behaviors, validation_report["split"])
    impressions = build_impressions_table(behaviors, requests, topic_lookup)
    user_state = build_user_state_table(behaviors, requests, topic_lookup)
    item_state = build_item_state_table(news)

    return {
        "requests": requests,
        "impressions": impressions,
        "user_state": user_state,
        "item_state": item_state,
    }


def resolve_source_paths(config: dict[str, Any]) -> dict[str, Path]:
    input_config = config["input"]
    source_mode = input_config["source_mode"]
    source_section = config["smoke_fixture"] if source_mode == "smoke_fixture" else config["raw_input"]
    root_dir = Path(source_section["root_dir"])
    files = source_section["files"]
    return {
        "root_dir": root_dir,
        "behaviors": root_dir / files["behaviors"],
        "news": root_dir / files["news"],
    }


def load_behaviors(path: Path, row_limit: int | None) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=BEHAVIORS_COLUMNS,
        nrows=row_limit,
        keep_default_na=False,
    )
    frame["request_ts"] = pd.to_datetime(frame["time"], format="%m/%d/%Y %I:%M:%S %p")
    frame["history_item_ids"] = frame["history"].map(_parse_history)
    frame["parsed_impressions"] = frame["impressions"].map(_parse_impressions)
    frame = frame.sort_values(["user_id", "request_ts", "impression_id"]).reset_index(drop=True)
    frame["session_number"] = _derive_session_numbers(frame)
    frame["session_id"] = frame.apply(
        lambda row: f"{row['user_id']}-session-{int(row['session_number'])}", axis=1
    )
    frame["request_index_in_session"] = (
        frame.groupby("session_id").cumcount() + 1
    )
    return frame


def load_news(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=NEWS_COLUMNS,
        keep_default_na=False,
    )
    return frame


def build_requests_table(behaviors: pd.DataFrame, split: str) -> pd.DataFrame:
    requests = pd.DataFrame(
        {
            "request_id": behaviors["impression_id"].map(lambda value: f"{split}-request-{value}"),
            "user_id": behaviors["user_id"],
            "session_id": behaviors["session_id"],
            "request_ts": behaviors["request_ts"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "split": split,
            "candidate_count": behaviors["parsed_impressions"].map(len),
            "history_length": behaviors["history_item_ids"].map(len),
            "request_index_in_session": behaviors["request_index_in_session"],
        }
    )
    return requests


def build_impressions_table(
    behaviors: pd.DataFrame,
    requests: pd.DataFrame,
    topic_lookup: dict[str, str],
) -> pd.DataFrame:
    request_lookup = dict(zip(behaviors["impression_id"], requests["request_id"], strict=False))
    rows: list[dict[str, Any]] = []
    for behavior in behaviors.itertuples(index=False):
        request_id = request_lookup[behavior.impression_id]
        for position, impression in enumerate(behavior.parsed_impressions, start=1):
            item_id = impression["item_id"]
            rows.append(
                {
                    "impression_id": f"{request_id}-{item_id}",
                    "request_id": request_id,
                    "user_id": behavior.user_id,
                    "item_id": item_id,
                    "position": position,
                    "clicked": impression["clicked"],
                    "topic": topic_lookup.get(item_id, ""),
                }
            )
    return pd.DataFrame(rows)


def build_user_state_table(
    behaviors: pd.DataFrame,
    requests: pd.DataFrame,
    topic_lookup: dict[str, str],
) -> pd.DataFrame:
    request_lookup = dict(zip(behaviors["impression_id"], requests["request_id"], strict=False))
    rows: list[dict[str, Any]] = []
    for behavior in behaviors.itertuples(index=False):
        topic_counts = Counter(
            topic_lookup[item_id]
            for item_id in behavior.history_item_ids
            if item_id in topic_lookup
        )
        rows.append(
            {
                "request_id": request_lookup[behavior.impression_id],
                "user_id": behavior.user_id,
                "history_item_ids": json.dumps(behavior.history_item_ids),
                "history_click_count": len(behavior.history_item_ids),
                "is_cold_start": len(behavior.history_item_ids) == 0,
                "recent_topic_counts": json.dumps(dict(topic_counts), sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def build_item_state_table(news: pd.DataFrame) -> pd.DataFrame:
    items = news.copy()
    items["publisher"] = "unknown_publisher"
    items["published_ts"] = ""
    items["entity_ids"] = items.apply(_merge_entity_fields, axis=1)
    return items[
        [
            "item_id",
            "topic",
            "subcategory",
            "title",
            "publisher",
            "published_ts",
            "abstract",
            "entity_ids",
        ]
    ]


def build_run_metrics(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    requests = tables["requests"]
    impressions = tables["impressions"]
    user_state = tables["user_state"]
    item_state = tables["item_state"]
    return {
        "row_counts": {name: int(len(table)) for name, table in tables.items()},
        "distinct_users": int(requests["user_id"].nunique()),
        "distinct_sessions": int(requests["session_id"].nunique()),
        "distinct_items": int(item_state["item_id"].nunique()),
        "clicked_impressions": int(impressions["clicked"].sum()),
        "cold_start_requests": int(user_state["is_cold_start"].sum()),
    }


def build_manifest(
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "contract_version": "v1",
        "dataset": config["input"]["dataset"],
        "source_mode": config["input"]["source_mode"],
        "split": config["input"]["split"],
        "output_dir": str(output_dir),
        "tables": {
            table_name: {
                "path": str(output_dir / f"{table_name}.csv"),
                "format": "csv",
                "row_count": row_count,
            }
            for table_name, row_count in metrics["row_counts"].items()
        },
        "assumptions": list(SCHEMA_ASSUMPTIONS)
        + [
            "Sessions are inferred with a 30-minute inactivity gap within each user.",
            "Item publisher is set to 'unknown_publisher' in the smoke slice because the raw-like fixture does not include a source field.",
            "Published timestamps are left blank in the smoke slice because the fixture omits them.",
        ],
    }


def _parse_history(value: str) -> list[str]:
    if not value.strip():
        return []
    return value.split()


def _parse_impressions(value: str) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for token in value.split():
        item_id, clicked = token.rsplit("-", maxsplit=1)
        parsed.append({"item_id": item_id, "clicked": int(clicked)})
    return parsed


def _derive_session_numbers(frame: pd.DataFrame) -> pd.Series:
    minute_gaps = frame.groupby("user_id")["request_ts"].diff().dt.total_seconds().div(60)
    first_request_flags = frame.groupby("user_id").cumcount().eq(0)
    gap_flags = first_request_flags | minute_gaps.gt(SESSION_GAP_MINUTES).fillna(False)
    return gap_flags.groupby(frame["user_id"]).cumsum().astype(int)


def _merge_entity_fields(row: pd.Series) -> str:
    entity_ids: list[str] = []
    for column in ("title_entities", "abstract_entities"):
        raw_value = row[column]
        if raw_value and raw_value != "[]":
            entity_ids.append(raw_value)
    return json.dumps(entity_ids)
