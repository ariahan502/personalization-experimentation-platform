from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from personalization_platform.retrieval.common import load_event_log_inputs

CANDIDATE_REQUIRED_COLUMNS = {
    "request_id",
    "user_id",
    "item_id",
    "candidate_source",
    "merged_rank",
    "merged_score",
    "source_rank",
    "topic",
    "source_count",
    "source_list",
    "source_details",
}


def build_ranking_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    event_log_inputs = load_event_log_inputs(config)
    candidates_dir = resolve_candidates_dir(config)
    candidates = pd.read_csv(candidates_dir / "candidates.csv")
    validate_candidates_frame(candidates, candidates_dir=candidates_dir)

    requests = event_log_inputs["requests"].copy()
    impressions = event_log_inputs["impressions"].copy()
    user_state = event_log_inputs["user_state"].copy()

    requests["request_ts"] = pd.to_datetime(requests["request_ts"])
    impression_labels = impressions[["request_id", "item_id", "clicked", "position"]].rename(
        columns={"clicked": "label", "position": "observed_position"}
    )

    dataset = candidates.merge(impression_labels, on=["request_id", "item_id"], how="left")
    dataset["label"] = dataset["label"].fillna(0).astype(int)
    dataset["observed_position"] = dataset["observed_position"].fillna(-1).astype(int)

    request_features = requests[
        [
            "request_id",
            "user_id",
            "session_id",
            "request_ts",
            "split",
            "candidate_count",
            "history_length",
            "request_index_in_session",
        ]
    ]
    dataset = dataset.merge(request_features, on=["request_id", "user_id"], how="left")
    dataset = dataset.merge(
        user_state[
            ["request_id", "history_click_count", "is_cold_start", "recent_topic_counts"]
        ],
        on="request_id",
        how="left",
    )
    validate_join_completeness(dataset)
    dataset["event_log_input_dir"] = str(event_log_inputs["event_log_dir"])

    dataset["source_list_parsed"] = dataset["source_list"].map(json.loads)
    dataset["recent_topic_counts_parsed"] = dataset["recent_topic_counts"].map(json.loads)
    dataset["has_affinity_source"] = dataset["source_list_parsed"].map(
        lambda sources: int("affinity" in sources)
    )
    dataset["has_trending_source"] = dataset["source_list_parsed"].map(
        lambda sources: int("trending" in sources)
    )
    dataset["is_affinity_primary"] = (dataset["candidate_source"] == "affinity").astype(int)
    dataset["is_trending_primary"] = (dataset["candidate_source"] == "trending").astype(int)
    dataset["has_multi_source_provenance"] = (dataset["source_count"] > 1).astype(int)
    dataset["topic_history_count"] = dataset.apply(
        lambda row: int(row["recent_topic_counts_parsed"].get(row["topic"], 0)),
        axis=1,
    )
    dataset["normalized_merged_rank"] = dataset["merged_rank"] / dataset["candidate_count"].clip(lower=1)
    dataset["candidate_seen_in_impressions"] = (dataset["observed_position"] > 0).astype(int)
    dataset["request_hour"] = dataset["request_ts"].dt.hour

    dataset["dataset_split"] = assign_dataset_splits(dataset=dataset, config=config)

    dataset = dataset.sort_values(["request_ts", "request_id", "merged_rank", "item_id"]).reset_index(drop=True)
    dataset["request_ts"] = dataset["request_ts"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    output_columns = [
        "request_id",
        "user_id",
        "session_id",
        "item_id",
        "topic",
        "label",
        "dataset_split",
        "merged_rank",
        "normalized_merged_rank",
        "merged_score",
        "candidate_source",
        "source_rank",
        "source_count",
        "has_multi_source_provenance",
        "has_affinity_source",
        "has_trending_source",
        "is_affinity_primary",
        "is_trending_primary",
        "candidate_count",
        "history_length",
        "history_click_count",
        "is_cold_start",
        "topic_history_count",
        "request_index_in_session",
        "request_hour",
        "candidate_seen_in_impressions",
        "observed_position",
        "source_list",
        "source_details",
        "request_ts",
        "split",
        "event_log_input_dir",
    ]
    dataset = dataset[output_columns]

    metrics = build_ranking_dataset_metrics(dataset=dataset, candidates_dir=candidates_dir)
    manifest = build_ranking_dataset_manifest(config=config, metrics=metrics)
    return dataset, metrics, manifest


def resolve_candidates_dir(config: dict[str, Any]) -> Path:
    ranking_input = config["input"]
    base_dir = Path(ranking_input["candidates_base_dir"])
    run_name = ranking_input["candidates_run_name"]
    matches = sorted(base_dir.glob(f"*_{run_name}"))
    if not matches:
        raise FileNotFoundError(
            f"No candidate outputs found under {base_dir} matching '*_{run_name}'."
        )
    return matches[-1]


def validate_candidates_frame(candidates: pd.DataFrame, *, candidates_dir: Path) -> None:
    if candidates.empty:
        raise ValueError(
            f"Candidate input at {candidates_dir} contains zero rows; ranking dataset build requires at least one candidate row."
        )
    missing_columns = sorted(CANDIDATE_REQUIRED_COLUMNS - set(candidates.columns))
    if missing_columns:
        raise ValueError(
            f"Candidate input at {candidates_dir} is missing required columns: {', '.join(missing_columns)}."
        )


def validate_join_completeness(dataset: pd.DataFrame) -> None:
    missing_request_features = dataset["request_ts"].isna() | dataset["history_click_count"].isna()
    if missing_request_features.any():
        missing_request_ids = sorted(dataset.loc[missing_request_features, "request_id"].astype(str).unique().tolist())
        raise ValueError(
            "Ranking dataset build found candidate rows without matching request or user-state context for request_ids: "
            + ", ".join(missing_request_ids[:5])
            + ("..." if len(missing_request_ids) > 5 else "")
            + "."
        )


def assign_dataset_splits(*, dataset: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    split_config = config.get("split", {})
    strategy = split_config.get("strategy", "latest_timestamp_bucket")
    if dataset.empty:
        raise ValueError("Ranking dataset is empty before split assignment; verify candidate generation outputs.")

    if strategy == "latest_timestamp_bucket":
        unique_request_ts = sorted(dataset["request_ts"].unique())
        valid_cutoff = unique_request_ts[-1]
        return dataset["request_ts"].map(lambda ts: "valid" if ts == valid_cutoff else "train")

    if strategy == "tail_request_count":
        valid_request_count = int(split_config.get("valid_request_count", 1))
        if valid_request_count <= 0:
            raise ValueError("Config field 'split.valid_request_count' must be positive for tail_request_count.")

        request_order = (
            dataset[["request_id", "request_ts"]]
            .drop_duplicates()
            .sort_values(["request_ts", "request_id"])
            .reset_index(drop=True)
        )
        if len(request_order) <= valid_request_count:
            raise ValueError("tail_request_count must leave at least one training request.")
        valid_request_ids = set(request_order.tail(valid_request_count)["request_id"].tolist())
        return dataset["request_id"].map(lambda request_id: "valid" if request_id in valid_request_ids else "train")

    raise ValueError(f"Unsupported ranking dataset split strategy '{strategy}'.")


def build_ranking_dataset_metrics(*, dataset: pd.DataFrame, candidates_dir: Path) -> dict[str, Any]:
    event_log_input_dir = (
        str(dataset["event_log_input_dir"].iloc[0]) if not dataset.empty and "event_log_input_dir" in dataset.columns else ""
    )
    split_counts = dataset["dataset_split"].value_counts().to_dict()
    split_request_counts = dataset.groupby("dataset_split")["request_id"].nunique().to_dict()
    label_rate = float(dataset["label"].mean()) if not dataset.empty else 0.0
    return {
        "dataset_name": "ranking_dataset",
        "candidate_input_dir": str(candidates_dir),
        "event_log_input_dir": event_log_input_dir,
        "row_count": int(len(dataset)),
        "request_count": int(dataset["request_id"].nunique()),
        "item_count": int(dataset["item_id"].nunique()),
        "positive_labels": int(dataset["label"].sum()),
        "positive_rate": label_rate,
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "split_request_counts": {key: int(value) for key, value in split_request_counts.items()},
        "primary_source_counts": {
            key: int(value) for key, value in dataset["candidate_source"].value_counts().to_dict().items()
        },
        "multi_source_rate": (
            float(dataset["has_multi_source_provenance"].mean()) if not dataset.empty else 0.0
        ),
        "cold_start_row_count": int(dataset["is_cold_start"].sum()),
    }


def build_ranking_dataset_manifest(*, config: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    split_config = config.get("split", {})
    strategy = split_config.get("strategy", "latest_timestamp_bucket")
    if strategy == "tail_request_count":
        split_logic = (
            f"The last {int(split_config.get('valid_request_count', 1))} requests in time order are assigned to validation; "
            "earlier requests are train."
        )
    else:
        split_logic = (
            "The most recent request timestamp bucket in the smoke dataset is assigned to validation; earlier rows are train."
        )
    return {
        "dataset_name": "ranking_dataset",
        "row_grain": "One row per request-item candidate pair.",
        "candidate_input_dir": metrics["candidate_input_dir"],
        "label_definition": "Binary click label joined from impressions; non-clicked or unserved candidates receive label 0.",
        "split_logic": split_logic,
        "feature_groups": {
            "candidate_ordering": ["merged_rank", "normalized_merged_rank", "merged_score", "source_rank"],
            "provenance": [
                "candidate_source",
                "source_count",
                "has_multi_source_provenance",
                "has_affinity_source",
                "has_trending_source",
                "is_affinity_primary",
                "is_trending_primary",
            ],
            "request_context": [
                "candidate_count",
                "history_length",
                "history_click_count",
                "is_cold_start",
                "request_index_in_session",
                "request_hour",
            ],
            "topic_features": ["topic", "topic_history_count"],
            "debug_labels": [
                "candidate_seen_in_impressions",
                "observed_position",
                "source_list",
                "source_details",
            ],
        },
        "assumptions": [
            "The first baseline ranking dataset keeps only explainable hand-built features from retrieval provenance and request context.",
            "The split logic is time-ordered and is intended for reproducibility rather than statistical rigor.",
            "Candidates that were not part of the original impression list remain valid negative examples with label 0 in this first slice.",
        ],
        "config_snapshot": config,
    }
