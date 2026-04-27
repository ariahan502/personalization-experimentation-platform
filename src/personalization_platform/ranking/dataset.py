from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from personalization_platform.retrieval.common import load_event_log_inputs


def build_ranking_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    event_log_inputs = load_event_log_inputs(config)
    candidates_dir = resolve_candidates_dir(config)
    candidates = pd.read_csv(candidates_dir / "candidates.csv")

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

    unique_request_ts = sorted(dataset["request_ts"].unique())
    valid_cutoff = unique_request_ts[-1]
    dataset["dataset_split"] = dataset["request_ts"].map(
        lambda ts: "valid" if ts == valid_cutoff else "train"
    )

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


def build_ranking_dataset_metrics(*, dataset: pd.DataFrame, candidates_dir: Path) -> dict[str, Any]:
    split_counts = dataset["dataset_split"].value_counts().to_dict()
    label_rate = float(dataset["label"].mean()) if not dataset.empty else 0.0
    return {
        "dataset_name": "ranking_dataset",
        "candidate_input_dir": str(candidates_dir),
        "row_count": int(len(dataset)),
        "request_count": int(dataset["request_id"].nunique()),
        "item_count": int(dataset["item_id"].nunique()),
        "positive_labels": int(dataset["label"].sum()),
        "positive_rate": label_rate,
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "primary_source_counts": {
            key: int(value) for key, value in dataset["candidate_source"].value_counts().to_dict().items()
        },
        "multi_source_rate": (
            float(dataset["has_multi_source_provenance"].mean()) if not dataset.empty else 0.0
        ),
        "cold_start_row_count": int(dataset["is_cold_start"].sum()),
    }


def build_ranking_dataset_manifest(*, config: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_name": "ranking_dataset",
        "row_grain": "One row per request-item candidate pair.",
        "candidate_input_dir": metrics["candidate_input_dir"],
        "label_definition": "Binary click label joined from impressions; non-clicked or unserved candidates receive label 0.",
        "split_logic": "The most recent request timestamp bucket in the smoke dataset is assigned to validation; earlier rows are train.",
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
            "The smoke split logic is time-ordered within the tiny fixture and is intended for reproducibility rather than statistical rigor.",
            "Candidates that were not part of the original impression list remain valid negative examples with label 0 in this first slice.",
        ],
        "config_snapshot": config,
    }
