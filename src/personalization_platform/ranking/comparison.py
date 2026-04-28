from __future__ import annotations

import math
from typing import Any

import pandas as pd

from personalization_platform.ranking.logistic_baseline import (
    build_request_ranking_metrics,
    train_ranker_model,
)


DEFAULT_HISTORY_SEGMENTS = [
    {"name": "cold_start", "min_history_length": 0, "max_history_length": 0},
    {"name": "short_history", "min_history_length": 1, "max_history_length": 2},
    {"name": "medium_history", "min_history_length": 3, "max_history_length": 5},
    {"name": "long_history", "min_history_length": 6, "max_history_length": None},
]


def compare_rankers(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    variant_configs = build_variant_configs(config)
    comparison_name = config.get("run_name", "ranker_compare")
    primary_variant_name = config.get("primary_variant_name", "logistic_regression_baseline")

    variant_metrics: dict[str, dict[str, Any]] = {}
    variant_rows: dict[str, pd.DataFrame] = {}
    variant_manifests: dict[str, dict[str, Any]] = {}
    dataset_input_dir: str | None = None

    for variant_name, variant_config in variant_configs.items():
        trained_metrics, scored_rows, manifest = train_ranker_model(variant_config)
        valid_rows = scored_rows.loc[scored_rows["dataset_split"] == "valid"].copy()
        dataset_input_dir = trained_metrics["ranking_dataset_input_dir"]
        variant_rows[variant_name] = valid_rows
        variant_manifests[variant_name] = manifest
        variant_metrics[variant_name] = build_variant_metrics(
            valid_rows,
            variant_name=trained_metrics["model_name"],
            dataset_input_dir=trained_metrics["ranking_dataset_input_dir"],
        )

    if dataset_input_dir is None:
        raise ValueError("No model variants were configured for comparison.")

    fallback_rows = build_retrieval_order_baseline_rows(next(iter(variant_rows.values())))
    fallback_metrics = build_variant_metrics(
        fallback_rows,
        variant_name="retrieval_order_baseline",
        dataset_input_dir=dataset_input_dir,
    )
    variant_rows["retrieval_order_baseline"] = fallback_rows
    variant_metrics["retrieval_order_baseline"] = fallback_metrics

    if primary_variant_name not in variant_metrics:
        raise ValueError(
            f"Configured primary_variant_name={primary_variant_name!r} was not produced. "
            f"Available variants: {sorted(variant_metrics)}."
        )

    comparison_metrics = {
        "comparison_name": comparison_name,
        "primary_variant_name": primary_variant_name,
        "ranking_dataset_input_dir": dataset_input_dir,
        "variants": variant_metrics,
        "metric_deltas": compute_metric_deltas(
            candidate=variant_metrics[primary_variant_name],
            baseline=fallback_metrics,
        ),
        "variant_deltas_vs_retrieval": {
            variant_name: compute_metric_deltas(candidate=metrics, baseline=fallback_metrics)
            for variant_name, metrics in variant_metrics.items()
            if variant_name != "retrieval_order_baseline"
        },
    }
    diagnostics = build_diagnostics(
        variant_rows=variant_rows,
        variant_manifests=variant_manifests,
        comparison_metrics=comparison_metrics,
        config=config,
    )
    return comparison_metrics, diagnostics


def build_variant_configs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "comparison_models" not in config:
        return {
            "logistic_regression_baseline": {
                "input": dict(config["input"]),
                "features": config["features"],
                "model": {"model_type": "logistic_regression"} | dict(config["model"]),
                "output": config.get("output", {}),
                "artifacts": config.get("artifacts", {}),
            }
        }

    variant_configs: dict[str, dict[str, Any]] = {}
    for variant_name, model_config in config["comparison_models"].items():
        variant_configs[variant_name] = {
            "input": dict(config["input"]),
            "features": config["features"],
            "model": dict(model_config),
            "output": config.get("output", {}),
            "artifacts": config.get("artifacts", {}),
        }
    return variant_configs


def build_retrieval_order_baseline_rows(rows: pd.DataFrame) -> pd.DataFrame:
    fallback = rows.copy()
    fallback["prediction"] = 1.0 / fallback["merged_rank"].astype(float)
    fallback["predicted_label"] = (fallback["merged_rank"] == 1).astype(int)
    return fallback


def build_variant_metrics(
    rows: pd.DataFrame,
    *,
    variant_name: str,
    dataset_input_dir: str,
) -> dict[str, Any]:
    return {
        "model_name": variant_name,
        "ranking_dataset_input_dir": dataset_input_dir,
        "classification_metrics": build_score_metrics(rows),
        "ranking_metrics": build_request_ranking_metrics(rows),
    }


def build_score_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    y_true = rows["label"].astype(int)
    y_score = rows["prediction"].astype(float).clip(1e-6, 1 - 1e-6)
    y_pred = rows["predicted_label"].astype(int)

    accuracy = float((y_true == y_pred).mean()) if len(rows) else 0.0
    log_loss = float(-(y_true * y_score.map(math.log) + (1 - y_true) * (1 - y_score).map(math.log)).mean())
    metrics = {
        "row_count": int(len(rows)),
        "positive_labels": int(y_true.sum()),
        "accuracy": accuracy,
        "log_loss": log_loss,
    }
    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(compute_auc(y_true.tolist(), y_score.tolist()))
    return metrics


def compute_auc(labels: list[int], scores: list[float]) -> float:
    positives = [(score, label) for score, label in zip(scores, labels, strict=True) if label == 1]
    negatives = [(score, label) for score, label in zip(scores, labels, strict=True) if label == 0]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    total = 0
    for pos_score, _ in positives:
        for neg_score, _ in negatives:
            total += 1
            if pos_score > neg_score:
                wins += 1
            elif pos_score == neg_score:
                wins += 0.5
    return wins / total


def compute_metric_deltas(*, candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    deltas: dict[str, float] = {}
    for metric_name in ("accuracy", "log_loss", "roc_auc"):
        if metric_name in candidate["classification_metrics"] and metric_name in baseline["classification_metrics"]:
            deltas[f"classification.{metric_name}"] = (
                candidate["classification_metrics"][metric_name]
                - baseline["classification_metrics"][metric_name]
            )
    for metric_name in ("mean_reciprocal_rank", "hit_rate_at_1", "hit_rate_at_3"):
        deltas[f"ranking.{metric_name}"] = (
            candidate["ranking_metrics"][metric_name] - baseline["ranking_metrics"][metric_name]
        )
    return deltas


def build_diagnostics(
    *,
    variant_rows: dict[str, pd.DataFrame],
    variant_manifests: dict[str, dict[str, Any]],
    comparison_metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    primary_variant_name = comparison_metrics["primary_variant_name"]
    return {
        "calibration_summary": {
            variant_name: build_calibration_summary(rows) for variant_name, rows in variant_rows.items()
        },
        "slice_summary": {
            variant_name: build_slice_summary(rows, config=config) for variant_name, rows in variant_rows.items()
        },
        "segment_delta_summary": build_segment_delta_summary(
            variant_rows=variant_rows,
            primary_variant_name=primary_variant_name,
            baseline_variant_name="retrieval_order_baseline",
            config=config,
        ),
        "top_scored_examples": {
            variant_name: select_top_examples(rows) for variant_name, rows in variant_rows.items()
        },
        "comparison_notes": [
            "The fallback baseline uses retrieval order only, scored as inverse merged rank.",
            "The comparison bundle can score multiple model families against the same valid split and retrieval-order fallback.",
            "Request-level segments are defined on cold-start status and history depth so ranking behavior can be inspected beyond one aggregate metric.",
            "Offline gains on a local validation slice should be treated as directional evidence, not shipment evidence.",
        ],
        "baseline_feature_manifest": variant_manifests.get("logistic_regression_baseline", {}).get(
            "top_feature_weights",
            [],
        )[:5],
        "primary_variant_manifest": variant_manifests.get(primary_variant_name, {}),
        "metric_deltas": comparison_metrics["metric_deltas"],
    }


def build_calibration_summary(rows: pd.DataFrame) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    summary_rows = rows.copy()
    bucket_labels = list(range(min(3, len(summary_rows))))
    summary_rows["bucket"] = pd.qcut(
        summary_rows["prediction"].rank(method="first"),
        q=len(bucket_labels),
        labels=bucket_labels,
        duplicates="drop",
    )
    grouped = (
        summary_rows.groupby("bucket")
        .agg(
            row_count=("label", "size"),
            avg_prediction=("prediction", "mean"),
            empirical_positive_rate=("label", "mean"),
        )
        .reset_index()
    )
    return grouped.to_dict(orient="records")


def build_slice_summary(rows: pd.DataFrame, *, config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    prepared_rows = add_history_segment_column(rows, config=config)
    return {
        "by_candidate_source": build_row_slice_summary(prepared_rows, group_column="candidate_source"),
        "by_multi_source_provenance": build_row_slice_summary(
            prepared_rows,
            group_column="has_multi_source_provenance",
        ),
        "by_cold_start_request": build_request_slice_summary(prepared_rows, group_column="is_cold_start"),
        "by_history_depth_request": build_request_slice_summary(
            prepared_rows,
            group_column="history_segment",
            ordered_labels=history_segment_names(config),
        ),
    }


def select_top_examples(rows: pd.DataFrame) -> list[dict[str, Any]]:
    columns = ["request_id", "item_id", "label", "prediction", "candidate_source", "merged_rank", "topic"]
    return rows.sort_values("prediction", ascending=False).head(5)[columns].to_dict(orient="records")


def add_history_segment_column(rows: pd.DataFrame, *, config: dict[str, Any]) -> pd.DataFrame:
    prepared = rows.copy()
    prepared["history_segment"] = prepared["history_length"].map(
        lambda history_length: assign_history_segment(int(history_length), config=config)
    )
    return prepared


def assign_history_segment(history_length: int, *, config: dict[str, Any]) -> str:
    for segment in resolve_history_segments(config):
        max_history_length = segment.get("max_history_length")
        if history_length < int(segment["min_history_length"]):
            continue
        if max_history_length is None or history_length <= int(max_history_length):
            return str(segment["name"])
    return "unmapped_history"


def resolve_history_segments(config: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics_config = config.get("diagnostics", {})
    return diagnostics_config.get("history_segments", DEFAULT_HISTORY_SEGMENTS)


def history_segment_names(config: dict[str, Any]) -> list[str]:
    return [str(segment["name"]) for segment in resolve_history_segments(config)]


def build_row_slice_summary(rows: pd.DataFrame, *, group_column: str) -> list[dict[str, Any]]:
    slice_rows: list[dict[str, Any]] = []
    for segment_value, segment_rows in rows.groupby(group_column, dropna=False):
        slice_rows.append(
            {
                "segment": normalize_segment_value(segment_value),
                "row_count": int(len(segment_rows)),
                "request_count": int(segment_rows["request_id"].nunique()),
                "avg_prediction": float(segment_rows["prediction"].mean()),
                "positive_rate": float(segment_rows["label"].mean()),
                "classification_metrics": build_score_metrics(segment_rows),
            }
        )
    return slice_rows


def build_request_slice_summary(
    rows: pd.DataFrame,
    *,
    group_column: str,
    ordered_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    grouped_summary: list[dict[str, Any]] = []
    for segment_value, segment_rows in rows.groupby(group_column, dropna=False):
        grouped_summary.append(
            {
                "segment": normalize_segment_value(segment_value),
                "row_count": int(len(segment_rows)),
                "request_count": int(segment_rows["request_id"].nunique()),
                "avg_prediction": float(segment_rows["prediction"].mean()),
                "positive_rate": float(segment_rows["label"].mean()),
                "classification_metrics": build_score_metrics(segment_rows),
                "ranking_metrics": build_request_ranking_metrics(segment_rows),
            }
        )
    if ordered_labels is None:
        return grouped_summary
    ordering = {label: index for index, label in enumerate(ordered_labels)}
    return sorted(grouped_summary, key=lambda row: ordering.get(str(row["segment"]), len(ordering)))


def build_segment_delta_summary(
    *,
    variant_rows: dict[str, pd.DataFrame],
    primary_variant_name: str,
    baseline_variant_name: str,
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    if primary_variant_name not in variant_rows or baseline_variant_name not in variant_rows:
        return {}
    primary_rows = add_history_segment_column(variant_rows[primary_variant_name], config=config)
    baseline_rows = add_history_segment_column(variant_rows[baseline_variant_name], config=config)
    return {
        "by_cold_start_request": build_request_segment_deltas(
            primary_rows,
            baseline_rows,
            group_column="is_cold_start",
        ),
        "by_history_depth_request": build_request_segment_deltas(
            primary_rows,
            baseline_rows,
            group_column="history_segment",
            ordered_labels=history_segment_names(config),
        ),
    }


def build_request_segment_deltas(
    primary_rows: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    *,
    group_column: str,
    ordered_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    shared_segments = sorted(
        set(primary_rows[group_column].dropna().tolist()) & set(baseline_rows[group_column].dropna().tolist()),
        key=lambda value: normalize_segment_value(value),
    )
    deltas: list[dict[str, Any]] = []
    for segment_value in shared_segments:
        primary_segment = primary_rows.loc[primary_rows[group_column] == segment_value]
        baseline_segment = baseline_rows.loc[baseline_rows[group_column] == segment_value]
        primary_metrics = {
            "classification_metrics": build_score_metrics(primary_segment),
            "ranking_metrics": build_request_ranking_metrics(primary_segment),
        }
        baseline_metrics = {
            "classification_metrics": build_score_metrics(baseline_segment),
            "ranking_metrics": build_request_ranking_metrics(baseline_segment),
        }
        deltas.append(
            {
                "segment": normalize_segment_value(segment_value),
                "row_count": int(len(primary_segment)),
                "request_count": int(primary_segment["request_id"].nunique()),
                "metric_deltas": compute_metric_deltas(
                    candidate=primary_metrics,
                    baseline=baseline_metrics,
                ),
            }
        )
    if ordered_labels is None:
        return deltas
    ordering = {label: index for index, label in enumerate(ordered_labels)}
    return sorted(deltas, key=lambda row: ordering.get(str(row["segment"]), len(ordering)))


def normalize_segment_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "cold_start" if value else "warm"
    return value
