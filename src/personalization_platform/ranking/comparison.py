from __future__ import annotations

import math
from typing import Any

import pandas as pd

from personalization_platform.ranking.logistic_baseline import (
    build_request_ranking_metrics,
    train_logistic_baseline,
)


def compare_rankers(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_config = build_baseline_config(config)
    baseline_metrics, scored_rows, baseline_manifest = train_logistic_baseline(baseline_config)
    baseline_rows = scored_rows.loc[scored_rows["dataset_split"] == "valid"].copy()

    fallback_rows = build_retrieval_order_baseline_rows(baseline_rows)
    logistic_metrics = build_variant_metrics(
        baseline_rows,
        variant_name="logistic_regression_baseline",
        dataset_input_dir=baseline_metrics["ranking_dataset_input_dir"],
    )
    fallback_metrics = build_variant_metrics(
        fallback_rows,
        variant_name="retrieval_order_baseline",
        dataset_input_dir=baseline_metrics["ranking_dataset_input_dir"],
    )

    comparison_metrics = {
        "comparison_name": "ranker_compare_smoke",
        "ranking_dataset_input_dir": logistic_metrics["ranking_dataset_input_dir"],
        "variants": {
            "logistic_regression_baseline": logistic_metrics,
            "retrieval_order_baseline": fallback_metrics,
        },
        "metric_deltas": compute_metric_deltas(
            candidate=logistic_metrics,
            baseline=fallback_metrics,
        ),
    }
    diagnostics = build_diagnostics(
        logistic_rows=baseline_rows,
        fallback_rows=fallback_rows,
        baseline_manifest=baseline_manifest,
        comparison_metrics=comparison_metrics,
    )
    return comparison_metrics, diagnostics


def build_baseline_config(config: dict[str, Any]) -> dict[str, Any]:
    baseline_config = {
        "input": dict(config["input"]),
        "features": config["features"],
        "model": config["model"],
        "output": config.get("output", {}),
        "artifacts": config.get("artifacts", {}),
    }
    return baseline_config


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
    logistic_rows: pd.DataFrame,
    fallback_rows: pd.DataFrame,
    baseline_manifest: dict[str, Any],
    comparison_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "calibration_summary": {
            "logistic_regression_baseline": build_calibration_summary(logistic_rows),
            "retrieval_order_baseline": build_calibration_summary(fallback_rows),
        },
        "slice_summary": {
            "logistic_regression_baseline": build_slice_summary(logistic_rows),
            "retrieval_order_baseline": build_slice_summary(fallback_rows),
        },
        "top_scored_examples": {
            "logistic_regression_baseline": select_top_examples(logistic_rows),
            "retrieval_order_baseline": select_top_examples(fallback_rows),
        },
        "comparison_notes": [
            "The fallback baseline uses retrieval order only, scored as inverse merged rank.",
            "Smoke comparison is intended to verify the evaluation bundle shape rather than establish a reliable model winner.",
            "Offline gains on this tiny fixture should be treated as plumbing checks, not shipment evidence.",
        ],
        "baseline_feature_manifest": baseline_manifest["top_feature_weights"][:5],
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


def build_slice_summary(rows: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    by_source = (
        rows.groupby("candidate_source")
        .agg(
            row_count=("label", "size"),
            avg_prediction=("prediction", "mean"),
            positive_rate=("label", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    by_cold_start = (
        rows.groupby("is_cold_start")
        .agg(
            row_count=("label", "size"),
            avg_prediction=("prediction", "mean"),
            positive_rate=("label", "mean"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "by_candidate_source": by_source,
        "by_cold_start": by_cold_start,
    }


def select_top_examples(rows: pd.DataFrame) -> list[dict[str, Any]]:
    columns = ["request_id", "item_id", "label", "prediction", "candidate_source", "merged_rank", "topic"]
    return rows.sort_values("prediction", ascending=False).head(5)[columns].to_dict(orient="records")
