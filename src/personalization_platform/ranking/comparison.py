from __future__ import annotations

import math
from typing import Any

import pandas as pd

from personalization_platform.ranking.logistic_baseline import (
    build_request_ranking_metrics,
    train_ranker_model,
)


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
) -> dict[str, Any]:
    primary_variant_name = comparison_metrics["primary_variant_name"]
    return {
        "calibration_summary": {
            variant_name: build_calibration_summary(rows) for variant_name, rows in variant_rows.items()
        },
        "slice_summary": {
            variant_name: build_slice_summary(rows) for variant_name, rows in variant_rows.items()
        },
        "top_scored_examples": {
            variant_name: select_top_examples(rows) for variant_name, rows in variant_rows.items()
        },
        "comparison_notes": [
            "The fallback baseline uses retrieval order only, scored as inverse merged rank.",
            "The comparison bundle can score multiple model families against the same valid split and retrieval-order fallback.",
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
