from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_SAMPLES = 500
DEFAULT_RANDOM_SEED = 42


def resolve_uncertainty_config(config: dict[str, Any]) -> dict[str, Any]:
    uncertainty = config.get("uncertainty", {})
    return {
        "confidence_level": float(uncertainty.get("confidence_level", DEFAULT_CONFIDENCE_LEVEL)),
        "bootstrap_samples": int(uncertainty.get("bootstrap_samples", DEFAULT_BOOTSTRAP_SAMPLES)),
        "random_seed": int(uncertainty.get("random_seed", DEFAULT_RANDOM_SEED)),
    }


def summarize_mean_metric(
    values: list[float],
    *,
    metric_name: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    uncertainty_config = resolve_uncertainty_config(config)
    array = np.asarray(values, dtype=float)
    point_estimate = float(array.mean()) if len(array) else 0.0
    lower, upper = bootstrap_mean_interval(
        array,
        bootstrap_samples=uncertainty_config["bootstrap_samples"],
        confidence_level=uncertainty_config["confidence_level"],
        random_seed=uncertainty_config["random_seed"],
    )
    return {
        "metric_name": metric_name,
        "sample_size": int(len(array)),
        "point_estimate": point_estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence_level": uncertainty_config["confidence_level"],
        "bootstrap_samples": uncertainty_config["bootstrap_samples"],
        "random_seed": uncertainty_config["random_seed"],
        "method": "bootstrap_mean",
    }


def summarize_mean_delta(
    candidate_values: list[float],
    baseline_values: list[float],
    *,
    metric_name: str,
    config: dict[str, Any],
    paired: bool = False,
) -> dict[str, Any]:
    uncertainty_config = resolve_uncertainty_config(config)
    candidate_array = np.asarray(candidate_values, dtype=float)
    baseline_array = np.asarray(baseline_values, dtype=float)
    point_estimate = (
        float(candidate_array.mean() - baseline_array.mean())
        if len(candidate_array) and len(baseline_array)
        else 0.0
    )
    lower, upper = bootstrap_mean_delta_interval(
        candidate_array,
        baseline_array,
        bootstrap_samples=uncertainty_config["bootstrap_samples"],
        confidence_level=uncertainty_config["confidence_level"],
        random_seed=uncertainty_config["random_seed"],
        paired=paired,
    )
    return {
        "metric_name": metric_name,
        "candidate_sample_size": int(len(candidate_array)),
        "baseline_sample_size": int(len(baseline_array)),
        "point_estimate": point_estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence_level": uncertainty_config["confidence_level"],
        "bootstrap_samples": uncertainty_config["bootstrap_samples"],
        "random_seed": uncertainty_config["random_seed"],
        "method": "paired_bootstrap_mean_delta" if paired else "bootstrap_mean_delta",
    }


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    bootstrap_samples: int,
    confidence_level: float,
    random_seed: int,
) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) == 1:
        point = float(values[0])
        return point, point

    rng = np.random.default_rng(random_seed)
    means = np.empty(bootstrap_samples, dtype=float)
    for index in range(bootstrap_samples):
        sampled = rng.choice(values, size=len(values), replace=True)
        means[index] = sampled.mean()
    alpha = 1.0 - confidence_level
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def bootstrap_mean_delta_interval(
    candidate_values: np.ndarray,
    baseline_values: np.ndarray,
    *,
    bootstrap_samples: int,
    confidence_level: float,
    random_seed: int,
    paired: bool,
) -> tuple[float, float]:
    if len(candidate_values) == 0 or len(baseline_values) == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(random_seed)
    deltas = np.empty(bootstrap_samples, dtype=float)
    if paired:
        if len(candidate_values) != len(baseline_values):
            raise ValueError("Paired bootstrap delta requires equal-length candidate and baseline values.")
        differences = candidate_values - baseline_values
        if len(differences) == 1:
            point = float(differences[0])
            return point, point
        for index in range(bootstrap_samples):
            sampled = rng.choice(differences, size=len(differences), replace=True)
            deltas[index] = sampled.mean()
    else:
        if len(candidate_values) == 1 and len(baseline_values) == 1:
            point = float(candidate_values[0] - baseline_values[0])
            return point, point
        for index in range(bootstrap_samples):
            candidate_sample = rng.choice(candidate_values, size=len(candidate_values), replace=True)
            baseline_sample = rng.choice(baseline_values, size=len(baseline_values), replace=True)
            deltas[index] = candidate_sample.mean() - baseline_sample.mean()
    alpha = 1.0 - confidence_level
    return float(np.quantile(deltas, alpha / 2.0)), float(np.quantile(deltas, 1.0 - alpha / 2.0))
