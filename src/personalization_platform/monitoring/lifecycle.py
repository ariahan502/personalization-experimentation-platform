from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def analyze_model_lifecycle(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = load_lifecycle_inputs(config["input"])
    thresholds = config.get("thresholds", {})
    policy = config.get("policy", {})
    candidate_treatment_id = str(policy.get("candidate_treatment_id", "reranked_policy"))
    control_treatment_id = str(policy.get("control_treatment_id", "control"))

    candidate_model = build_candidate_model_summary(
        ranker_manifest=inputs["ranker_manifest"],
        ranker_metrics=inputs["ranker_metrics"],
        comparison_metrics=inputs["comparison_metrics"],
    )
    live_summary = inputs["live_summary"]
    monitoring_summary = inputs["monitoring_summary"]
    live_guardrails = live_summary.get("live_guardrails", {})
    sample_size_summary = live_summary.get("sample_size_summary", {})
    live_primary_metrics = live_summary.get("primary_metrics", {})
    candidate_primary_metrics = live_primary_metrics.get(candidate_treatment_id, {})
    candidate_uncertainty = candidate_primary_metrics.get("uncertainty", {})

    promotion_checks = [
        build_min_check(
            name="offline_mrr_delta_vs_retrieval",
            observed=float(inputs["comparison_metrics"]["metric_deltas"].get("ranking.mean_reciprocal_rank", 0.0)),
            threshold=float(thresholds.get("min_offline_mrr_delta", 0.0)),
            description="Primary model should not underperform retrieval-order baseline on valid-request MRR.",
        ),
        build_max_check(
            name="offline_log_loss_delta_vs_retrieval",
            observed=float(inputs["comparison_metrics"]["metric_deltas"].get("classification.log_loss", 0.0)),
            threshold=float(thresholds.get("max_offline_log_loss_delta", 0.0)),
            description="Primary model should not increase validation log loss versus retrieval order.",
        ),
        build_min_check(
            name="live_control_request_count",
            observed=float(sample_size_summary.get(control_treatment_id, {}).get("request_count", 0)),
            threshold=float(thresholds.get("min_control_request_count", 1)),
            description="Control must see enough live-style requests to support a launch comparison.",
        ),
        build_min_check(
            name="live_candidate_request_count",
            observed=float(sample_size_summary.get(candidate_treatment_id, {}).get("request_count", 0)),
            threshold=float(thresholds.get("min_candidate_request_count", 1)),
            description="Candidate policy must see enough live-style requests to support a launch comparison.",
        ),
        build_min_check(
            name="live_srm_p_value",
            observed=float(live_summary.get("srm_check", {}).get("approx_p_value", 0.0)),
            threshold=float(thresholds.get("min_live_srm_p_value", 0.05)),
            description="Serving assignment should not show strong sample-ratio mismatch before promotion.",
        ),
        build_min_check(
            name="live_top1_ctr_lift_vs_control",
            observed=float(candidate_primary_metrics.get("lift_vs_control", 0.0)),
            threshold=float(thresholds.get("min_live_top1_ctr_lift", 0.0)),
            description="Candidate policy should show non-negative top-1 CTR lift versus control before promotion.",
        ),
        build_min_check(
            name="live_top1_ctr_lift_ci_lower",
            observed=float(candidate_uncertainty.get("top1_ctr_lift_vs_control", {}).get("ci_lower", 0.0)),
            threshold=float(thresholds.get("min_live_top1_ctr_lift_ci_lower", -0.1)),
            description="Candidate policy should not have a strongly negative lower confidence bound on top-1 CTR lift.",
        ),
    ]

    rollback_checks = [
        build_equals_check(
            name="offline_monitoring_status",
            observed=str(monitoring_summary.get("overall_status", "unknown")),
            expected="pass",
            description="Offline monitoring should remain green before the candidate model stays active.",
        ),
        build_max_check(
            name="live_candidate_degraded_request_rate",
            observed=float(live_guardrails.get(candidate_treatment_id, {}).get("degraded_request_rate", 0.0)),
            threshold=float(thresholds.get("max_live_degraded_request_rate", 0.5)),
            description="Serving degraded-mode rate should remain below the rollback threshold.",
        ),
        build_max_check(
            name="live_candidate_fallback_request_rate",
            observed=float(live_guardrails.get(candidate_treatment_id, {}).get("fallback_request_rate", 0.0)),
            threshold=float(thresholds.get("max_live_fallback_request_rate", 0.5)),
            description="Trending-only fallback usage should remain below the rollback threshold.",
        ),
    ]

    decision, decision_reasons = derive_lifecycle_decision(
        promotion_checks=promotion_checks,
        rollback_checks=rollback_checks,
    )

    summary = {
        "lifecycle_name": config.get("run_name", "model_lifecycle_smoke"),
        "decision": decision,
        "decision_reasons": decision_reasons,
        "candidate_model_name": candidate_model["model_name"],
        "candidate_model_run_id": candidate_model["run_metadata"].get("run_id", ""),
        "candidate_treatment_id": candidate_treatment_id,
        "control_treatment_id": control_treatment_id,
        "fallback_target": {
            "treatment_id": str(policy.get("fallback_treatment_id", control_treatment_id)),
            "variant_name": str(policy.get("fallback_variant_name", "retrieval_order_baseline")),
        },
        "offline_status": {
            "monitoring_status": monitoring_summary.get("overall_status", "unknown"),
            "monitoring_flagged_check_count": int(monitoring_summary.get("flagged_check_count", 0)),
            "primary_variant_name": inputs["comparison_metrics"].get("primary_variant_name", ""),
        },
        "live_status": {
            "experiment_id": live_summary.get("experiment_id", ""),
            "candidate_request_count": int(sample_size_summary.get(candidate_treatment_id, {}).get("request_count", 0)),
            "control_request_count": int(sample_size_summary.get(control_treatment_id, {}).get("request_count", 0)),
            "candidate_top1_ctr_lift_vs_control": float(candidate_primary_metrics.get("lift_vs_control", 0.0)),
            "candidate_top1_ctr_lift_ci_lower": float(
                candidate_uncertainty.get("top1_ctr_lift_vs_control", {}).get("ci_lower", 0.0)
            ),
            "candidate_degraded_request_rate": float(
                live_guardrails.get(candidate_treatment_id, {}).get("degraded_request_rate", 0.0)
            ),
            "candidate_fallback_request_rate": float(
                live_guardrails.get(candidate_treatment_id, {}).get("fallback_request_rate", 0.0)
            ),
        },
    }
    report = {
        "lifecycle_name": config.get("run_name", "model_lifecycle_smoke"),
        "decision": decision,
        "decision_reasons": decision_reasons,
        "input_dirs": inputs["input_dirs"],
        "candidate_model": candidate_model,
        "promotion_checks": promotion_checks,
        "rollback_checks": rollback_checks,
        "offline_evidence": {
            "ranker_metrics": inputs["ranker_metrics"],
            "comparison_metrics": summarize_comparison_metrics(inputs["comparison_metrics"]),
            "monitoring_summary": monitoring_summary,
        },
        "live_evidence": {
            "live_summary": live_summary,
            "candidate_guardrails": live_guardrails.get(candidate_treatment_id, {}),
            "control_guardrails": live_guardrails.get(control_treatment_id, {}),
        },
        "rollback_policy": {
            "fallback_treatment_id": str(policy.get("fallback_treatment_id", control_treatment_id)),
            "fallback_variant_name": str(policy.get("fallback_variant_name", "retrieval_order_baseline")),
            "serving_action": str(
                policy.get(
                    "serving_action",
                    "Route serving back to the control treatment and retrieval-order baseline until the candidate model is retrained.",
                )
            ),
        },
        "caveats": [
            "This lifecycle bundle combines offline smoke monitoring with local serving-log readout, not production deployment telemetry.",
            "Promotion decisions should remain conservative when live request counts are tiny or treatment coverage is one-sided.",
            "Rollback guidance is explicit here, but serving still uses local smoke wiring rather than an external deployment control plane.",
        ],
    }
    return summary, report


def load_lifecycle_inputs(input_config: dict[str, Any]) -> dict[str, Any]:
    ranker_dir = resolve_run_dir(
        input_config["ranker_base_dir"],
        input_config["ranker_run_name"],
        required_files=["manifest.json", "metrics.json"],
    )
    ranker_compare_dir = resolve_run_dir(
        input_config["ranker_compare_base_dir"],
        input_config["ranker_compare_run_name"],
        required_files=["metrics.json"],
    )
    monitoring_dir = resolve_run_dir(
        input_config["monitoring_base_dir"],
        input_config["monitoring_run_name"],
        required_files=["summary.json"],
    )
    live_experiment_dir = resolve_run_dir(
        input_config["live_experiment_base_dir"],
        input_config["live_experiment_run_name"],
        required_files=["summary.json"],
    )
    return {
        "input_dirs": {
            "ranker_dir": str(ranker_dir),
            "ranker_compare_dir": str(ranker_compare_dir),
            "monitoring_dir": str(monitoring_dir),
            "live_experiment_dir": str(live_experiment_dir),
        },
        "ranker_manifest": read_json(ranker_dir / "manifest.json"),
        "ranker_metrics": read_json(ranker_dir / "metrics.json"),
        "comparison_metrics": read_json(ranker_compare_dir / "metrics.json"),
        "monitoring_summary": read_json(monitoring_dir / "summary.json"),
        "live_summary": read_json(live_experiment_dir / "summary.json"),
    }


def resolve_run_dir(base_dir: str, run_name: str, *, required_files: list[str]) -> Path:
    matches = [
        path
        for path in sorted(Path(base_dir).glob(f"*_{run_name}"))
        if all((path / filename).exists() for filename in required_files)
    ]
    if not matches:
        raise FileNotFoundError(f"No completed outputs found under {base_dir} matching '*_{run_name}'.")
    return matches[-1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_candidate_model_summary(
    *,
    ranker_manifest: dict[str, Any],
    ranker_metrics: dict[str, Any],
    comparison_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": str(ranker_manifest.get("model_name", ranker_metrics.get("model_name", "unknown_model"))),
        "model_type": str(ranker_manifest.get("model_type", "unknown_model_type")),
        "ranking_dataset_input_dir": str(ranker_manifest.get("ranking_dataset_input_dir", "")),
        "run_metadata": ranker_manifest.get("run_metadata", {}),
        "upstream_runs": ranker_manifest.get("upstream_runs", []),
        "feature_spec": ranker_manifest.get("feature_spec", {}),
        "feature_count_after_vectorization": int(ranker_manifest.get("feature_count_after_vectorization", 0)),
        "primary_variant_name": str(comparison_metrics.get("primary_variant_name", "")),
        "valid_metrics": ranker_metrics.get("valid_metrics", {}),
        "valid_ranking_metrics": ranker_metrics.get("valid_ranking_metrics", {}),
    }


def summarize_comparison_metrics(comparison_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "comparison_name": comparison_metrics.get("comparison_name", ""),
        "primary_variant_name": comparison_metrics.get("primary_variant_name", ""),
        "metric_deltas": comparison_metrics.get("metric_deltas", {}),
        "metric_delta_uncertainty": comparison_metrics.get("metric_delta_uncertainty", {}),
    }


def derive_lifecycle_decision(
    *,
    promotion_checks: list[dict[str, Any]],
    rollback_checks: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    failed_rollback_checks = [check for check in rollback_checks if check["status"] != "pass"]
    if failed_rollback_checks:
        return "rollback", [format_reason(check) for check in failed_rollback_checks]
    failed_promotion_checks = [check for check in promotion_checks if check["status"] != "pass"]
    if failed_promotion_checks:
        return "hold", [format_reason(check) for check in failed_promotion_checks]
    return "promote", ["Offline monitoring is green, offline deltas are acceptable, and live guardrails meet promotion thresholds."]


def format_reason(check: dict[str, Any]) -> str:
    return (
        f"{check['name']} {check['comparison']} {check['threshold']} failed "
        f"(observed={check['observed']})."
    )


def build_min_check(*, name: str, observed: float, threshold: float, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if observed >= threshold else "fail",
        "comparison": ">=",
        "observed": observed,
        "threshold": threshold,
        "description": description,
    }


def build_max_check(*, name: str, observed: float, threshold: float, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if observed <= threshold else "fail",
        "comparison": "<=",
        "observed": observed,
        "threshold": threshold,
        "description": description,
    }


def build_equals_check(*, name: str, observed: str, expected: str, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if observed == expected else "fail",
        "comparison": "==",
        "observed": observed,
        "threshold": expected,
        "description": description,
    }
