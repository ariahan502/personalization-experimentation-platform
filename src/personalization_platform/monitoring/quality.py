from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def analyze_monitoring(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = load_monitoring_inputs(config["input"])
    thresholds = config["thresholds"]

    funnel = build_stage_funnel(inputs)
    event_log_summary = build_event_log_summary(inputs["requests"], inputs["impressions"])
    candidate_summary = build_candidate_summary(inputs["candidates"], inputs["ranking_dataset"])
    score_summary = build_score_summary(inputs["scored_rows"])
    rerank_summary = build_rerank_summary(inputs["reranked_rows"])
    experiment_summary = build_experiment_summary(
        inputs["experiment_summary"],
        inputs["experiment_readout"],
    )

    checks = [
        build_min_check(
            name="candidate_request_coverage",
            observed=funnel["candidate_request_coverage"],
            threshold=float(thresholds["min_candidate_request_coverage"]),
            description="Share of event-log requests that survive into the candidate stage.",
        ),
        build_max_check(
            name="candidate_source_mix_gap",
            observed=candidate_summary["split_candidate_source_mix"]["max_share_gap"],
            threshold=float(thresholds["max_candidate_source_mix_gap"]),
            description="Largest absolute train-valid share gap across candidate sources.",
        ),
        build_max_check(
            name="mean_prediction_gap",
            observed=score_summary["split_score_stability"]["mean_prediction_gap"],
            threshold=float(thresholds["max_mean_prediction_gap"]),
            description="Absolute train-valid gap in average model score.",
        ),
        build_max_check(
            name="average_rank_shift",
            observed=rerank_summary["average_absolute_rank_shift"],
            threshold=float(thresholds["max_average_rank_shift"]),
            description="Average absolute shift introduced by reranking.",
        ),
        build_min_check(
            name="srm_p_value",
            observed=experiment_summary["srm_check"]["approx_p_value"],
            threshold=float(thresholds["min_srm_p_value"]),
            description="Approximate SRM p-value from the experiment readout.",
        ),
    ]

    flagged_checks = [check for check in checks if check["status"] != "pass"]
    observability = build_observability_summary(inputs=inputs, checks=checks, flagged_checks=flagged_checks)
    summary = {
        "monitoring_name": config.get("run_name", "monitoring_smoke"),
        "overall_status": "warn" if flagged_checks else "pass",
        "flagged_check_count": len(flagged_checks),
        "checks": checks,
        "observability": observability["headline"],
        "stage_funnel": funnel,
        "event_log": event_log_summary["headline"],
        "candidate_quality": candidate_summary["headline"],
        "score_stability": score_summary["headline"],
        "rerank_activity": rerank_summary["headline"],
        "experiment_integrity": experiment_summary["headline"],
    }
    diagnostics = {
        "monitoring_name": config.get("run_name", "monitoring_smoke"),
        "input_dirs": inputs["input_dirs"],
        "funnel": funnel,
        "event_log": event_log_summary,
        "candidate_quality": candidate_summary,
        "score_stability": score_summary,
        "rerank_activity": rerank_summary,
        "experiment_integrity": experiment_summary,
        "checks": checks,
        "observability": observability,
        "caveats": [
            "This monitoring bundle is an offline smoke-quality view, not a production monitoring system.",
            "Train-valid drift on the tiny fixture is useful for plumbing checks but not for operational thresholds in a live feed.",
            "Experiment integrity checks summarize simulated assignment outputs and should be treated as reproducibility checks rather than launch evidence.",
        ],
    }
    return summary, diagnostics


def load_monitoring_inputs(input_config: dict[str, Any]) -> dict[str, Any]:
    event_log_dir = resolve_run_dir(input_config["event_log_base_dir"], input_config["event_log_run_name"])
    candidate_dir = resolve_run_dir(input_config["candidate_base_dir"], input_config["candidate_run_name"])
    ranking_dataset_dir = resolve_run_dir(
        input_config["ranking_dataset_base_dir"],
        input_config["ranking_dataset_run_name"],
    )
    ranker_dir = resolve_run_dir(input_config["ranker_base_dir"], input_config["ranker_run_name"])
    rerank_dir = resolve_run_dir(input_config["rerank_base_dir"], input_config["rerank_run_name"])
    experiment_analysis_dir = resolve_run_dir(
        input_config["experiment_analysis_base_dir"],
        input_config["experiment_analysis_run_name"],
    )

    return {
        "input_dirs": {
            "event_log_dir": str(event_log_dir),
            "candidate_dir": str(candidate_dir),
            "ranking_dataset_dir": str(ranking_dataset_dir),
            "ranker_dir": str(ranker_dir),
            "rerank_dir": str(rerank_dir),
            "experiment_analysis_dir": str(experiment_analysis_dir),
        },
        "requests": pd.read_csv(event_log_dir / "requests.csv"),
        "impressions": pd.read_csv(event_log_dir / "impressions.csv"),
        "candidates": pd.read_csv(candidate_dir / "candidates.csv"),
        "ranking_dataset": pd.read_csv(ranking_dataset_dir / "ranking_dataset.csv"),
        "scored_rows": pd.read_csv(ranker_dir / "scored_rows.csv"),
        "reranked_rows": pd.read_csv(rerank_dir / "reranked_rows.csv"),
        "experiment_summary": read_json(experiment_analysis_dir / "summary.json"),
        "experiment_readout": read_json(experiment_analysis_dir / "readout.json"),
    }


def resolve_run_dir(base_dir: str, run_name: str) -> Path:
    matches = sorted(Path(base_dir).glob(f"*_{run_name}"))
    if not matches:
        raise FileNotFoundError(f"No outputs found under {base_dir} matching '*_{run_name}'.")
    return matches[-1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_stage_funnel(inputs: dict[str, Any]) -> dict[str, Any]:
    event_log_requests = int(inputs["requests"]["request_id"].nunique())
    candidate_requests = int(inputs["candidates"]["request_id"].nunique())
    ranking_dataset_requests = int(inputs["ranking_dataset"]["request_id"].nunique())
    rerank_requests = int(inputs["reranked_rows"]["request_id"].nunique())
    experiment_requests = int(
        sum(
            summary["request_count"]
            for summary in inputs["experiment_summary"]["treatment_summaries"].values()
        )
    )
    coverage = candidate_requests / event_log_requests if event_log_requests else 0.0
    return {
        "event_log_request_count": event_log_requests,
        "candidate_request_count": candidate_requests,
        "ranking_dataset_request_count": ranking_dataset_requests,
        "rerank_request_count": rerank_requests,
        "experiment_request_count": experiment_requests,
        "candidate_request_coverage": coverage,
        "request_dropoff_from_event_log": event_log_requests - candidate_requests,
    }


def build_event_log_summary(requests: pd.DataFrame, impressions: pd.DataFrame) -> dict[str, Any]:
    cold_start_rate = float((requests["history_length"] == 0).mean()) if len(requests) else 0.0
    split_counts = requests["split"].value_counts().to_dict()
    session_count = int(requests["session_id"].nunique())
    topic_mix = impressions["topic"].value_counts(normalize=True).round(4).to_dict()
    return {
        "headline": {
            "request_count": int(len(requests)),
            "impression_count": int(len(impressions)),
            "cold_start_rate": cold_start_rate,
            "session_count": session_count,
        },
        "split_request_counts": {str(key): int(value) for key, value in split_counts.items()},
        "mean_history_length_by_split": {
            str(key): float(value)
            for key, value in requests.groupby("split")["history_length"].mean().to_dict().items()
        },
        "top_impression_topics": top_share_dict(topic_mix, limit=5),
    }


def build_observability_summary(
    *,
    inputs: dict[str, Any],
    checks: list[dict[str, Any]],
    flagged_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    input_dir_status = [
        {
            "name": name,
            "path": path,
            "status": "present" if Path(path).exists() else "missing",
        }
        for name, path in inputs["input_dirs"].items()
    ]
    degraded_stages = [check["name"] for check in flagged_checks]
    return {
        "headline": {
            "input_dir_count": len(input_dir_status),
            "present_input_dir_count": sum(1 for row in input_dir_status if row["status"] == "present"),
            "check_pass_rate": (len(checks) - len(flagged_checks)) / len(checks) if checks else 1.0,
            "degraded_stage_count": len(degraded_stages),
        },
        "input_dir_status": input_dir_status,
        "degraded_stages": degraded_stages,
        "stage_row_counts": {
            "event_log_requests": int(inputs["requests"]["request_id"].nunique()),
            "candidate_rows": int(len(inputs["candidates"])),
            "ranking_dataset_rows": int(len(inputs["ranking_dataset"])),
            "scored_rows": int(len(inputs["scored_rows"])),
            "reranked_rows": int(len(inputs["reranked_rows"])),
        },
        "health_contract": {
            "checks_evaluated": len(checks),
            "warn_checks": len(flagged_checks),
            "check_names": [check["name"] for check in checks],
        },
    }


def build_candidate_summary(candidates: pd.DataFrame, ranking_dataset: pd.DataFrame) -> dict[str, Any]:
    source_mix = normalize_counts(candidates["candidate_source"].value_counts().to_dict())
    multi_source_rate = (
        float((candidates["source_count"].astype(int) > 1).mean()) if len(candidates) else 0.0
    )
    split_source_mix = {
        split_name: normalize_counts(frame["candidate_source"].value_counts().to_dict())
        for split_name, frame in ranking_dataset.groupby("dataset_split")
    }
    max_share_gap = compute_max_share_gap(
        split_source_mix.get("train", {}),
        split_source_mix.get("valid", {}),
    )
    return {
        "headline": {
            "candidate_row_count": int(len(candidates)),
            "request_count": int(candidates["request_id"].nunique()),
            "multi_source_rate": multi_source_rate,
            "distinct_item_count": int(candidates["item_id"].nunique()),
        },
        "overall_candidate_source_mix": source_mix,
        "split_candidate_source_mix": {
            "train": split_source_mix.get("train", {}),
            "valid": split_source_mix.get("valid", {}),
            "max_share_gap": max_share_gap,
        },
        "topic_mix": normalize_counts(candidates["topic"].value_counts().to_dict()),
    }


def build_score_summary(scored_rows: pd.DataFrame) -> dict[str, Any]:
    split_prediction_means = {
        split_name: float(frame["prediction"].mean())
        for split_name, frame in scored_rows.groupby("dataset_split")
    }
    split_positive_rates = {
        split_name: float(frame["label"].mean())
        for split_name, frame in scored_rows.groupby("dataset_split")
    }
    train_mean = split_prediction_means.get("train", 0.0)
    valid_mean = split_prediction_means.get("valid", 0.0)
    top_prediction_rows = (
        scored_rows.sort_values(["request_id", "prediction"], ascending=[True, False])
        .groupby("request_id", as_index=False)
        .head(1)
    )
    return {
        "headline": {
            "scored_row_count": int(len(scored_rows)),
            "mean_prediction": float(scored_rows["prediction"].mean()) if len(scored_rows) else 0.0,
            "positive_label_rate": float(scored_rows["label"].mean()) if len(scored_rows) else 0.0,
            "top_prediction_ctr": float(top_prediction_rows["label"].mean()) if len(top_prediction_rows) else 0.0,
        },
        "split_score_stability": {
            "mean_prediction_by_split": split_prediction_means,
            "positive_rate_by_split": split_positive_rates,
            "mean_prediction_gap": abs(train_mean - valid_mean),
            "positive_rate_gap": abs(
                split_positive_rates.get("train", 0.0) - split_positive_rates.get("valid", 0.0)
            ),
        },
        "prediction_quantiles": {
            "p10": float(scored_rows["prediction"].quantile(0.1)) if len(scored_rows) else 0.0,
            "p50": float(scored_rows["prediction"].quantile(0.5)) if len(scored_rows) else 0.0,
            "p90": float(scored_rows["prediction"].quantile(0.9)) if len(scored_rows) else 0.0,
        },
    }


def build_rerank_summary(reranked_rows: pd.DataFrame) -> dict[str, Any]:
    changed_requests = reranked_rows.groupby("request_id")["rank_shift"].apply(
        lambda values: bool((values.abs() > 0).any())
    )
    top_rows = reranked_rows.loc[reranked_rows["post_rank"] == 1]
    return {
        "headline": {
            "request_count": int(reranked_rows["request_id"].nunique()),
            "changed_request_rate": float(changed_requests.mean()) if len(changed_requests) else 0.0,
            "average_rank_shift": float(reranked_rows["rank_shift"].abs().mean()) if len(reranked_rows) else 0.0,
            "top1_creator_count": int(top_rows["creator_id"].nunique()),
        },
        "changed_request_count": int(changed_requests.sum()) if len(changed_requests) else 0,
        "average_absolute_rank_shift": (
            float(reranked_rows["rank_shift"].abs().mean()) if len(reranked_rows) else 0.0
        ),
        "top1_topic_mix": normalize_counts(top_rows["topic"].value_counts().to_dict()),
        "top1_creator_mix": normalize_counts(top_rows["creator_id"].value_counts().to_dict()),
    }


def build_experiment_summary(
    experiment_summary: dict[str, Any],
    experiment_readout: dict[str, Any],
) -> dict[str, Any]:
    valid_only = experiment_readout["diagnostics"].get("valid_only_summary", [])
    primary_metrics = experiment_summary["primary_metrics"]
    control_id = primary_metrics["control_treatment_id"]
    treatment_lifts = {
        treatment_id: metric_payload["lift_vs_control"]
        for treatment_id, metric_payload in primary_metrics.items()
        if isinstance(metric_payload, dict) and "lift_vs_control" in metric_payload
    }
    return {
        "headline": {
            "experiment_id": experiment_summary["experiment_id"],
            "treatment_count": len(experiment_summary["treatment_summaries"]),
            "srm_flagged": bool(experiment_summary["srm_check"]["flagged"]),
            "control_top1_ctr": float(primary_metrics[control_id]["top1_ctr"]),
        },
        "srm_check": experiment_summary["srm_check"],
        "treatment_top1_ctr": {
            treatment_id: float(metric_payload["top1_ctr"])
            for treatment_id, metric_payload in primary_metrics.items()
            if isinstance(metric_payload, dict) and "top1_ctr" in metric_payload
        },
        "treatment_lift_vs_control": treatment_lifts,
        "guardrails": experiment_summary["guardrails"],
        "valid_only_summary": valid_only,
    }


def build_min_check(*, name: str, observed: float, threshold: float, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if observed >= threshold else "warn",
        "comparison": ">=",
        "observed": observed,
        "threshold": threshold,
        "description": description,
    }


def build_max_check(*, name: str, observed: float, threshold: float, description: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if observed <= threshold else "warn",
        "comparison": "<=",
        "observed": observed,
        "threshold": threshold,
        "description": description,
    }


def normalize_counts(counts: dict[Any, Any]) -> dict[str, float]:
    total = float(sum(float(value) for value in counts.values()))
    if total <= 0:
        return {}
    return {
        str(key): float(value) / total
        for key, value in sorted(counts.items(), key=lambda item: str(item[0]))
    }


def top_share_dict(shares: dict[str, float], *, limit: int) -> dict[str, float]:
    ranked = sorted(shares.items(), key=lambda item: (-item[1], item[0]))
    return {key: value for key, value in ranked[:limit]}


def compute_max_share_gap(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    return max(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)
