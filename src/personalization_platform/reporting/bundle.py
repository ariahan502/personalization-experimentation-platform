from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_reporting_bundle(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    inputs = load_reporting_inputs(config["input"])
    project = config["project"]

    executive_summary = build_executive_summary(project=project, inputs=inputs)
    report_payload = build_report_payload(project=project, inputs=inputs, executive_summary=executive_summary)
    report_markdown = build_report_markdown(project=project, inputs=inputs, executive_summary=executive_summary)
    architecture_markdown = build_architecture_note(project=project, inputs=inputs)
    return executive_summary, report_payload, report_markdown, architecture_markdown


def load_reporting_inputs(input_config: dict[str, Any]) -> dict[str, Any]:
    event_log_dir = resolve_run_dir(input_config["event_log_base_dir"], input_config["event_log_run_name"])
    candidate_dir = resolve_run_dir(input_config["candidate_base_dir"], input_config["candidate_run_name"])
    ranker_dir = resolve_run_dir(input_config["ranker_base_dir"], input_config["ranker_run_name"])
    ranker_compare_dir = resolve_run_dir(
        input_config["ranker_compare_base_dir"],
        input_config["ranker_compare_run_name"],
    )
    rerank_dir = resolve_run_dir(input_config["rerank_base_dir"], input_config["rerank_run_name"])
    experiment_analysis_dir = resolve_run_dir(
        input_config["experiment_analysis_base_dir"],
        input_config["experiment_analysis_run_name"],
    )
    monitoring_dir = resolve_run_dir(input_config["monitoring_base_dir"], input_config["monitoring_run_name"])
    local_api_dir = resolve_run_dir(input_config["local_api_base_dir"], input_config["local_api_run_name"])

    return {
        "input_dirs": {
            "event_log_dir": str(event_log_dir),
            "candidate_dir": str(candidate_dir),
            "ranker_dir": str(ranker_dir),
            "ranker_compare_dir": str(ranker_compare_dir),
            "rerank_dir": str(rerank_dir),
            "experiment_analysis_dir": str(experiment_analysis_dir),
            "monitoring_dir": str(monitoring_dir),
            "local_api_dir": str(local_api_dir),
        },
        "event_log_metrics": read_json_from_artifacts(
            input_config["artifacts_base_dir"],
            input_config["event_log_run_name"],
            "metrics.json",
        ),
        "candidate_metrics": read_json_from_artifacts(
            input_config["artifacts_base_dir"],
            input_config["candidate_run_name"],
            "metrics.json",
        ),
        "ranker_metrics": read_json_from_artifacts(
            input_config["artifacts_base_dir"],
            input_config["ranker_run_name"],
            "metrics.json",
        ),
        "ranker_compare_metrics": read_json(ranker_compare_dir / "metrics.json"),
        "ranker_compare_diagnostics": read_json(ranker_compare_dir / "diagnostics.json"),
        "rerank_metrics": read_json_from_artifacts(
            input_config["artifacts_base_dir"],
            input_config["rerank_run_name"],
            "metrics.json",
        ),
        "experiment_summary": read_json(experiment_analysis_dir / "summary.json"),
        "monitoring_summary": read_json(monitoring_dir / "summary.json"),
        "monitoring_diagnostics": read_json(monitoring_dir / "diagnostics.json"),
        "local_api_summary": read_json(local_api_dir / "summary.json"),
        "local_api_response": read_json(local_api_dir / "smoke_response.json"),
    }


def read_json_from_artifacts(base_dir: str, run_name: str, filename: str) -> dict[str, Any]:
    run_dir = resolve_run_dir(base_dir, run_name)
    return read_json(run_dir / filename)


def resolve_run_dir(base_dir: str, run_name: str) -> Path:
    matches = sorted(Path(base_dir).glob(f"*_{run_name}"))
    if not matches:
        raise FileNotFoundError(f"No outputs found under {base_dir} matching '*_{run_name}'.")
    return matches[-1]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Expected artifact file does not exist: {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def build_executive_summary(*, project: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    event_log_metrics = inputs["event_log_metrics"]
    candidate_metrics = inputs["candidate_metrics"]
    ranker_compare_metrics = inputs["ranker_compare_metrics"]
    rerank_metrics = inputs["rerank_metrics"]
    experiment_summary = inputs["experiment_summary"]
    monitoring_summary = inputs["monitoring_summary"]
    local_api_summary = inputs["local_api_summary"]

    return {
        "project_name": project["name"],
        "report_name": project["report_name"],
        "system_status": "validated_smoke_flow",
        "artifact_inputs": inputs["input_dirs"],
        "system_scope": {
            "event_log_requests": event_log_metrics["row_counts"]["requests"],
            "candidate_requests": candidate_metrics["requests_with_candidates"],
            "reranked_requests": rerank_metrics["request_count"],
            "experiment_requests": sum(
                summary["request_count"] for summary in experiment_summary["treatment_summaries"].values()
            ),
        },
        "headline_metrics": {
            "clicked_item_hit_rate": candidate_metrics["clicked_item_hit_rate"],
            "valid_log_loss": inputs["ranker_metrics"]["valid_metrics"]["log_loss"],
            "log_loss_delta_vs_retrieval_order": ranker_compare_metrics["metric_deltas"]["classification.log_loss"],
            "rerank_changed_request_rate": rerank_metrics["changed_request_rate"],
            "experiment_treatment_lift": experiment_summary["primary_metrics"]["reranked_policy"]["lift_vs_control"],
            "monitoring_flagged_check_count": monitoring_summary["flagged_check_count"],
            "local_api_top_item_id": local_api_summary["top_item_id"],
            "local_api_contextual_top_item_id": local_api_summary.get("contextual_top_item_id"),
        },
        "credibility_notes": [
            "This is an offline smoke-sized system validation, not production evidence.",
            "The raw source is public, but the implemented system layers around it are explicit, reproducible, and engineering-oriented.",
            "The project now supports a reproducible path from event-log build through local API replay and contextual scoring using only repo-local assets.",
            "The repo is complete for its current offline scope, but it is not presented as a live production recommender system.",
        ],
    }


def build_report_payload(
    *,
    project: dict[str, Any],
    inputs: dict[str, Any],
    executive_summary: dict[str, Any],
) -> dict[str, Any]:
    comparison_metrics = inputs["ranker_compare_metrics"]
    primary_variant_name = comparison_metrics["primary_variant_name"]
    experiment_summary = inputs["experiment_summary"]
    monitoring_summary = inputs["monitoring_summary"]
    monitoring_diagnostics = inputs["monitoring_diagnostics"]
    return {
        "project": project,
        "executive_summary": executive_summary,
        "stage_metrics": {
            "event_log": inputs["event_log_metrics"],
            "retrieval": inputs["candidate_metrics"],
            "ranking": inputs["ranker_metrics"],
            "ranking_comparison": comparison_metrics,
            "reranking": inputs["rerank_metrics"],
            "experimentation": experiment_summary,
            "monitoring": monitoring_summary,
            "local_api": inputs["local_api_summary"],
        },
        "business_takeaways": [
            "The candidate layer combines affinity, content, and trending retrieval while preserving trending as an explicit cold-start fallback.",
            f"The primary offline ranker ({primary_variant_name}) improves valid log loss over retrieval-order ranking on the available offline validation slices, but the evidence should still be treated as directional rather than shipment-ready.",
            "Reranking changes some requests without degrading the tiny smoke MRR, which is the intended shape for explicit policy constraints.",
            "Experiment assignment, SRM checking, and reporting are wired end to end so ranking changes can be discussed in a product-decision frame rather than as isolated offline metrics.",
        ],
        "monitoring_checks": monitoring_summary["checks"],
        "top_feature_weights": inputs["ranker_compare_diagnostics"]["baseline_feature_manifest"],
        "api_example": inputs["local_api_response"],
        "caveats": monitoring_diagnostics["caveats"]
        + [
            "The smoke valid split is extremely small, and even the medium path remains a local offline evaluation surface rather than a product-performance claim.",
            "The local API still uses local artifact-backed heuristics for contextual scoring rather than a production feature service.",
            "This bundle is appropriate for portfolio and architecture storytelling, not for claiming production serving maturity or live-traffic lift.",
        ],
    }


def build_report_markdown(*, project: dict[str, Any], inputs: dict[str, Any], executive_summary: dict[str, Any]) -> str:
    candidate_metrics = inputs["candidate_metrics"]
    ranker_metrics = inputs["ranker_metrics"]
    ranker_compare_metrics = inputs["ranker_compare_metrics"]
    primary_variant_name = ranker_compare_metrics["primary_variant_name"]
    rerank_metrics = inputs["rerank_metrics"]
    experiment_summary = inputs["experiment_summary"]
    monitoring_summary = inputs["monitoring_summary"]
    local_api_summary = inputs["local_api_summary"]

    lines = [
        f"# {project['report_name']}",
        "",
        "## What This Project Demonstrates",
        "",
        f"{project['name']} is a reproducible offline personalization stack for a content feed. "
        "It converts raw interaction inputs into request-level event logs, builds multi-source candidates, "
        "trains a baseline ranker, applies explicit reranking constraints, assigns experiments deterministically, "
        "runs offline monitoring, and exposes both replay-style and contextual-scoring paths through a local demo API.",
        "",
        "## Current System Outcome",
        "",
        f"- Event-log scope: {executive_summary['system_scope']['event_log_requests']} requests, {inputs['event_log_metrics']['row_counts']['impressions']} impressions, and {inputs['event_log_metrics']['distinct_sessions']} inferred sessions.",
        f"- Retrieval coverage: {candidate_metrics['requests_with_candidates']} of {inputs['event_log_metrics']['row_counts']['requests']} requests received candidates, with clicked-item hit rate {candidate_metrics['clicked_item_hit_rate']:.3f}.",
        f"- Ranking baseline: valid log loss {ranker_metrics['valid_metrics']['log_loss']:.3f} and valid MRR {ranker_metrics['valid_ranking_metrics']['mean_reciprocal_rank']:.3f}.",
        f"- Comparison view: primary variant `{primary_variant_name}` changed valid accuracy by {ranker_compare_metrics['metric_deltas']['classification.accuracy']:.3f} and valid log loss by {ranker_compare_metrics['metric_deltas']['classification.log_loss']:.3f} versus retrieval-order ranking on the current offline slice.",
        f"- Reranking policy: changed {rerank_metrics['changed_request_count']} of {rerank_metrics['request_count']} requests with average absolute rank shift {rerank_metrics['average_absolute_rank_shift']:.3f}.",
        f"- Experiment readout: treatment top-1 CTR moved from {experiment_summary['primary_metrics']['control']['top1_ctr']:.3f} in control to {experiment_summary['primary_metrics']['reranked_policy']['top1_ctr']:.3f} in reranked policy, with SRM flagged = {str(experiment_summary['srm_check']['flagged']).lower()}.",
        f"- Monitoring posture: {monitoring_summary['flagged_check_count']} smoke checks flagged; overall status = `{monitoring_summary['overall_status']}`.",
        f"- Delivery surface: local API replay returned top item `{local_api_summary['top_item_id']}` for request `{local_api_summary['request_id']}`, and contextual scoring returned `{local_api_summary.get('contextual_top_item_id')}` for `{local_api_summary.get('contextual_request_id')}`.",
        "",
        "## Business Framing",
        "",
        "- Cold-start and low-history behavior are handled explicitly by keeping trending retrieval as a fallback source.",
        "- Retrieval is intentionally multi-source: affinity, content, and trending are kept visible rather than hidden behind one opaque scorer.",
        "- Personalization is separated into retrieval, ranking, reranking, and experimentation stages so tradeoffs remain inspectable.",
        "- Constraint-aware reranking is treated as policy logic, not hidden model behavior, which makes creator spread and diversity easier to reason about.",
        "- The project demonstrates that experimentation and monitoring belong in the same system conversation as ranking quality.",
        "",
        "## Caveats",
        "",
        "- This bundle is based on an offline smoke fixture, so metric changes should be interpreted as system-validation evidence rather than product-performance proof.",
        "- A richer medium validation path exists in the repo, but this reporting bundle is still based on local offline artifacts rather than live traffic.",
        "- The raw source does not expose every production logging field directly, so some request-level and operational attributes are inferred or config-backed in the offline workflow.",
        "- The local API is still a local artifact-backed demo surface, not a production serving system or online feature platform.",
    ]
    return "\n".join(lines) + "\n"


def build_architecture_note(*, project: dict[str, Any], inputs: dict[str, Any]) -> str:
    feature_weights = inputs["ranker_compare_diagnostics"]["baseline_feature_manifest"]
    monitoring_checks = inputs["monitoring_summary"]["checks"]
    top_feature_summary = ", ".join(
        f"{row['feature']} ({row['coefficient']:.3f})" for row in feature_weights[:5]
    )
    lines = [
        f"# {project['architecture_note_name']}",
        "",
        "## System Layers",
        "",
        "1. Event-log foundation converts raw interaction inputs into requests, impressions, user_state, and item_state tables.",
        "2. Retrieval merges affinity, content, and trending candidate sources with deterministic provenance tracking.",
        "3. Ranking turns candidate rows into a supervised dataset and fits an interpretable logistic baseline.",
        "4. Reranking applies explicit freshness, topic-diversity, and creator-spread rules on top of model scores.",
        "5. Experimentation assigns request-level treatments deterministically and emits guardrails plus SRM checks.",
        "6. Monitoring and delivery summarize system health and expose a local replay API for fixture-sized demos.",
        "",
        "## Why The Architecture Is Credible",
        "",
        "- Each stage is config-driven and writes a run bundle under `artifacts/runs/`.",
        "- The repo includes a smoke command for full-path validation and a medium command for richer retrieval-and-ranking validation.",
        "- Experiment analysis and monitoring are first-class outputs, which keeps the project grounded in decision quality rather than model-only metrics.",
        "",
        "## Interpretable Signals",
        "",
        f"- Top baseline feature weights on the smoke dataset: {top_feature_summary}.",
        f"- Monitoring checks currently tracked: {', '.join(check['name'] for check in monitoring_checks)}.",
        "",
        "## Offline Boundaries",
        "",
        "- The ranking and experiment metrics are intentionally offline and local in scope; they should be used as structural evidence, not as proof of live product impact.",
        "- The API serves replayed local artifacts, which is appropriate for portfolio delivery but not for production latency or feature-freshness claims.",
        "- Future work would focus on richer data, stronger validation splits, and a more realistic online-serving abstraction.",
    ]
    return "\n".join(lines) + "\n"
