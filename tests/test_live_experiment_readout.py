import json

import pandas as pd

from personalization_platform.experiments.live_readout import (
    analyze_live_experiment,
    build_live_guardrails,
)


def test_build_live_guardrails_reports_degraded_and_fallback_rates():
    response_events = pd.DataFrame(
        [
            {"treatment_id": "control", "degraded_modes": json.dumps([]), "fallback_used": 0},
            {"treatment_id": "control", "degraded_modes": json.dumps(["trending_only_fallback"]), "fallback_used": 1},
            {"treatment_id": "treatment", "degraded_modes": json.dumps(["source_budget_truncated"]), "fallback_used": 0},
        ]
    )
    request_events = pd.DataFrame(
        [
            {"treatment_id": "control"},
            {"treatment_id": "control"},
            {"treatment_id": "treatment"},
        ]
    )

    guardrails = build_live_guardrails(response_events=response_events, request_events=request_events)

    assert guardrails["control"]["request_count"] == 2
    assert guardrails["control"]["degraded_request_rate"] == 0.5
    assert guardrails["control"]["fallback_request_rate"] == 0.5
    assert guardrails["treatment"]["degraded_mode_counts"]["source_budget_truncated"] == 1


def test_analyze_live_experiment_consumes_local_api_logs(tmp_path):
    run_dir = tmp_path / "artifacts" / "20260430_010101_local_api_smoke"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "event_id": "r1-request-1",
                "event_ts": "2026-04-30T10:00:00Z",
                "api_name": "test_api",
                "request_id": "r1",
                "mode": "fixture_replay",
                "user_id": "u1",
                "top_k": 1,
                "candidate_input_count": 0,
                "returned_item_count": 1,
                "experiment_id": "exp-1",
                "assignment_unit": "user_id",
                "assignment_unit_id": "u1",
                "hash_bucket": 0.1,
                "treatment_id": "control",
                "treatment_name": "control_feed",
                "is_control": 1,
                "source_rerank_dir": "rerank_dir",
            },
            {
                "event_id": "r2-request-1",
                "event_ts": "2026-04-30T10:01:00Z",
                "api_name": "test_api",
                "request_id": "r2",
                "mode": "request_time_assembly",
                "user_id": "u2",
                "top_k": 1,
                "candidate_input_count": 0,
                "returned_item_count": 1,
                "experiment_id": "exp-1",
                "assignment_unit": "user_id",
                "assignment_unit_id": "u2",
                "hash_bucket": 0.8,
                "treatment_id": "treatment",
                "treatment_name": "treatment_feed",
                "is_control": 0,
                "source_rerank_dir": "rerank_dir",
            },
        ]
    ).to_csv(run_dir / "request_events.csv", index=False)
    pd.DataFrame(
        [
            {
                "event_id": "r1-exposure-1",
                "event_ts": "2026-04-30T10:00:01Z",
                "request_event_id": "r1-request-1",
                "request_id": "r1",
                "api_name": "test_api",
                "mode": "fixture_replay",
                "user_id": "u1",
                "experiment_id": "exp-1",
                "assignment_unit": "user_id",
                "assignment_unit_id": "u1",
                "hash_bucket": 0.1,
                "treatment_id": "control",
                "treatment_name": "control_feed",
                "is_control": 1,
                "item_id": "i1",
                "candidate_source": "affinity",
                "topic": "Tech",
                "creator_id": "c1",
                "pre_rank": 1,
                "post_rank": 1,
                "rank_shift": 0,
                "prediction": 0.7,
                "rerank_score": 0.8,
                "freshness_bonus": 0.1,
            },
            {
                "event_id": "r2-exposure-1",
                "event_ts": "2026-04-30T10:01:01Z",
                "request_event_id": "r2-request-1",
                "request_id": "r2",
                "api_name": "test_api",
                "mode": "request_time_assembly",
                "user_id": "u2",
                "experiment_id": "exp-1",
                "assignment_unit": "user_id",
                "assignment_unit_id": "u2",
                "hash_bucket": 0.8,
                "treatment_id": "treatment",
                "treatment_name": "treatment_feed",
                "is_control": 0,
                "item_id": "i2",
                "candidate_source": "trending",
                "topic": "World",
                "creator_id": "c2",
                "pre_rank": 1,
                "post_rank": 1,
                "rank_shift": 0,
                "prediction": 0.6,
                "rerank_score": 0.7,
                "freshness_bonus": 0.1,
            },
        ]
    ).to_csv(run_dir / "exposure_events.csv", index=False)
    pd.DataFrame(
        [
            {
                "event_id": "r2-click-1",
                "event_ts": "2026-04-30T10:01:02Z",
                "request_event_id": "r2-request-1",
                "request_id": "r2",
                "api_name": "test_api",
                "mode": "request_time_assembly",
                "user_id": "u2",
                "experiment_id": "exp-1",
                "assignment_unit": "user_id",
                "assignment_unit_id": "u2",
                "hash_bucket": 0.8,
                "treatment_id": "treatment",
                "treatment_name": "treatment_feed",
                "is_control": 0,
                "item_id": "i2",
                "click_label": 1,
            }
        ]
    ).to_csv(run_dir / "click_events.csv", index=False)
    pd.DataFrame(
        [
            {
                "event_id": "r1-response-1",
                "event_ts": "2026-04-30T10:00:01Z",
                "request_event_id": "r1-request-1",
                "request_id": "r1",
                "api_name": "test_api",
                "mode": "fixture_replay",
                "user_id": "u1",
                "experiment_id": "exp-1",
                "assignment_unit": "user_id",
                "assignment_unit_id": "u1",
                "hash_bucket": 0.1,
                "treatment_id": "control",
                "treatment_name": "control_feed",
                "is_control": 1,
                "degraded_modes": "[]",
                "fallback_used": 0,
                "status": "served",
                "returned_item_count": 1,
                "top_item_id": "i1",
                "source_rerank_dir": "rerank_dir",
            },
            {
                "event_id": "r2-response-1",
                "event_ts": "2026-04-30T10:01:01Z",
                "request_event_id": "r2-request-1",
                "request_id": "r2",
                "api_name": "test_api",
                "mode": "request_time_assembly",
                "user_id": "u2",
                "experiment_id": "exp-1",
                "assignment_unit": "user_id",
                "assignment_unit_id": "u2",
                "hash_bucket": 0.8,
                "treatment_id": "treatment",
                "treatment_name": "treatment_feed",
                "is_control": 0,
                "degraded_modes": "[\"trending_only_fallback\"]",
                "fallback_used": 1,
                "status": "served",
                "returned_item_count": 1,
                "top_item_id": "i2",
                "source_rerank_dir": "rerank_dir",
            },
        ]
    ).to_csv(run_dir / "response_events.csv", index=False)

    summary, readout = analyze_live_experiment(
        {
            "input": {
                "local_api_base_dir": str(tmp_path / "artifacts"),
                "local_api_run_name": "local_api_smoke",
            },
            "experiment": {
                "experiment_id": "exp-1",
                "treatments": [
                    {"treatment_id": "control", "is_control": True, "weight": 0.5, "treatment_name": "control_feed"},
                    {"treatment_id": "treatment", "is_control": False, "weight": 0.5, "treatment_name": "treatment_feed"},
                ],
            },
            "uncertainty": {"bootstrap_samples": 50},
        }
    )

    assert summary["experiment_id"] == "exp-1"
    assert summary["primary_metrics"]["control"]["top1_ctr"] == 0.0
    assert summary["primary_metrics"]["treatment"]["top1_ctr"] == 1.0
    assert summary["live_guardrails"]["treatment"]["fallback_request_rate"] == 1.0
    assert readout["analysis_scope"] == "local_serving_log_readout"
