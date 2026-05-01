import json

from personalization_platform.monitoring.lifecycle import (
    analyze_model_lifecycle,
    derive_lifecycle_decision,
)


def test_derive_lifecycle_decision_prefers_rollback_over_hold():
    decision, reasons = derive_lifecycle_decision(
        promotion_checks=[
            {
                "name": "live_control_request_count",
                "status": "fail",
                "comparison": ">=",
                "observed": 0,
                "threshold": 1,
            }
        ],
        rollback_checks=[
            {
                "name": "offline_monitoring_status",
                "status": "fail",
                "comparison": "==",
                "observed": "warn",
                "threshold": "pass",
            }
        ],
    )

    assert decision == "rollback"
    assert reasons == ["offline_monitoring_status == pass failed (observed=warn)."]


def test_analyze_model_lifecycle_holds_when_live_control_is_missing(tmp_path):
    ranker_dir = tmp_path / "artifacts" / "20260430_100000_ranker_smoke"
    ranker_dir.mkdir(parents=True)
    (ranker_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_name": "logistic_regression_baseline",
                "model_type": "logistic_regression",
                "ranking_dataset_input_dir": "data/processed/ranking_dataset/20260430_095959_ranking_dataset_smoke",
                "feature_spec": {"numeric": ["merged_rank"], "binary": [], "categorical": []},
                "feature_count_after_vectorization": 3,
                "run_metadata": {
                    "run_id": "20260430_100000_ranker_smoke",
                    "timestamp": "20260430_100000",
                    "run_name": "ranker_smoke",
                    "path": str(ranker_dir),
                },
                "upstream_runs": [
                    {
                        "label": "ranking_dataset",
                        "run_id": "20260430_095959_ranking_dataset_smoke",
                        "run_name": "ranking_dataset_smoke",
                        "path": "data/processed/ranking_dataset/20260430_095959_ranking_dataset_smoke",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (ranker_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model_name": "logistic_regression_baseline",
                "valid_metrics": {"row_count": 3, "accuracy": 0.67, "log_loss": 0.9, "roc_auc": 0.5},
                "valid_ranking_metrics": {"request_count": 1, "mean_reciprocal_rank": 0.5, "hit_rate_at_1": 0.0},
            }
        ),
        encoding="utf-8",
    )

    compare_dir = tmp_path / "artifacts" / "20260430_100001_ranker_compare_smoke"
    compare_dir.mkdir(parents=True)
    (compare_dir / "metrics.json").write_text(
        json.dumps(
            {
                "comparison_name": "ranker_compare_smoke",
                "primary_variant_name": "logistic_regression_baseline",
                "metric_deltas": {
                    "ranking.mean_reciprocal_rank": 0.1,
                    "classification.log_loss": -0.2,
                },
                "metric_delta_uncertainty": {},
            }
        ),
        encoding="utf-8",
    )

    monitoring_dir = tmp_path / "artifacts" / "20260430_100002_monitoring_smoke"
    monitoring_dir.mkdir(parents=True)
    (monitoring_dir / "summary.json").write_text(
        json.dumps({"overall_status": "pass", "flagged_check_count": 0, "checks": []}),
        encoding="utf-8",
    )

    live_dir = tmp_path / "artifacts" / "20260430_100003_live_experiment_readout_smoke"
    live_dir.mkdir(parents=True)
    (live_dir / "summary.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp-1",
                "srm_check": {"approx_p_value": 0.9},
                "sample_size_summary": {
                    "control": {"request_count": 1},
                    "reranked_policy": {"request_count": 3},
                },
                "primary_metrics": {
                    "reranked_policy": {
                        "lift_vs_control": 0.1,
                        "uncertainty": {
                            "top1_ctr_lift_vs_control": {
                                "ci_lower": -0.05,
                            }
                        },
                    }
                },
                "live_guardrails": {
                    "reranked_policy": {
                        "request_count": 3,
                        "degraded_request_rate": 0.0,
                        "fallback_request_rate": 0.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    summary, lifecycle = analyze_model_lifecycle(
        {
            "run_name": "model_lifecycle_smoke",
            "input": {
                "ranker_base_dir": str(tmp_path / "artifacts"),
                "ranker_run_name": "ranker_smoke",
                "ranker_compare_base_dir": str(tmp_path / "artifacts"),
                "ranker_compare_run_name": "ranker_compare_smoke",
                "monitoring_base_dir": str(tmp_path / "artifacts"),
                "monitoring_run_name": "monitoring_smoke",
                "live_experiment_base_dir": str(tmp_path / "artifacts"),
                "live_experiment_run_name": "live_experiment_readout_smoke",
            },
            "policy": {
                "candidate_treatment_id": "reranked_policy",
                "control_treatment_id": "control",
                "fallback_treatment_id": "control",
                "fallback_variant_name": "retrieval_order_baseline",
            },
            "thresholds": {
                "min_offline_mrr_delta": 0.0,
                "max_offline_log_loss_delta": 0.0,
                "min_control_request_count": 2,
                "min_candidate_request_count": 2,
                "min_live_srm_p_value": 0.05,
                "min_live_top1_ctr_lift": 0.0,
                "min_live_top1_ctr_lift_ci_lower": -0.1,
                "max_live_degraded_request_rate": 0.5,
                "max_live_fallback_request_rate": 0.5,
            },
        }
    )

    assert summary["decision"] == "hold"
    assert summary["candidate_model_name"] == "logistic_regression_baseline"
    assert "live_control_request_count" in summary["decision_reasons"][0]
    assert lifecycle["rollback_policy"]["fallback_treatment_id"] == "control"


def test_analyze_model_lifecycle_holds_when_live_effect_uncertainty_is_too_negative(tmp_path):
    ranker_dir = tmp_path / "artifacts" / "20260430_100000_ranker_smoke"
    ranker_dir.mkdir(parents=True)
    (ranker_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_name": "logistic_regression_baseline",
                "model_type": "logistic_regression",
                "ranking_dataset_input_dir": "data/processed/ranking_dataset/20260430_095959_ranking_dataset_smoke",
                "feature_spec": {"numeric": ["merged_rank"], "binary": [], "categorical": []},
                "feature_count_after_vectorization": 3,
                "run_metadata": {"run_id": "20260430_100000_ranker_smoke"},
                "upstream_runs": [],
            }
        ),
        encoding="utf-8",
    )
    (ranker_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model_name": "logistic_regression_baseline",
                "valid_metrics": {"row_count": 3, "accuracy": 0.67, "log_loss": 0.9, "roc_auc": 0.5},
                "valid_ranking_metrics": {"request_count": 1, "mean_reciprocal_rank": 0.5, "hit_rate_at_1": 0.0},
            }
        ),
        encoding="utf-8",
    )

    compare_dir = tmp_path / "artifacts" / "20260430_100001_ranker_compare_smoke"
    compare_dir.mkdir(parents=True)
    (compare_dir / "metrics.json").write_text(
        json.dumps(
            {
                "comparison_name": "ranker_compare_smoke",
                "primary_variant_name": "logistic_regression_baseline",
                "metric_deltas": {
                    "ranking.mean_reciprocal_rank": 0.1,
                    "classification.log_loss": -0.2,
                },
                "metric_delta_uncertainty": {},
            }
        ),
        encoding="utf-8",
    )

    monitoring_dir = tmp_path / "artifacts" / "20260430_100002_monitoring_smoke"
    monitoring_dir.mkdir(parents=True)
    (monitoring_dir / "summary.json").write_text(
        json.dumps({"overall_status": "pass", "flagged_check_count": 0, "checks": []}),
        encoding="utf-8",
    )

    live_dir = tmp_path / "artifacts" / "20260430_100003_live_experiment_readout_smoke"
    live_dir.mkdir(parents=True)
    (live_dir / "summary.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp-1",
                "srm_check": {"approx_p_value": 0.9},
                "sample_size_summary": {
                    "control": {"request_count": 2},
                    "reranked_policy": {"request_count": 2},
                },
                "primary_metrics": {
                    "reranked_policy": {
                        "lift_vs_control": 0.0,
                        "uncertainty": {
                            "top1_ctr_lift_vs_control": {
                                "ci_lower": -0.5,
                            }
                        },
                    }
                },
                "live_guardrails": {
                    "reranked_policy": {
                        "request_count": 2,
                        "degraded_request_rate": 0.0,
                        "fallback_request_rate": 0.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    summary, _ = analyze_model_lifecycle(
        {
            "run_name": "model_lifecycle_smoke",
            "input": {
                "ranker_base_dir": str(tmp_path / "artifacts"),
                "ranker_run_name": "ranker_smoke",
                "ranker_compare_base_dir": str(tmp_path / "artifacts"),
                "ranker_compare_run_name": "ranker_compare_smoke",
                "monitoring_base_dir": str(tmp_path / "artifacts"),
                "monitoring_run_name": "monitoring_smoke",
                "live_experiment_base_dir": str(tmp_path / "artifacts"),
                "live_experiment_run_name": "live_experiment_readout_smoke",
            },
            "policy": {
                "candidate_treatment_id": "reranked_policy",
                "control_treatment_id": "control",
                "fallback_treatment_id": "control",
                "fallback_variant_name": "retrieval_order_baseline",
            },
            "thresholds": {
                "min_offline_mrr_delta": 0.0,
                "max_offline_log_loss_delta": 0.0,
                "min_control_request_count": 2,
                "min_candidate_request_count": 2,
                "min_live_srm_p_value": 0.05,
                "min_live_top1_ctr_lift": 0.0,
                "min_live_top1_ctr_lift_ci_lower": -0.1,
                "max_live_degraded_request_rate": 0.5,
                "max_live_fallback_request_rate": 0.5,
            },
        }
    )

    assert summary["decision"] == "hold"
    assert "live_top1_ctr_lift_ci_lower" in summary["decision_reasons"][0]
