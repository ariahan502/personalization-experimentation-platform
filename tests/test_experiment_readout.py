import pandas as pd
import pytest

from personalization_platform.experiments.readout import (
    build_guardrail_metrics,
    build_primary_metrics,
    build_treatment_diagnostics,
    build_treatment_summaries,
)


def test_build_primary_metrics_includes_multiple_outcomes():
    treatment_summaries = {
        "control": {
            "top1_ctr": 0.5,
            "top2_ctr": 0.25,
            "mean_label": 0.2,
        },
        "treatment": {
            "top1_ctr": 0.75,
            "top2_ctr": 0.5,
            "mean_label": 0.3,
        },
    }
    experiment = {
        "treatments": [
            {"treatment_id": "control", "is_control": True},
            {"treatment_id": "treatment", "is_control": False},
        ]
    }

    metrics = build_primary_metrics(treatment_summaries, experiment)

    assert metrics["control"]["top1_ctr"] == 0.5
    assert metrics["treatment"]["lift_vs_control"] == 0.25
    assert metrics["treatment"]["top2_ctr_lift_vs_control"] == 0.25
    assert metrics["treatment"]["mean_label_lift_vs_control"] == pytest.approx(0.1)


def test_build_guardrail_metrics_includes_repeat_and_concentration_fields():
    treatment_summaries = {
        "control": {
            "average_rank_shift": 0.1,
            "mean_prediction": 0.3,
            "mean_rerank_delta": 0.05,
            "top2_creator_repeat_rate": 0.25,
            "top2_topic_repeat_rate": 0.5,
            "top1_creator_mix": {"c1": 3, "c2": 1},
            "top1_topic_mix": {"Tech": 2, "World": 2},
        }
    }
    experiment = {"treatments": [{"treatment_id": "control", "is_control": True}]}

    guardrails = build_guardrail_metrics(treatment_summaries, experiment)

    assert guardrails["control"]["top2_creator_repeat_rate"] == 0.25
    assert guardrails["control"]["top2_topic_repeat_rate"] == 0.5
    assert guardrails["control"]["top1_creator_concentration"] == 0.75
    assert guardrails["control"]["top1_topic_concentration"] == 0.5


def test_build_treatment_diagnostics_emits_cold_start_and_history_slices():
    rows = pd.DataFrame(
        [
            {
                "treatment_id": "control",
                "request_id": "r1",
                "item_id": "i1",
                "dataset_split": "valid",
                "topic": "Tech",
                "creator_id": "c1",
                "candidate_source": "affinity",
                "post_rank": 1,
                "label": 1,
                "prediction": 0.8,
                "rerank_score": 0.85,
                "history_length": 0,
                "is_cold_start": True,
            },
            {
                "treatment_id": "control",
                "request_id": "r1",
                "item_id": "i2",
                "dataset_split": "valid",
                "topic": "Tech",
                "creator_id": "c1",
                "candidate_source": "trending",
                "post_rank": 2,
                "label": 0,
                "prediction": 0.3,
                "rerank_score": 0.25,
                "history_length": 0,
                "is_cold_start": True,
            },
            {
                "treatment_id": "treatment",
                "request_id": "r2",
                "item_id": "i3",
                "dataset_split": "valid",
                "topic": "World",
                "creator_id": "c2",
                "candidate_source": "content",
                "post_rank": 1,
                "label": 0,
                "prediction": 0.6,
                "rerank_score": 0.65,
                "history_length": 4,
                "is_cold_start": False,
            },
            {
                "treatment_id": "treatment",
                "request_id": "r2",
                "item_id": "i4",
                "dataset_split": "valid",
                "topic": "Finance",
                "creator_id": "c3",
                "candidate_source": "affinity",
                "post_rank": 2,
                "label": 1,
                "prediction": 0.4,
                "rerank_score": 0.45,
                "history_length": 4,
                "is_cold_start": False,
            },
        ]
    )

    diagnostics = build_treatment_diagnostics(rows, config={})

    cold_start_slices = diagnostics["treatment_slices"]["by_cold_start_request"]
    history_slices = diagnostics["treatment_slices"]["by_history_depth_request"]
    assert {row["segment"] for row in cold_start_slices} == {"cold_start", "warm"}
    assert {row["segment"] for row in history_slices} == {"cold_start", "medium_history"}
    assert diagnostics["valid_only_summary"][0]["exposure_rows"] >= 1


def test_build_treatment_summaries_tracks_repeat_rates_and_rerank_delta():
    assignments = pd.DataFrame(
        [
            {"request_id": "r1", "assignment_unit_id": "u1", "treatment_id": "control"},
            {"request_id": "r2", "assignment_unit_id": "u2", "treatment_id": "control"},
        ]
    )
    exposures = pd.DataFrame(
        [
            {
                "request_id": "r1",
                "assignment_unit_id": "u1",
                "treatment_id": "control",
                "item_id": "i1",
                "post_rank": 1,
                "label": 1,
                "prediction": 0.7,
                "rerank_score": 0.9,
                "rank_shift": 1,
                "topic": "Tech",
                "creator_id": "c1",
            },
            {
                "request_id": "r1",
                "assignment_unit_id": "u1",
                "treatment_id": "control",
                "item_id": "i2",
                "post_rank": 2,
                "label": 0,
                "prediction": 0.4,
                "rerank_score": 0.5,
                "rank_shift": 0,
                "topic": "Tech",
                "creator_id": "c1",
            },
            {
                "request_id": "r2",
                "assignment_unit_id": "u2",
                "treatment_id": "control",
                "item_id": "i3",
                "post_rank": 1,
                "label": 0,
                "prediction": 0.2,
                "rerank_score": 0.3,
                "rank_shift": -1,
                "topic": "World",
                "creator_id": "c2",
            },
        ]
    )

    summaries = build_treatment_summaries(assignments, exposures)

    assert summaries["control"]["top1_ctr"] == 0.5
    assert summaries["control"]["top2_creator_repeat_rate"] == 0.5
    assert summaries["control"]["top2_topic_repeat_rate"] == 0.5
    assert summaries["control"]["mean_rerank_delta"] > 0.0
