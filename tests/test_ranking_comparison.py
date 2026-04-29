import pandas as pd

from personalization_platform.ranking.comparison import (
    build_segment_delta_summary,
    build_slice_summary,
)


def test_build_slice_summary_emits_request_and_row_segments():
    rows = pd.DataFrame(
        [
            {
                "request_id": "r1",
                "item_id": "i1",
                "label": 1,
                "prediction": 0.8,
                "predicted_label": 1,
                "candidate_source": "affinity",
                "has_multi_source_provenance": 1,
                "is_cold_start": False,
                "history_length": 3,
            },
            {
                "request_id": "r1",
                "item_id": "i2",
                "label": 0,
                "prediction": 0.2,
                "predicted_label": 0,
                "candidate_source": "content",
                "has_multi_source_provenance": 0,
                "is_cold_start": False,
                "history_length": 3,
            },
            {
                "request_id": "r2",
                "item_id": "i3",
                "label": 0,
                "prediction": 0.6,
                "predicted_label": 1,
                "candidate_source": "trending",
                "has_multi_source_provenance": 0,
                "is_cold_start": True,
                "history_length": 0,
            },
            {
                "request_id": "r2",
                "item_id": "i4",
                "label": 1,
                "prediction": 0.4,
                "predicted_label": 0,
                "candidate_source": "trending",
                "has_multi_source_provenance": 0,
                "is_cold_start": True,
                "history_length": 0,
            },
        ]
    )

    summary = build_slice_summary(rows, config={})

    assert {row["segment"] for row in summary["by_candidate_source"]} == {"affinity", "content", "trending"}
    assert {row["segment"] for row in summary["by_multi_source_provenance"]} == {0, 1}
    assert [row["segment"] for row in summary["by_cold_start_request"]] == ["warm", "cold_start"]
    assert [row["segment"] for row in summary["by_history_depth_request"]] == [
        "cold_start",
        "medium_history",
    ]


def test_build_segment_delta_summary_compares_primary_against_retrieval():
    primary_rows = pd.DataFrame(
        [
            {
                "request_id": "r1",
                "item_id": "i1",
                "label": 1,
                "prediction": 0.9,
                "predicted_label": 1,
                "candidate_source": "affinity",
                "has_multi_source_provenance": 1,
                "is_cold_start": False,
                "history_length": 2,
            },
            {
                "request_id": "r1",
                "item_id": "i2",
                "label": 0,
                "prediction": 0.1,
                "predicted_label": 0,
                "candidate_source": "content",
                "has_multi_source_provenance": 0,
                "is_cold_start": False,
                "history_length": 2,
            },
        ]
    )
    fallback_rows = primary_rows.copy()
    fallback_rows["prediction"] = [0.4, 0.6]
    fallback_rows["predicted_label"] = [0, 1]

    deltas = build_segment_delta_summary(
        variant_rows={
            "logistic_regression_baseline": primary_rows,
            "retrieval_order_baseline": fallback_rows,
        },
        primary_variant_name="logistic_regression_baseline",
        baseline_variant_name="retrieval_order_baseline",
        config={},
    )

    short_history = deltas["by_history_depth_request"][0]
    assert short_history["segment"] == "short_history"
    assert short_history["request_count"] == 1
    assert short_history["metric_deltas"]["classification.accuracy"] == 1.0
