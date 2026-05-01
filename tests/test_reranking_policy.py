import pandas as pd

from personalization_platform.reranking.policy import restrict_to_prediction_guard


def test_restrict_to_prediction_guard_filters_large_prediction_gaps():
    rows = pd.DataFrame(
        [
            {"item_id": "i1", "prediction": 0.73, "rerank_score": 1.2},
            {"item_id": "i2", "prediction": 0.05, "rerank_score": 1.3},
            {"item_id": "i3", "prediction": 0.26, "rerank_score": 0.8},
        ]
    )

    eligible = restrict_to_prediction_guard(available=rows, prediction_guard_margin=0.35)

    assert eligible["item_id"].tolist() == ["i1"]


def test_restrict_to_prediction_guard_allows_close_predictions():
    rows = pd.DataFrame(
        [
            {"item_id": "i1", "prediction": 0.42, "rerank_score": 0.79},
            {"item_id": "i2", "prediction": 0.08, "rerank_score": 1.36},
            {"item_id": "i3", "prediction": 0.03, "rerank_score": 1.31},
        ]
    )

    eligible = restrict_to_prediction_guard(available=rows, prediction_guard_margin=0.35)

    assert set(eligible["item_id"]) == {"i1", "i2"}
