import json

import pandas as pd
import pytest

from personalization_platform.pipeline.build_candidates import build_candidates_bundle
from personalization_platform.ranking.dataset import build_ranking_dataset
from personalization_platform.reporting.bundle import load_reporting_inputs
from personalization_platform.retrieval.common import load_event_log_inputs


def write_event_log_dir(base_dir, run_name: str, *, requests=None, impressions=None, user_state=None, item_state=None):
    event_log_dir = base_dir / f"20260429_000000_{run_name}"
    event_log_dir.mkdir(parents=True)
    (requests if requests is not None else pd.DataFrame([{"request_id": "r1", "user_id": "u1", "request_ts": "2026-04-29T10:00:00"}])).to_csv(
        event_log_dir / "requests.csv", index=False
    )
    (impressions if impressions is not None else pd.DataFrame([{"request_id": "r1", "item_id": "i1", "clicked": 1}])).to_csv(
        event_log_dir / "impressions.csv", index=False
    )
    (
        user_state
        if user_state is not None
        else pd.DataFrame(
            [
                {
                    "request_id": "r1",
                    "history_item_ids": "[]",
                    "recent_topic_counts": "{}",
                    "history_click_count": 0,
                    "is_cold_start": True,
                }
            ]
        )
    ).to_csv(event_log_dir / "user_state.csv", index=False)
    (item_state if item_state is not None else pd.DataFrame([{"item_id": "i1", "topic": "Tech"}])).to_csv(
        event_log_dir / "item_state.csv", index=False
    )
    return event_log_dir


def write_candidates_dir(base_dir, run_name: str, rows: pd.DataFrame):
    candidates_dir = base_dir / f"20260429_000000_{run_name}"
    candidates_dir.mkdir(parents=True)
    rows.to_csv(candidates_dir / "candidates.csv", index=False)
    return candidates_dir


def test_load_event_log_inputs_fails_on_missing_required_file(tmp_path):
    event_log_dir = write_event_log_dir(tmp_path, "mind_smoke_event_log")
    (event_log_dir / "user_state.csv").unlink()

    config = {"input": {"event_log_base_dir": str(tmp_path), "event_log_run_name": "mind_smoke_event_log"}}

    with pytest.raises(FileNotFoundError, match="missing required files: user_state.csv"):
        load_event_log_inputs(config)


def test_build_candidates_bundle_fails_when_all_sources_empty(tmp_path):
    requests = pd.DataFrame(
        [{"request_id": "r1", "user_id": "u1", "request_ts": "2026-04-29T10:00:00", "candidate_count": 0}]
    )
    impressions = pd.DataFrame(columns=["request_id", "item_id", "clicked"])
    user_state = pd.DataFrame(
        [
            {
                "request_id": "r1",
                "history_item_ids": "[]",
                "recent_topic_counts": "{}",
                "history_click_count": 0,
                "is_cold_start": True,
            }
        ]
    )
    item_state = pd.DataFrame(columns=["item_id", "topic", "published_ts"])
    write_event_log_dir(
        tmp_path / "event_logs",
        "mind_smoke_event_log",
        requests=requests,
        impressions=impressions,
        user_state=user_state,
        item_state=item_state,
    )
    output_dir = tmp_path / "candidate_outputs"
    output_dir.mkdir()
    config = {
        "input": {"event_log_base_dir": str(tmp_path / "event_logs"), "event_log_run_name": "mind_smoke_event_log"},
        "retrieval": {"candidate_count": 5, "sources": [{"name": "trending", "candidate_count": 5, "priority": 1}]},
    }

    with pytest.raises(ValueError, match="produced zero rows across all configured sources"):
        build_candidates_bundle(config=config, output_dir=output_dir)


def test_build_ranking_dataset_fails_on_missing_candidate_columns(tmp_path):
    write_event_log_dir(
        tmp_path / "event_logs",
        "mind_smoke_event_log",
        requests=pd.DataFrame(
            [
                {
                    "request_id": "r1",
                    "user_id": "u1",
                    "session_id": "s1",
                    "request_ts": "2026-04-29T10:00:00",
                    "split": "train",
                    "candidate_count": 1,
                    "history_length": 0,
                    "request_index_in_session": 1,
                }
            ]
        ),
        impressions=pd.DataFrame([{"request_id": "r1", "item_id": "i1", "clicked": 1, "position": 1}]),
        user_state=pd.DataFrame(
            [
                {
                    "request_id": "r1",
                    "history_item_ids": "[]",
                    "recent_topic_counts": "{}",
                    "history_click_count": 0,
                    "is_cold_start": True,
                }
            ]
        ),
        item_state=pd.DataFrame([{"item_id": "i1", "topic": "Tech"}]),
    )
    write_candidates_dir(
        tmp_path / "candidates",
        "candidates_smoke",
        pd.DataFrame([{"request_id": "r1", "user_id": "u1", "item_id": "i1"}]),
    )
    config = {
        "input": {
            "event_log_base_dir": str(tmp_path / "event_logs"),
            "event_log_run_name": "mind_smoke_event_log",
            "candidates_base_dir": str(tmp_path / "candidates"),
            "candidates_run_name": "candidates_smoke",
        },
        "split": {"strategy": "tail_request_count", "valid_request_count": 1},
    }

    with pytest.raises(ValueError, match="missing required columns"):
        build_ranking_dataset(config)


def test_build_ranking_dataset_fails_on_unmatched_request_context(tmp_path):
    write_event_log_dir(
        tmp_path / "event_logs",
        "mind_smoke_event_log",
        requests=pd.DataFrame(
            [
                {
                    "request_id": "r1",
                    "user_id": "u1",
                    "session_id": "s1",
                    "request_ts": "2026-04-29T10:00:00",
                    "split": "train",
                    "candidate_count": 1,
                    "history_length": 0,
                    "request_index_in_session": 1,
                }
            ]
        ),
        impressions=pd.DataFrame([{"request_id": "r1", "item_id": "i1", "clicked": 1, "position": 1}]),
        user_state=pd.DataFrame(
            [
                {
                    "request_id": "r1",
                    "history_item_ids": "[]",
                    "recent_topic_counts": "{}",
                    "history_click_count": 0,
                    "is_cold_start": True,
                }
            ]
        ),
        item_state=pd.DataFrame([{"item_id": "i1", "topic": "Tech"}]),
    )
    write_candidates_dir(
        tmp_path / "candidates",
        "candidates_smoke",
        pd.DataFrame(
            [
                {
                    "request_id": "missing-request",
                    "user_id": "u1",
                    "item_id": "i1",
                    "candidate_source": "trending",
                    "merged_rank": 1,
                    "merged_score": 1.0,
                    "source_rank": 1,
                    "topic": "Tech",
                    "source_count": 1,
                    "source_list": json.dumps(["trending"]),
                    "source_details": json.dumps([{"source": "trending"}]),
                }
            ]
        ),
    )
    config = {
        "input": {
            "event_log_base_dir": str(tmp_path / "event_logs"),
            "event_log_run_name": "mind_smoke_event_log",
            "candidates_base_dir": str(tmp_path / "candidates"),
            "candidates_run_name": "candidates_smoke",
        },
        "split": {"strategy": "latest_timestamp_bucket"},
    }

    with pytest.raises(ValueError, match="without matching request or user-state context"):
        build_ranking_dataset(config)


def test_load_reporting_inputs_fails_when_expected_artifact_file_missing(tmp_path):
    def make_run(base_name: str, run_name: str, files: dict[str, str]) -> None:
        run_dir = tmp_path / base_name / f"20260429_000000_{run_name}"
        run_dir.mkdir(parents=True)
        for filename, contents in files.items():
            (run_dir / filename).write_text(contents, encoding="utf-8")

    json_blob = "{}"
    make_run("artifacts", "mind_smoke_event_log", {"metrics.json": json_blob})
    make_run("artifacts", "candidates_smoke", {"metrics.json": json_blob})
    make_run("artifacts", "ranker_smoke", {"metrics.json": json_blob})
    make_run("ranker_compare", "ranker_compare_smoke", {"diagnostics.json": json_blob})
    make_run("artifacts", "rerank_smoke", {"metrics.json": json_blob})
    make_run("experiment_analysis", "experiment_analysis_smoke", {"summary.json": json_blob})
    make_run("monitoring", "monitoring_smoke", {"summary.json": json_blob, "diagnostics.json": json_blob})
    make_run("local_api", "local_api_smoke", {"summary.json": json_blob, "smoke_response.json": json_blob})

    config = {
        "event_log_base_dir": str(tmp_path / "artifacts"),
        "event_log_run_name": "mind_smoke_event_log",
        "candidate_base_dir": str(tmp_path / "artifacts"),
        "candidate_run_name": "candidates_smoke",
        "ranker_base_dir": str(tmp_path / "artifacts"),
        "ranker_run_name": "ranker_smoke",
        "ranker_compare_base_dir": str(tmp_path / "ranker_compare"),
        "ranker_compare_run_name": "ranker_compare_smoke",
        "rerank_base_dir": str(tmp_path / "artifacts"),
        "rerank_run_name": "rerank_smoke",
        "experiment_analysis_base_dir": str(tmp_path / "experiment_analysis"),
        "experiment_analysis_run_name": "experiment_analysis_smoke",
        "monitoring_base_dir": str(tmp_path / "monitoring"),
        "monitoring_run_name": "monitoring_smoke",
        "local_api_base_dir": str(tmp_path / "local_api"),
        "local_api_run_name": "local_api_smoke",
        "artifacts_base_dir": str(tmp_path / "artifacts"),
    }

    with pytest.raises(FileNotFoundError, match="Expected artifact file does not exist: .*metrics.json"):
        load_reporting_inputs(config)
