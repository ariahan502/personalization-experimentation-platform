from pathlib import Path

from personalization_platform.utils.artifacts import (
    attach_lineage,
    build_run_manifest_metadata,
    build_upstream_run_entry,
    parse_run_dir_metadata,
)


def test_parse_run_dir_metadata_extracts_timestamp_and_run_name():
    metadata = parse_run_dir_metadata(Path("artifacts/runs/20260430_010203_123456_ranker_smoke"))

    assert metadata["run_id"] == "20260430_010203_123456_ranker_smoke"
    assert metadata["timestamp"] == "20260430_010203_123456"
    assert metadata["run_name"] == "ranker_smoke"


def test_attach_lineage_adds_current_and_upstream_run_metadata(tmp_path):
    run_dir = tmp_path / "artifacts" / "20260430_010203_123456_candidates_smoke"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "data" / "processed" / run_dir.name
    output_dir.mkdir(parents=True)

    manifest = attach_lineage(
        {"source_name": "merged_candidates"},
        run_dir=run_dir,
        output_dir=output_dir,
        config={"artifacts": {"base_dir": str(tmp_path / "artifacts")}},
        upstream_runs=[
            build_upstream_run_entry(
                label="event_log",
                path=tmp_path / "data" / "interim" / "20260430_000000_000000_mind_smoke_event_log",
            )
        ],
    )

    assert manifest["run_metadata"]["run_name"] == "candidates_smoke"
    assert manifest["run_metadata"]["output_dir"] == str(output_dir)
    assert manifest["upstream_runs"][0]["label"] == "event_log"
    assert manifest["upstream_runs"][0]["run_name"] == "mind_smoke_event_log"


def test_build_run_manifest_metadata_handles_nonstandard_dir_names(tmp_path):
    run_dir = tmp_path / "artifacts" / "manual_bundle"
    run_dir.mkdir(parents=True)

    metadata = build_run_manifest_metadata(run_dir=run_dir)

    assert metadata["run_id"] == "manual_bundle"
    assert metadata["run_name"] == "manual_bundle"
