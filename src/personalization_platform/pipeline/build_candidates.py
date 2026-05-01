from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from personalization_platform.retrieval.affinity import build_affinity_source_candidates
from personalization_platform.retrieval.common import get_source_configs, load_event_log_inputs
from personalization_platform.retrieval.content import build_content_source_candidates
from personalization_platform.retrieval.trending import build_trending_manifest, build_trending_source_candidates
from personalization_platform.utils.artifacts import (
    attach_lineage,
    build_upstream_run_entry,
    create_run_dir,
    write_json,
    write_yaml,
)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate sets from event-log outputs.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = config.get("run_name", "candidate_build")
    artifact_base_dir = config["artifacts"]["base_dir"]
    run_dir = create_run_dir(run_name, base_dir=artifact_base_dir)
    output_dir = Path(config["output"]["base_dir"]) / run_dir.name
    output_dir.mkdir(parents=True, exist_ok=False)

    candidates, metrics, manifest = build_candidates_bundle(config=config, output_dir=output_dir)
    candidates.to_csv(output_dir / "candidates.csv", index=False)
    manifest = attach_lineage(
        manifest,
        run_dir=run_dir,
        output_dir=output_dir,
        config=config,
        upstream_runs=[build_upstream_run_entry(label="event_log", path=metrics["event_log_input_dir"])],
    )

    write_yaml(run_dir / "config.yaml", config)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "manifest.json", manifest)
    print(f"Wrote candidate bundle to {run_dir}")
    print(f"Wrote candidate outputs to {output_dir}")


def build_candidates_bundle(
    *,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    event_log_inputs = load_event_log_inputs(config)
    source_configs = get_source_configs(config)

    source_frames: list[pd.DataFrame] = []
    source_metrics: dict[str, dict[str, Any]] = {}
    for source_config in source_configs:
        source_name = source_config["name"]
        candidate_count = int(source_config["candidate_count"])
        if source_name == "trending":
            frame = build_trending_source_candidates(
                event_log_inputs=event_log_inputs,
                candidate_count=candidate_count,
            )
        elif source_name == "affinity":
            frame = build_affinity_source_candidates(
                event_log_inputs=event_log_inputs,
                candidate_count=candidate_count,
            )
        elif source_name == "content":
            frame = build_content_source_candidates(
                event_log_inputs=event_log_inputs,
                candidate_count=candidate_count,
            )
        else:
            raise ValueError(f"Unsupported retrieval source '{source_name}'.")

        frame = frame.copy()
        frame["source_priority"] = int(source_config["priority"])
        source_frames.append(frame)
        source_metrics[source_name] = {
            "candidate_rows": int(len(frame)),
            "requests_with_candidates": int(frame["request_id"].nunique()) if not frame.empty else 0,
            "distinct_items": int(frame["item_id"].nunique()) if not frame.empty else 0,
        }

    merged_candidates = merge_candidates(
        source_frames=source_frames,
        final_candidate_count=int(config["retrieval"]["candidate_count"]),
    )
    if merged_candidates.empty:
        raise ValueError(
            "Candidate generation produced zero rows across all configured sources. "
            "Check event-log inputs, retrieval source settings, or fixture coverage."
        )
    metrics = build_multi_source_metrics(
        merged_candidates=merged_candidates,
        source_metrics=source_metrics,
        event_log_inputs=event_log_inputs,
        final_candidate_count=int(config["retrieval"]["candidate_count"]),
    )
    manifest = build_multi_source_manifest(
        config=config,
        metrics=metrics,
        output_dir=output_dir,
        source_configs=source_configs,
    )
    return merged_candidates, metrics, manifest


def merge_candidates(
    *,
    source_frames: list[pd.DataFrame],
    final_candidate_count: int,
) -> pd.DataFrame:
    non_empty_frames = [frame for frame in source_frames if not frame.empty]
    if not non_empty_frames:
        return pd.DataFrame(
            columns=[
                "request_id",
                "user_id",
                "item_id",
                "candidate_source",
                "merged_rank",
                "merged_score",
                "source_rank",
                "topic",
                "source_count",
                "source_list",
                "source_details",
            ]
        )

    combined = pd.concat(non_empty_frames, ignore_index=True, sort=False)
    combined = combined.sort_values(
        ["request_id", "source_priority", "source_rank", "source_score", "item_id"],
        ascending=[True, True, True, False, True],
    )

    merged_rows: list[dict[str, Any]] = []
    for request_id, request_frame in combined.groupby("request_id", sort=False):
        deduped: list[dict[str, Any]] = []
        seen_items: set[str] = set()
        for item_id, item_frame in request_frame.groupby("item_id", sort=False):
            primary = item_frame.iloc[0]
            provenance = [
                {
                    "source": row.candidate_source,
                    "priority": int(row.source_priority),
                    "source_rank": int(row.source_rank),
                    "source_score": float(row.source_score),
                }
                for row in item_frame.itertuples(index=False)
            ]
            deduped.append(
                {
                    "request_id": primary["request_id"],
                    "user_id": primary["user_id"],
                    "item_id": item_id,
                    "candidate_source": primary["candidate_source"],
                    "merged_score": float(primary["source_score"]),
                    "source_rank": int(primary["source_rank"]),
                    "topic": primary.get("topic", ""),
                    "source_count": len(provenance),
                    "source_list": json.dumps([entry["source"] for entry in provenance]),
                    "source_details": json.dumps(provenance),
                    "_source_priority": int(primary["source_priority"]),
                }
            )
            seen_items.add(item_id)

        deduped_frame = pd.DataFrame(deduped).sort_values(
            ["_source_priority", "source_rank", "merged_score", "item_id"],
            ascending=[True, True, False, True],
        )
        deduped_frame = deduped_frame.head(final_candidate_count).reset_index(drop=True)
        deduped_frame["merged_rank"] = deduped_frame.index + 1
        merged_rows.extend(deduped_frame.to_dict(orient="records"))

    merged = pd.DataFrame(merged_rows)
    if merged.empty:
        return merged
    return merged[
        [
            "request_id",
            "user_id",
            "item_id",
            "candidate_source",
            "merged_rank",
            "merged_score",
            "source_rank",
            "topic",
            "source_count",
            "source_list",
            "source_details",
        ]
    ]


def build_multi_source_metrics(
    *,
    merged_candidates: pd.DataFrame,
    source_metrics: dict[str, dict[str, Any]],
    event_log_inputs: dict[str, Any],
    final_candidate_count: int,
) -> dict[str, Any]:
    requests = event_log_inputs["requests"]
    clicked_lookup = event_log_inputs["clicked_lookup"]
    candidate_requests = (
        set(merged_candidates["request_id"].unique()) if not merged_candidates.empty else set()
    )
    hit_requests = 0
    for request_id, clicked_items in clicked_lookup.items():
        request_candidates = set(
            merged_candidates.loc[merged_candidates["request_id"] == request_id, "item_id"].tolist()
        )
        if request_candidates.intersection(clicked_items):
            hit_requests += 1

    requests_with_click = sum(1 for items in clicked_lookup.values() if items)
    clicked_item_hit_rate = hit_requests / requests_with_click if requests_with_click else 0.0
    primary_source_counts = (
        merged_candidates["candidate_source"].value_counts().to_dict()
        if not merged_candidates.empty
        else {}
    )
    multi_source_rows = (
        int((merged_candidates["source_count"] > 1).sum()) if not merged_candidates.empty else 0
    )

    return {
        "source_name": "merged_candidates",
        "candidate_count_requested": final_candidate_count,
        "event_log_input_dir": str(event_log_inputs["event_log_dir"]),
        "row_counts": {
            "requests": int(len(requests)),
            "candidates": int(len(merged_candidates)),
        },
        "requests_with_candidates": int(len(candidate_requests)),
        "requests_without_candidates": int(len(requests) - len(candidate_requests)),
        "average_candidates_per_scored_request": (
            float(len(merged_candidates) / len(candidate_requests)) if candidate_requests else 0.0
        ),
        "distinct_candidate_items": (
            int(merged_candidates["item_id"].nunique()) if not merged_candidates.empty else 0
        ),
        "requests_with_click": int(requests_with_click),
        "clicked_item_hit_rate": clicked_item_hit_rate,
        "source_metrics": source_metrics,
        "primary_source_counts": primary_source_counts,
        "multi_source_rows": multi_source_rows,
    }


def build_multi_source_manifest(
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    output_dir: Path,
    source_configs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "source_name": "merged_candidates",
        "source_type": "multi_source_retrieval",
        "event_log_input_dir": metrics["event_log_input_dir"],
        "candidate_output_path": str(output_dir / "candidates.csv"),
        "candidate_count_requested": config["retrieval"]["candidate_count"],
        "sources": source_configs,
        "candidate_columns": [
            "request_id",
            "user_id",
            "item_id",
            "candidate_source",
            "merged_rank",
            "merged_score",
            "source_rank",
            "topic",
            "source_count",
            "source_list",
            "source_details",
        ],
        "assumptions": build_trending_manifest(config=config, metrics=metrics, output_dir=output_dir)["assumptions"]
        + [
            "Affinity candidates are scored from request-time visible topic history using item topics from item_state.",
            "Content candidates are scored from request-time metadata similarity using topic, subcategory, publisher, creator, and title-token overlap from item_state.",
            "When multiple sources retrieve the same item, the merged row keeps the highest-priority source as candidate_source and preserves all contributing sources in source_details.",
            "Cold-start requests rely on trending fallback because affinity requires prior user history.",
        ],
    }


if __name__ == "__main__":
    main()
