from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from personalization_platform.delivery.event_logging import (
    CLICK_EVENT_COLUMNS,
    EXPOSURE_EVENT_COLUMNS,
    REQUEST_EVENT_COLUMNS,
    RESPONSE_EVENT_COLUMNS,
    build_event_log_summary,
)
from personalization_platform.experiments.assignment import choose_treatment, compute_hash_bucket, validate_experiment_config


def simulate_serving_bundle(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Any]]:
    rerank_dir = resolve_rerank_dir(config)
    reranked_rows = pd.read_csv(rerank_dir / "reranked_rows.csv")
    experiment = validate_experiment_config(config)
    simulation = config["simulation"]
    request_rows = (
        reranked_rows[["request_id", "user_id", "dataset_split"]]
        .drop_duplicates()
        .sort_values(["request_id"])
        .reset_index(drop=True)
    )

    log_frames = build_simulated_logs(
        reranked_rows=reranked_rows,
        request_rows=request_rows,
        rerank_dir=rerank_dir,
        experiment=experiment,
        simulation=simulation,
        api_name=str(config.get("api_name", "deterministic_replay_simulator")),
    )
    summary = {
        "simulation_name": config.get("run_name", "simulated_serving_smoke"),
        "rerank_input_dir": str(rerank_dir),
        "experiment_id": experiment["experiment_id"],
        "assignment_unit": experiment["assignment_unit"],
        "online_event_log_summary": build_event_log_summary(log_frames),
        "simulation_config": {
            "assignment_strategy": str(simulation.get("assignment_strategy", "hashed_assignment")),
            "rounds": int(simulation.get("rounds", 1)),
            "top_k": int(simulation.get("top_k", 2)),
            "random_seed": int(simulation.get("random_seed", 17)),
            "positive_click_probability": float(simulation.get("positive_click_probability", 0.8)),
            "negative_click_probability": float(simulation.get("negative_click_probability", 0.05)),
        },
    }
    manifest = {
        "simulation_name": config.get("run_name", "simulated_serving_smoke"),
        "rerank_input_dir": str(rerank_dir),
        "experiment_id": experiment["experiment_id"],
        "assignment_unit": experiment["assignment_unit"],
        "treatments": experiment["treatments"],
        "assumptions": [
            "This bundle is a deterministic serving-log simulation derived from reranked offline outputs.",
            "Clicks are generated from offline labels plus a fixed seeded click policy rather than hand-authored request fixtures.",
            "The simulation is intended as a stronger validation tier than the tiny local API smoke path, not as real online evidence.",
            "When assignment_strategy is paired_treatment_replay, each base request is replayed once per treatment per round to reduce request-mix variance in treatment comparison.",
        ],
        "simulation_config": summary["simulation_config"],
    }
    return summary, log_frames, manifest


def build_simulated_logs(
    *,
    reranked_rows: pd.DataFrame,
    request_rows: pd.DataFrame,
    rerank_dir: Path,
    experiment: dict[str, Any],
    simulation: dict[str, Any],
    api_name: str,
) -> dict[str, pd.DataFrame]:
    rounds = int(simulation.get("rounds", 1))
    top_k = int(simulation.get("top_k", 2))
    assignment_strategy = str(simulation.get("assignment_strategy", "hashed_assignment"))
    base_timestamp = parse_base_timestamp(simulation.get("base_timestamp", "2026-04-30T12:00:00Z"))
    request_event_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    click_rows: list[dict[str, Any]] = []
    request_counter = 0

    for round_index in range(1, rounds + 1):
        for request in request_rows.itertuples(index=False):
            treatment_runs = resolve_treatment_runs(
                experiment=experiment,
                assignment_strategy=assignment_strategy,
                request_id=str(request.request_id),
                user_id=str(request.user_id),
                round_index=round_index,
            )
            for treatment_run in treatment_runs:
                request_counter += 1
                simulated_request_id = treatment_run["simulated_request_id"]
                event_ts = format_timestamp(base_timestamp + timedelta(seconds=request_counter))
                served_rows = select_served_rows(
                    reranked_rows.loc[reranked_rows["request_id"] == request.request_id].copy(),
                    treatment_id=treatment_run["treatment_id"],
                    top_k=top_k,
                    control_treatment_id=experiment["control_treatment_id"],
                )
                request_event_id = f"{simulated_request_id}-request-1"
                request_event_rows.append(
                    {
                        "event_id": request_event_id,
                        "event_ts": event_ts,
                        "api_name": api_name,
                        "request_id": simulated_request_id,
                        "mode": "deterministic_replay_simulation",
                        "user_id": str(request.user_id),
                        "top_k": top_k,
                        "candidate_input_count": int(len(served_rows)),
                        "returned_item_count": int(len(served_rows)),
                        "experiment_id": experiment["experiment_id"],
                        "assignment_unit": experiment["assignment_unit"],
                        "assignment_unit_id": treatment_run["assignment_unit_id"],
                        "hash_bucket": treatment_run["hash_bucket"],
                        "treatment_id": treatment_run["treatment_id"],
                        "treatment_name": treatment_run["treatment_name"],
                        "is_control": treatment_run["is_control"],
                        "source_rerank_dir": str(rerank_dir),
                    }
                )
                response_rows.append(
                    {
                        "event_id": f"{simulated_request_id}-response-1",
                        "event_ts": event_ts,
                        "request_event_id": request_event_id,
                        "request_id": simulated_request_id,
                        "api_name": api_name,
                        "mode": "deterministic_replay_simulation",
                        "user_id": str(request.user_id),
                        "experiment_id": experiment["experiment_id"],
                        "assignment_unit": experiment["assignment_unit"],
                        "assignment_unit_id": treatment_run["assignment_unit_id"],
                        "hash_bucket": treatment_run["hash_bucket"],
                        "treatment_id": treatment_run["treatment_id"],
                        "treatment_name": treatment_run["treatment_name"],
                        "is_control": treatment_run["is_control"],
                        "degraded_modes": "[]",
                        "fallback_used": 0,
                        "status": "served",
                        "returned_item_count": int(len(served_rows)),
                        "top_item_id": str(served_rows.iloc[0]["item_id"]) if len(served_rows) else "",
                        "source_rerank_dir": str(rerank_dir),
                    }
                )

                for row in served_rows.itertuples(index=False):
                    exposure_rows.append(
                        {
                            "event_id": f"{simulated_request_id}-exposure-{row.served_post_rank}",
                            "event_ts": event_ts,
                            "request_event_id": request_event_id,
                            "request_id": simulated_request_id,
                            "api_name": api_name,
                            "mode": "deterministic_replay_simulation",
                            "user_id": str(request.user_id),
                            "experiment_id": experiment["experiment_id"],
                            "assignment_unit": experiment["assignment_unit"],
                            "assignment_unit_id": treatment_run["assignment_unit_id"],
                            "hash_bucket": treatment_run["hash_bucket"],
                            "treatment_id": treatment_run["treatment_id"],
                            "treatment_name": treatment_run["treatment_name"],
                            "is_control": treatment_run["is_control"],
                            "item_id": str(row.item_id),
                            "candidate_source": str(row.candidate_source),
                            "topic": str(row.topic),
                            "creator_id": str(row.creator_id),
                            "pre_rank": int(row.pre_rank),
                            "post_rank": int(row.served_post_rank),
                            "rank_shift": int(row.served_rank_shift),
                            "prediction": float(row.prediction),
                            "rerank_score": float(row.rerank_score),
                            "freshness_bonus": float(row.freshness_bonus),
                        }
                    )
                    if simulate_click(
                        seed=int(simulation.get("random_seed", 17)),
                        request_id=str(request.request_id),
                        simulated_request_id=simulated_request_id,
                        item_id=str(row.item_id),
                        treatment_id=treatment_run["treatment_id"],
                        assignment_strategy=assignment_strategy,
                        position=int(row.served_post_rank),
                        label=int(row.label),
                        round_index=round_index,
                        positive_click_probability=float(simulation.get("positive_click_probability", 0.8)),
                        negative_click_probability=float(simulation.get("negative_click_probability", 0.05)),
                        position_decay=list(simulation.get("position_decay", [1.0, 0.7, 0.4])),
                    ):
                        click_rows.append(
                            {
                                "event_id": f"{simulated_request_id}-click-{row.served_post_rank}",
                                "event_ts": event_ts,
                                "request_event_id": request_event_id,
                                "request_id": simulated_request_id,
                                "api_name": api_name,
                                "mode": "deterministic_replay_simulation",
                                "user_id": str(request.user_id),
                                "experiment_id": experiment["experiment_id"],
                                "assignment_unit": experiment["assignment_unit"],
                                "assignment_unit_id": treatment_run["assignment_unit_id"],
                                "hash_bucket": treatment_run["hash_bucket"],
                                "treatment_id": treatment_run["treatment_id"],
                                "treatment_name": treatment_run["treatment_name"],
                                "is_control": treatment_run["is_control"],
                                "item_id": str(row.item_id),
                                "click_label": 1,
                            }
                        )

    return {
        "request_events": pd.DataFrame(request_event_rows, columns=REQUEST_EVENT_COLUMNS),
        "exposure_events": pd.DataFrame(exposure_rows, columns=EXPOSURE_EVENT_COLUMNS),
        "response_events": pd.DataFrame(response_rows, columns=RESPONSE_EVENT_COLUMNS),
        "click_events": pd.DataFrame(click_rows, columns=CLICK_EVENT_COLUMNS),
    }


def resolve_rerank_dir(config: dict[str, Any]) -> Path:
    simulation_input = config["input"]
    base_dir = Path(simulation_input["rerank_base_dir"])
    run_name = simulation_input["rerank_run_name"]
    matches = sorted(base_dir.glob(f"*_{run_name}"))
    if not matches:
        raise FileNotFoundError(f"No rerank outputs found under {base_dir} matching '*_{run_name}'.")
    return matches[-1]


def select_served_rows(
    request_rows: pd.DataFrame,
    *,
    treatment_id: str,
    top_k: int,
    control_treatment_id: str,
) -> pd.DataFrame:
    ranking_column = "pre_rank" if treatment_id == control_treatment_id else "post_rank"
    served_rows = (
        request_rows.sort_values([ranking_column, "item_id"])
        .head(top_k)
        .reset_index(drop=True)
        .copy()
    )
    served_rows["served_post_rank"] = range(1, len(served_rows) + 1)
    served_rows["served_rank_shift"] = served_rows["served_post_rank"] - served_rows["pre_rank"].astype(int)
    return served_rows


def simulate_click(
    *,
    seed: int,
    request_id: str,
    simulated_request_id: str,
    item_id: str,
    treatment_id: str,
    assignment_strategy: str,
    position: int,
    label: int,
    round_index: int,
    positive_click_probability: float,
    negative_click_probability: float,
    position_decay: list[float],
) -> bool:
    decay = position_decay[min(position - 1, len(position_decay) - 1)] if position_decay else 1.0
    base_probability = positive_click_probability if int(label) == 1 else negative_click_probability
    click_probability = max(0.0, min(1.0, base_probability * float(decay)))
    if assignment_strategy == "paired_treatment_replay":
        key = f"{request_id}|round_{round_index}|{item_id}|{position}"
    else:
        key = f"{simulated_request_id}|{item_id}|{treatment_id}|{position}"
    draw = deterministic_uniform(
        seed=seed,
        key=key,
    )
    return draw < click_probability


def resolve_treatment_runs(
    *,
    experiment: dict[str, Any],
    assignment_strategy: str,
    request_id: str,
    user_id: str,
    round_index: int,
) -> list[dict[str, Any]]:
    if assignment_strategy == "paired_treatment_replay":
        runs: list[dict[str, Any]] = []
        midpoint_lookup = build_treatment_midpoint_lookup(experiment["treatments"])
        for treatment in experiment["treatments"]:
            simulated_request_id = f"{request_id}::sim_round_{round_index}::{treatment['treatment_id']}"
            assignment_unit_id = simulated_request_id
            runs.append(
                {
                    "simulated_request_id": simulated_request_id,
                    "assignment_unit_id": assignment_unit_id,
                    "hash_bucket": midpoint_lookup[treatment["treatment_id"]],
                    "treatment_id": treatment["treatment_id"],
                    "treatment_name": treatment["treatment_name"],
                    "is_control": int(bool(treatment["is_control"])),
                }
            )
        return runs

    assignment_unit_id = request_id if experiment["assignment_unit"] == "request_id" else user_id
    simulated_request_id = f"{request_id}::sim_round_{round_index}"
    hash_bucket = compute_hash_bucket(
        experiment_id=experiment["experiment_id"],
        salt=experiment["salt"],
        assignment_unit_id=assignment_unit_id,
    )
    treatment_id = choose_treatment(bucket=hash_bucket, treatments=experiment["treatments"])
    treatment = next(candidate for candidate in experiment["treatments"] if candidate["treatment_id"] == treatment_id)
    return [
        {
            "simulated_request_id": simulated_request_id,
            "assignment_unit_id": assignment_unit_id,
            "hash_bucket": hash_bucket,
            "treatment_id": treatment["treatment_id"],
            "treatment_name": treatment["treatment_name"],
            "is_control": int(bool(treatment["is_control"])),
        }
    ]


def build_treatment_midpoint_lookup(treatments: list[dict[str, Any]]) -> dict[str, float]:
    lookup: dict[str, float] = {}
    cumulative = 0.0
    for treatment in treatments:
        start = cumulative
        cumulative += float(treatment["weight"])
        lookup[str(treatment["treatment_id"])] = start + ((cumulative - start) / 2.0)
    return lookup


def deterministic_uniform(*, seed: int, key: str) -> float:
    digest = hashlib.md5(f"{seed}|{key}".encode("utf-8")).hexdigest()
    integer = int(digest[:12], 16)
    return integer / float(16**12)


def parse_base_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
