from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class ScoreFeedRequest(BaseModel):
    request_id: str = Field(..., description="Fixture request identifier from the local smoke pipeline.")
    top_k: int = Field(default=10, ge=1, le=50, description="Maximum number of ranked items to return.")
    user_id: str | None = Field(default=None, description="Optional user identifier for request-level consistency checks.")


class RankedItem(BaseModel):
    item_id: str
    topic: str
    creator_id: str
    candidate_source: str
    pre_rank: int
    post_rank: int
    rank_shift: int
    prediction: float
    rerank_score: float
    freshness_bonus: float


class ScoreFeedResponse(BaseModel):
    api_name: str
    mode: str
    request_id: str
    user_id: str
    dataset_split: str
    source_rerank_dir: str
    available_item_count: int
    returned_item_count: int
    items: list[RankedItem]
    assumptions: list[str]


def create_local_api_app(config: dict[str, Any]) -> FastAPI:
    rerank_dir = resolve_run_dir(
        base_dir=config["input"]["rerank_base_dir"],
        run_name=config["input"]["rerank_run_name"],
    )
    reranked_rows = pd.read_csv(rerank_dir / "reranked_rows.csv")
    api_config = config["api"]
    request_index = build_request_index(reranked_rows)
    assumptions = [
        "This API replays the latest local reranked smoke outputs rather than performing live feature retrieval.",
        "Request identifiers must come from the local fixture-compatible pipeline outputs.",
        "Returned scores and ranks are intended for local demo and integration validation only.",
    ]

    app = FastAPI(
        title=api_config.get("title", "Personalization Local API"),
        description="Local fixture-backed ranked feed demo API.",
        version="0.1.0",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "api_name": api_config.get("api_name", "local_ranked_feed_api"),
            "mode": "fixture_replay",
            "source_rerank_dir": str(rerank_dir),
            "request_count": len(request_index),
        }

    @app.post("/score/feed", response_model=ScoreFeedResponse)
    def score_feed(payload: ScoreFeedRequest) -> ScoreFeedResponse:
        if payload.request_id not in request_index:
            raise HTTPException(status_code=404, detail=f"Unknown request_id '{payload.request_id}'.")

        request_rows = request_index[payload.request_id].copy()
        request_user_id = str(request_rows.iloc[0]["user_id"])
        if payload.user_id is not None and payload.user_id != request_user_id:
            raise HTTPException(
                status_code=400,
                detail=f"Provided user_id '{payload.user_id}' does not match request owner '{request_user_id}'.",
            )

        ranked = request_rows.sort_values(["post_rank", "item_id"], ascending=[True, True]).head(payload.top_k)
        items = [
            RankedItem(
                item_id=str(row["item_id"]),
                topic=str(row["topic"]),
                creator_id=str(row["creator_id"]),
                candidate_source=str(row["candidate_source"]),
                pre_rank=int(row["pre_rank"]),
                post_rank=int(row["post_rank"]),
                rank_shift=int(row["rank_shift"]),
                prediction=float(row["prediction"]),
                rerank_score=float(row["rerank_score"]),
                freshness_bonus=float(row["freshness_bonus"]),
            )
            for row in ranked.to_dict(orient="records")
        ]
        return ScoreFeedResponse(
            api_name=api_config.get("api_name", "local_ranked_feed_api"),
            mode="fixture_replay",
            request_id=str(payload.request_id),
            user_id=request_user_id,
            dataset_split=str(request_rows.iloc[0]["dataset_split"]),
            source_rerank_dir=str(rerank_dir),
            available_item_count=int(len(request_rows)),
            returned_item_count=int(len(items)),
            items=items,
            assumptions=assumptions,
        )

    return app


def build_request_index(reranked_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(request_id): frame.copy()
        for request_id, frame in reranked_rows.groupby("request_id", sort=False)
    }


def resolve_run_dir(*, base_dir: str, run_name: str) -> Path:
    matches = sorted(Path(base_dir).glob(f"*_{run_name}"))
    if not matches:
        raise FileNotFoundError(f"No outputs found under {base_dir} matching '*_{run_name}'.")
    return matches[-1]
