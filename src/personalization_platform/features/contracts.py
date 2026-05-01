from __future__ import annotations

from typing import Any


DEFAULT_NUMERIC_TRAINING_FEATURES = [
    "merged_rank",
    "normalized_merged_rank",
    "merged_score",
    "source_rank",
    "source_count",
    "candidate_count",
    "history_length",
    "history_click_count",
    "topic_history_count",
    "request_index_in_session",
    "request_hour",
]

DEFAULT_BINARY_TRAINING_FEATURES = [
    "has_multi_source_provenance",
    "has_affinity_source",
    "has_trending_source",
    "is_affinity_primary",
    "is_trending_primary",
    "is_cold_start",
    "candidate_seen_in_impressions",
]

DEFAULT_CATEGORICAL_TRAINING_FEATURES = [
    "candidate_source",
    "topic",
]

SERVING_REQUEST_FIELDS = {
    "request_id": {
        "required": False,
        "description": "Optional fixture request identifier for replay mode or caller-provided request identifier.",
    },
    "top_k": {
        "required": False,
        "description": "Maximum ranked items to return.",
    },
    "user_id": {
        "required": False,
        "description": "Optional user identifier for replay validation or contextual scoring.",
    },
    "history_item_ids": {
        "required": False,
        "description": "Optional request-time item history for contextual scoring.",
    },
    "history_topics": {
        "required": False,
        "description": "Optional request-time topic history for contextual scoring.",
    },
    "candidate_items": {
        "required": False,
        "description": "Optional caller-provided candidate set; omitted when request-time assembly is used.",
    },
}

SERVING_CANDIDATE_FIELDS = {
    "item_id": {
        "required": True,
        "description": "Candidate item identifier to score.",
    },
    "topic": {
        "required": False,
        "description": "Optional request-time topic override for unseen items.",
    },
    "creator_id": {
        "required": False,
        "description": "Optional request-time creator override for unseen items.",
    },
    "publisher": {
        "required": False,
        "description": "Optional request-time publisher override for unseen items.",
    },
    "title": {
        "required": False,
        "description": "Optional request-time title override for unseen items.",
    },
    "candidate_source": {
        "required": False,
        "description": "Optional upstream source label carried into serving outputs.",
    },
}


def build_training_serving_feature_contract() -> dict[str, Any]:
    training_features = {
        "numeric": list(DEFAULT_NUMERIC_TRAINING_FEATURES),
        "binary": list(DEFAULT_BINARY_TRAINING_FEATURES),
        "categorical": list(DEFAULT_CATEGORICAL_TRAINING_FEATURES),
    }
    shared_request_time_fields = [
        "candidate_source",
        "topic",
        "candidate_count",
        "history_length",
    ]
    current_contextual_features = [
        "normalized_topic_affinity",
        "seen_history_penalty",
        "click_prior",
        "impression_prior",
        "recent_topic_click_share",
        "recent_item_ctr",
        "recent_user_click_rate",
        "freshness_minutes_since_last_seen",
        "freshness_bonus",
    ]
    unsupported_online_training_features = [
        {
            "feature": "merged_rank",
            "reason": "Depends on offline merged retrieval ordering that the current API does not compute at request time.",
        },
        {
            "feature": "normalized_merged_rank",
            "reason": "Depends on offline merged retrieval ordering that the current API does not compute at request time.",
        },
        {
            "feature": "merged_score",
            "reason": "Depends on offline merged retrieval scoring that is not supplied in the serving request schema.",
        },
        {
            "feature": "source_rank",
            "reason": "Depends on per-source retrieval ordering that is not supplied in the serving request schema.",
        },
        {
            "feature": "source_count",
            "reason": "Depends on merged multi-source provenance that is not supplied in the serving request schema.",
        },
        {
            "feature": "history_click_count",
            "reason": "Requires a reliable live aggregation of prior clicks; the current API only supports lightweight history payloads and local fallbacks.",
        },
        {
            "feature": "topic_history_count",
            "reason": "The current API derives a simplified affinity score from history topics rather than preserving the exact offline feature column.",
        },
        {
            "feature": "request_index_in_session",
            "reason": "Requires live session state that the current API does not maintain.",
        },
        {
            "feature": "has_multi_source_provenance",
            "reason": "Requires merged retrieval provenance across sources that the current API does not maintain.",
        },
        {
            "feature": "has_affinity_source",
            "reason": "Requires candidate provenance across retrieval sources that the current API does not maintain.",
        },
        {
            "feature": "has_trending_source",
            "reason": "Requires candidate provenance across retrieval sources that the current API does not maintain.",
        },
        {
            "feature": "is_affinity_primary",
            "reason": "Requires source-priority bookkeeping that the current API does not maintain.",
        },
        {
            "feature": "is_trending_primary",
            "reason": "Requires source-priority bookkeeping that the current API does not maintain.",
        },
        {
            "feature": "candidate_seen_in_impressions",
            "reason": "Leaks served-impression history from offline labels and is not a valid request-time feature.",
        },
    ]
    return {
        "contract_version": "v1",
        "serving_modes": {
            "fixture_replay": {
                "description": "Replay the latest artifact-backed reranked rows for a known fixture request.",
                "required_request_fields": ["request_id"],
            },
            "contextual_scoring": {
                "description": "Score caller-provided candidates using request-time history plus local priors.",
                "required_request_fields": ["candidate_items"],
            },
            "request_time_assembly": {
                "description": "Assemble request-time candidates from local retrieval sources, then score and rerank them.",
                "required_request_fields": ["user_id or request history"],
            },
        },
        "serving_request_schema": {
            "extra_fields_policy": "forbid",
            "request_fields": SERVING_REQUEST_FIELDS,
            "candidate_item_fields": SERVING_CANDIDATE_FIELDS,
        },
        "training_feature_inventory": training_features,
        "shared_request_time_fields": shared_request_time_fields,
        "current_contextual_features": current_contextual_features,
        "request_time_candidate_assembly": {
            "enabled_sources": ["affinity", "content", "trending"],
            "expected_output_fields": [
                "candidate_source",
                "merged_rank",
                "merged_score",
                "source_rank",
                "source_count",
                "source_list",
                "source_details",
            ],
        },
        "unsupported_online_training_features": unsupported_online_training_features,
        "assumptions": [
            "The local API intentionally exposes a narrower request-time contract than the offline training dataset.",
            "Contextual scoring is allowed to reuse local priors and item metadata, but it does not pretend to reproduce the offline ranker feature matrix exactly.",
            "Features that depend on merged retrieval provenance, offline labels, or live session counters are marked unsupported instead of silently approximated.",
        ],
    }
