# Training vs Serving Contract

This document defines the current boundary between the offline ranking feature surface and the local serving surface.

The code-backed source of truth lives in `personalization_platform.features.contracts`.

## Why This Exists

- the offline ranker trains on artifact-rich batch features
- the local API only has a narrow request-time surface
- productionization work should make that gap explicit rather than quietly blur it

## Offline Training Surface

The current baseline ranker trains on:

- retrieval ordering features such as `merged_rank`, `normalized_merged_rank`, `merged_score`, and `source_rank`
- provenance features such as `source_count`, `has_affinity_source`, and `has_multi_source_provenance`
- request context such as `candidate_count`, `history_length`, `history_click_count`, `request_index_in_session`, and `request_hour`
- categorical context such as `candidate_source` and `topic`

These fields come from the generated ranking dataset bundle, not from a live serving request.

## Serving Surface

The local API exposes two explicit modes:

- `fixture_replay`
- `contextual_scoring`
- `request_time_assembly`

The contextual scoring schema is intentionally narrow:

- request fields: `request_id`, `top_k`, `user_id`, `history_item_ids`, `history_topics`, `candidate_items`
- candidate fields: `item_id`, `topic`, `creator_id`, `publisher`, `title`, `candidate_source`

Unknown fields are rejected by the API schema rather than silently ignored.

If `candidate_items` are omitted, the local API can now assemble request-time candidates from lightweight local `affinity`, `content`, and `trending` retrieval sources before scoring.

## Shared vs Unsupported

Shared or request-time derivable fields today:

- `candidate_source`
- `topic`
- `candidate_count`
- `history_length`

Current contextual scoring features:

- `normalized_topic_affinity`
- `seen_history_penalty`
- `click_prior`
- `impression_prior`
- `recent_topic_click_share`
- `recent_item_ctr`
- `recent_user_click_rate`
- `freshness_minutes_since_last_seen`
- `freshness_bonus`

Unsupported online training features today include:

- offline retrieval ordering fields such as `merged_rank` and `merged_score`
- merged provenance fields such as `source_count` and `has_multi_source_provenance`
- live-session fields such as `request_index_in_session`
- label-leaking fields such as `candidate_seen_in_impressions`

These are not approximated by the serving API today.

## Implication

The current local API is useful for smoke validation and request-shape design, but it is not yet a production parity surface for the offline ranker.

The serving smoke workflow now also writes local request, exposure, response, and click log files so future experiment assignment and offline rebuild work can consume a serving-originated interaction surface instead of only replay artifacts.
