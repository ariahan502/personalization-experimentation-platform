# Event-Log Schema Contract

This document defines the first stable event-log surface for MIND-derived smoke pipelines.

The goal is to provide enough structure for downstream retrieval, ranking, reranking, and experimentation work without over-designing the full platform too early.

The code-backed source of truth for the contract lives in `personalization_platform.data.event_log_schema`.

## Tables

### `requests`

Row grain:
One row per feed request derived from a user impression group within a session.

Required columns:

- `request_id` (`string`): stable request key derived from split, impression lineage, and row position
- `user_id` (`string`): MIND user identifier
- `session_id` (`string`): inferred session identifier
- `request_ts` (`timestamp`): request timestamp from `behaviors.tsv`
- `split` (`string`): dataset split used for the row
- `candidate_count` (`int`): number of candidate impressions in the request

Optional first-pass columns:

- `history_length` (`int`)
- `request_index_in_session` (`int`)

Future columns:

- `device_type` (`string`)
- `experiment_unit_id` (`string`)

### `impressions`

Row grain:
One row per candidate item shown within a request.

Required columns:

- `impression_id` (`string`): stable request-item exposure identifier
- `request_id` (`string`): parent request key
- `user_id` (`string`): repeated user identifier
- `item_id` (`string`): candidate item identifier
- `position` (`int`): display position in the request
- `clicked` (`int`): binary click label

Optional first-pass columns:

- `topic` (`string`)
- `candidate_source` (`string`)

Future columns:

- `dwell_seconds` (`float`)
- `served_rank_score` (`float`)

### `user_state`

Row grain:
One row per request capturing the user state visible before ranking that request.

Required columns:

- `request_id` (`string`)
- `user_id` (`string`)
- `history_item_ids` (`array[string]`)
- `history_click_count` (`int`)
- `is_cold_start` (`bool`)

Optional first-pass columns:

- `recent_topic_counts` (`map[string,int]`)

Future columns:

- `fatigue_state` (`map[string,float]`)
- `subscription_tier` (`string`)

### `item_state`

Row grain:
One row per content item referenced by the event-log slice.

Required columns:

- `item_id` (`string`)
- `topic` (`string`)
- `subcategory` (`string`)
- `title` (`string`)
- `publisher` (`string`)
- `published_ts` (`timestamp`)

Optional first-pass columns:

- `abstract` (`string`)
- `entity_ids` (`array[string]`)

Future columns:

- `freshness_hours` (`float`)
- `creator_id` (`string`)

## MIND Mapping Notes

- `behaviors.tsv.user_id` maps directly to request, impression, and user-state user keys.
- `behaviors.tsv.time` maps to `requests.request_ts` and informs inferred `session_id`.
- `behaviors.tsv.history` maps to `user_state.history_item_ids`.
- `behaviors.tsv.impressions` explodes into item-level impression rows and click labels.
- `news.tsv.news_id`, `category`, `subcategory`, `title`, and `abstract` populate `item_state`.

## Assumptions

- A single MIND behaviors row is treated as the starting point for one request.
- Stable request and session identifiers are inferred because MIND does not provide both explicitly.
- User state is point-in-time and limited to information visible before the request.
- Publisher, creator, freshness, and experiment metadata may require derivation or semi-synthetic enrichment later.
- Binary clicks are the only required engagement label in the first slice.
