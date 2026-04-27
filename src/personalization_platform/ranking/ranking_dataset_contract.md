# Ranking Dataset Contract

This document defines the first reusable training dataset surface for the baseline ranking pipeline.

The code-backed builder for this contract lives in `personalization_platform.ranking.dataset`.

## Row Grain

One row per request-item candidate pair.

Each row joins:

- request context from the event-log surface
- merged candidate metadata from retrieval
- binary click label from the impression table when present

## Label Definition

- `label` is a binary click target.
- `label = 1` when the candidate item appears in the impression table for the same request and has `clicked = 1`.
- `label = 0` otherwise.

This means first-pass negatives include:

- shown but not clicked items
- retrieved candidates that were not part of the original served impression list

## Split Logic

The smoke dataset uses a simple time-ordered split:

- the most recent request timestamp bucket becomes `valid`
- earlier rows become `train`

This is intentionally narrow and reproducible for smoke validation rather than a production-ready evaluation policy.

## Feature Surface

### Candidate Ordering Features

- `merged_rank`
- `normalized_merged_rank`
- `merged_score`
- `source_rank`

### Provenance Features

- `candidate_source`
- `source_count`
- `has_multi_source_provenance`
- `has_affinity_source`
- `has_trending_source`
- `is_affinity_primary`
- `is_trending_primary`

### Request Context Features

- `candidate_count`
- `history_length`
- `history_click_count`
- `is_cold_start`
- `request_index_in_session`
- `request_hour`

### Topic Features

- `topic`
- `topic_history_count`

### Debug / Analysis Fields

- `candidate_seen_in_impressions`
- `observed_position`
- `source_list`
- `source_details`
- `request_ts`
- `split`

## Output Shape

The smoke builder writes:

- `data/processed/ranking_dataset/<timestamp>_ranking_dataset_smoke/ranking_dataset.csv`

And a run bundle with:

- `config.yaml`
- `metrics.json`
- `manifest.json`

## Assumptions

- the first baseline keeps features explainable and easy to debug
- no advanced negative sampling is applied yet
- freshness and diversity remain visible as future explicit signals rather than hidden transformations
