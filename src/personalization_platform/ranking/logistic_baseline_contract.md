# Logistic Baseline Ranker Contract

This document defines the first interpretable ranking model workflow for the project.

The code-backed implementation lives in `personalization_platform.ranking.logistic_baseline`.

## Model Choice

- model family: logistic regression
- objective: binary click prediction on request-item candidate rows
- goal: provide a transparent, reproducible baseline rather than maximize performance

## Training Surface

Input dataset:

- `data/processed/ranking_dataset/<timestamp>_ranking_dataset_smoke/ranking_dataset.csv`

Row grain:

- one row per request-item candidate pair

## Feature Groups

### Numeric

- `merged_rank`
- `normalized_merged_rank`
- `merged_score`
- `source_rank`
- `source_count`
- `candidate_count`
- `history_length`
- `history_click_count`
- `topic_history_count`
- `request_index_in_session`
- `request_hour`

### Binary

- `has_multi_source_provenance`
- `has_affinity_source`
- `has_trending_source`
- `is_affinity_primary`
- `is_trending_primary`
- `is_cold_start`
- `candidate_seen_in_impressions`

### Categorical

- `candidate_source`
- `topic`

## Evaluation Outputs

Classification metrics:

- accuracy
- log loss
- ROC AUC when both classes are present

Request-level ranking metrics:

- mean reciprocal rank
- hit rate at 1
- hit rate at 3

## Output Shape

The smoke trainer writes:

- `data/processed/ranker/<timestamp>_ranker_smoke/model.pkl`
- `data/processed/ranker/<timestamp>_ranker_smoke/scored_rows.csv`

And a run bundle with:

- `config.yaml`
- `metrics.json`
- `manifest.json`

## Assumptions

- the first baseline favors interpretability and inspectability
- smoke metrics are small-sample sanity checks, not decision-quality evidence
- later comparison work should reuse this exact training surface before introducing more model complexity
