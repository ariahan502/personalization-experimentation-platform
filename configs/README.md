# Configs

This directory stores config-backed entrypoints for the personalization project.

Planned config families:

- event-log conversion
- candidate generation
- ranking
- reranking
- replay-style policy evaluation
- experiment assignment
- experiment analysis
- monitoring and drift

The current scaffold config:

- `project_scaffold.yaml`

Schema and event-log foundation configs:

- `event_log_schema.yaml`
- `mind_smoke.yaml`
- `mind_medium.yaml`
- `mind_full.yaml`

Retrieval configs:

- `candidates_smoke.yaml`
- `candidates_medium.yaml`

Ranking configs:

- `ranking_dataset_smoke.yaml`
- `ranking_dataset_medium.yaml`
- `ranker_smoke.yaml`
- `ranker_medium.yaml`
- `ranker_tree_medium.yaml`
- `ranker_compare_smoke.yaml`
- `ranker_compare_medium.yaml`

Reranking configs:

- `rerank_smoke.yaml`

Experiment configs:

- `experiment_smoke.yaml`
- `experiment_analysis_smoke.yaml`

Monitoring configs:

- `monitoring_smoke.yaml`

Delivery configs:

- `local_api.yaml`
- `portfolio_report_smoke.yaml`
