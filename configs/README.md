# Configs

This directory stores config-backed entrypoints for the personalization project.

Validation tiers:

- `bash scripts/ci_smoke.sh` is the fastest whole-repo health check
- `bash scripts/ci_medium.sh` is the richer retrieval-and-ranking validation path
- medium validation should be the default follow-up after substantive retrieval or ranking changes

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

The experiment analysis config also carries history-depth segment definitions for treatment-slice diagnostics.

Monitoring configs:

- `monitoring_smoke.yaml`
- `model_lifecycle_smoke.yaml`

Delivery configs:

- `local_api.yaml`
- `portfolio_report_smoke.yaml`

The local API config now includes replay-style smoke requests, contextual candidate-payload smoke requests, and a request-time assembly smoke request. Its smoke bundle also emits a `serving_contract.json` artifact describing the accepted request schema and the offline-only ranking features that are not yet supported online, plus local request/exposure/response/click log files for the serving smoke flow.

The same config also points at the latest completed `local_api_smoke` bundle under `artifacts/runs/` so the serving path can hydrate fresh request-time features from prior serving logs when they exist.

It also defines request-time assembly controls such as `max_sources_per_request` and `fallback_to_trending_only`, which are reflected in the local API health output and in per-response degraded-mode reporting.

The local API config now also includes an `experiment` section so serving requests can receive deterministic treatment assignment and write treatment-aware exposure logs.
Its default smoke request lists use distinct user IDs so the local serving bundle covers both the `control` and `reranked_policy` treatments with a slightly richer traffic mix.

There is also a live-style experiment readout config:

- `live_experiment_smoke.yaml`
- `serving_simulation_smoke.yaml`
- `live_experiment_simulated_smoke.yaml`
- `model_lifecycle_simulated_smoke.yaml`

It analyzes the latest `local_api_smoke` serving logs and emits treatment metrics plus guardrails directly from request-time request/exposure/click records.

The lifecycle config then combines the latest ranker, ranker comparison, offline monitoring, and live-style experiment readout bundles to emit an explicit `promote`, `hold`, or `rollback` recommendation plus fallback guidance.

The simulated-live configs add a stronger validation tier: they build a deterministic serving-log bundle from reranked outputs with fixed seeds, then run the same live readout and lifecycle logic on top of that larger replay-style sample.
