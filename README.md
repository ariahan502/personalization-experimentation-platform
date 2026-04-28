# Personalization Experimentation Platform

This repository is a reproducible, config-driven personalization and experimentation system for a content feed.

The project is designed around a realistic product question:

- given a user session and a candidate set of content items, which items should the feed rank and show
- how should ranking interact with freshness, diversity, fatigue, and creator-spread constraints
- how should offline model changes be validated before and after an A/B experiment

The repository implements an offline-first personalization stack with the same major layers that appear in production feed systems:

- event and session preparation
- candidate generation
- ranking and reranking
- experiment assignment
- metric attribution and A/B analysis
- reporting, monitoring, and run artifacts

The current raw data source is `MIND`, the Microsoft News Dataset. In this repo, MIND is treated as a realistic input source for building a request-level feed-ranking workflow rather than as a benchmark table used in isolation.

## Product Context

This project is framed around common feed-ranking problems:

- new and light users do not have enough history for stable personalization
- returning users can be over-served the same topic or creator
- older high-CTR content can crowd out fresher content
- offline ranking metrics can improve while session quality or long-click quality does not
- teams need controlled experiments to know whether a new ranking treatment should ship

The project treats personalization as a decision system rather than a single model-training task. The objective is to optimize session-level engagement under explicit operational constraints, with experimentation as the decision layer for model and policy changes.

## Design Principles

- Build a realistic product system, not a benchmark-only model showcase.
- Keep the problem grounded in feed ranking and experimentation workflows that teams actually run.
- Prefer small, validated slices over large speculative builds.
- Treat offline metrics as pre-launch signals, not proof of product impact.
- Make constraints such as freshness, diversity, fatigue, and creator spread first-class design inputs.

## Why This Is Distinct From The Ads Project

This repository is meant to complement, not duplicate, the ads attribution project.

- The ads project focuses on acquisition-side decisioning: attribution, uplift diagnostics, policy simulation, and budget-oriented decision support.
- This project focuses on onsite engagement decisioning: candidate generation, feed ranking, reranking constraints, and experiment-driven product optimization.
- Both projects are decision systems, but they answer different business questions:
  - ads: which external traffic or campaigns create value
  - personalization: how internal feed inventory should be selected, ordered, constrained, and validated

Together, they are intended to show breadth across acquisition, engagement, offline modeling, and experimentation rather than repeating the same workflow in two repos.

## End-To-End Standard

This project is implemented as an end-to-end, code-first system rather than an analysis notebook.

- reusable logic should live under `src/personalization_platform/`
- changeable settings should live under `configs/`
- entrypoints should run from `personalization_platform.pipeline`
- outputs should be written as reproducible artifact bundles under `artifacts/runs/`
- notebooks may still be used for exploration, but they are not the source of truth for production-style logic

The intended end-to-end flow is:

1. build a request-level event-log surface from feed-interaction inputs derived from MIND
2. generate multi-source candidates
3. train and evaluate a baseline ranker
4. apply explicit reranking constraints
5. assign experiments deterministically and analyze treatment results
6. emit monitoring, reporting, and optional local serving artifacts

That end-to-end flow now exists in the repo as a smoke-validated path. The project can build the data surface, train and compare a baseline ranker, apply reranking policy logic, run experiment readouts, emit monitoring artifacts, replay ranked results through a local API, and package the results into a portfolio-facing report.

## System Scope

The target architecture is organized into six modules:

1. `data`
   Converts raw interaction inputs into request-level, session-aware event tables for downstream retrieval, ranking, and experimentation workflows.
2. `retrieval`
   Builds multi-source candidate sets such as trending, topic affinity, and history-based retrieval.
3. `ranking`
   Trains baseline rankers and produces personalized feed scores.
4. `reranking`
   Applies business and UX constraints such as freshness boosts, fatigue penalties, and creator caps.
5. `experiments`
   Handles deterministic bucketing, experiment configuration, guardrails, and A/B analysis.
6. `evaluation`
   Produces offline ranking metrics, strategy diagnostics, replay-style policy views, and experiment readouts.

## Current Repo Status

What exists today:

- request-level event-log build from fixture-backed feed-interaction inputs
- multi-source candidate generation using `affinity` plus `trending`
- baseline ranking dataset, logistic ranker, and fallback comparison bundle
- explicit reranking rules for freshness, topic diversity, and creator spread
- deterministic experiment assignment plus offline readout with guardrails and SRM
- monitoring bundle for funnel health, score stability, and experiment integrity
- local FastAPI demo surface for replaying ranked fixture requests
- portfolio-facing reporting bundle with a system summary and architecture note
- one-command repo smoke validation via `bash scripts/ci_smoke.sh`

What this means:

- the repo is now a coherent offline personalization system rather than a scaffold
- every major layer is config-driven and writes reproducible run bundles
- the current best validation path exercises the whole stack from scaffold through reporting

## Repository Layout

Tracked source-of-truth code is intended to live in:

- `src/personalization_platform/`
- `configs/`
- `scripts/`
- `README.md`
- `ROADMAP.md`

Local planning docs live under:

- `doc/`

Generated outputs should land under:

- `artifacts/runs/`

Local data should be organized under:

- `data/raw/`
- `data/interim/`
- `data/processed/`
- `data/fixtures/`

## Data Direction

The current implementation uses MIND as the raw source and reshapes it into a product-style log surface:

- impression and click events
- inferred request groups
- session boundaries
- user-state snapshots
- item-state snapshots
- inferred or config-backed fields for freshness, creator spread, and experiment assignment when the raw source does not expose them directly

This is an honest offline engineering pattern: use a real public interaction dataset, then build the missing request-level and operational surfaces needed for retrieval, ranking, reranking, and experimentation. The repo does not claim production traffic or online outcomes, but it does implement a real multi-stage system around realistic feed-ranking problems.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

For editable local development:

```bash
pip install -e ".[dev]"
```

Commands use the `src/` layout directly:

```bash
PYTHONPATH=src python -m ...
```

## Demo Path

If you want the fastest end-to-end validation, run:

```bash
bash scripts/ci_smoke.sh
```

This smoke script validates the whole project chain:

- scaffold and imports
- event-log build
- candidate generation
- ranking dataset and ranker
- fallback comparison
- reranking
- experiment assignment and readout
- monitoring
- local API replay
- portfolio reporting bundle

For a smaller demo flow after the smoke run succeeds:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.serve_ranked_feed --config configs/local_api.yaml
PYTHONPATH=src python -m personalization_platform.pipeline.build_portfolio_report --config configs/portfolio_report_smoke.yaml
```

The first command validates the local ranked-feed replay API. The second packages the latest smoke artifacts into a concise system summary and architecture note.

## Commands

The repo includes a minimal scaffold validation command:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.show_blueprint --config configs/project_scaffold.yaml
```

This writes a small run bundle under:

- `artifacts/runs/<timestamp>_project_scaffold/`

with:

- `config.yaml`
- `project_summary.json`

This command validates that the package, config loader, artifact writing path, and baseline repo structure are wired correctly.

The repo also includes a schema-contract command for the first event-log slice:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.describe_event_log_schema --config configs/event_log_schema.yaml
```

This writes a run bundle under `artifacts/runs/<timestamp>_event_log_schema/` with:

- `config.yaml`
- `schema_contract.json`
- `schema_summary.json`

This command validates the first-pass schema contract that downstream event-log conversion, retrieval, ranking, and experimentation stages build against.

A human-readable version of the contract lives in [src/personalization_platform/data/event_log_schema_contract.md](/Users/hanlingjuan/personalization-experimentation-platform/src/personalization_platform/data/event_log_schema_contract.md).

The repo also includes config contracts for the future event-log build:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.validate_event_log_config --config configs/mind_smoke.yaml
PYTHONPATH=src python -m personalization_platform.pipeline.validate_event_log_config --config configs/mind_full.yaml
```

These configs make the event-log input contract explicit:

- `input.source_mode` distinguishes smoke fixtures from raw dataset inputs
- `smoke_fixture.root_dir` defines the local smoke path under `data/fixtures/`
- `raw_input.root_dir` defines the raw dataset path under `data/raw/`
- `output.base_dir` defines where event-log tables should land
- `artifacts.base_dir` defines where run bundles should be written
- `validation.require_existing_inputs` controls whether future commands should fail fast on missing input paths

The first smoke event-log build command is:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.build_event_log --config configs/mind_smoke.yaml
```

This command reads the raw-like smoke fixture inputs, writes first-pass `requests`, `impressions`, `user_state`, and `item_state` tables under `data/interim/event_logs/<run_name>/`, and writes a run bundle with:

- `config.yaml`
- `metrics.json`
- `manifest.json`

The first smoke candidate-generation command is:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.build_candidates --config configs/candidates_smoke.yaml
```

This command reads the latest smoke event-log outputs, builds merged multi-source candidates from affinity plus trending retrieval, writes request-level candidates under `data/processed/candidates/<run_name>/`, and emits a run bundle with:

- `config.yaml`
- `metrics.json`
- `manifest.json`

The first smoke ranking-dataset command is:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.build_ranking_dataset --config configs/ranking_dataset_smoke.yaml
```

This command reads the latest smoke event-log and candidate outputs, writes one row per request-item candidate pair under `data/processed/ranking_dataset/<run_name>/`, and emits a run bundle with:

- `config.yaml`
- `metrics.json`
- `manifest.json`

The first smoke ranker-training command is:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.train_ranker --config configs/ranker_smoke.yaml
```

This command reads the latest smoke ranking dataset, trains a logistic-regression baseline ranker, writes scored rows and a serialized local model under `data/processed/ranker/<run_name>/`, and emits a run bundle with:

- `config.yaml`
- `metrics.json`
- `manifest.json`

The first smoke ranker-comparison command is:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.compare_rankers --config configs/ranker_compare_smoke.yaml
```

This command compares the logistic baseline against a simple retrieval-order fallback, and writes a diagnostics bundle with:

- `config.yaml`
- `metrics.json`
- `diagnostics.json`

The first smoke reranking command is:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.rerank_feed --config configs/rerank_smoke.yaml
```

This command applies explicit freshness, diversity, and creator-spread rules to the scored feed rows, writes reranked outputs under `data/processed/reranked_feed/<run_name>/`, and emits a run bundle with:

- `config.yaml`
- `metrics.json`
- `manifest.json`

The first smoke experiment-assignment command is:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.assign_experiment --config configs/experiment_smoke.yaml
```

This command applies deterministic control-versus-treatment assignment on top of the reranked request surface, writes assignment tables under `data/processed/experiment_assignment/<run_name>/`, and emits a run bundle with:

- `config.yaml`
- `metrics.json`
- `manifest.json`

The first smoke experiment-analysis command is:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.analyze_experiment --config configs/experiment_analysis_smoke.yaml
```

This command reads the latest assignment outputs, writes an offline experiment readout bundle with primary metrics, guardrails, and SRM checks, and emits:

- `config.yaml`
- `summary.json`
- `readout.json`

The first smoke monitoring command is:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.monitor_quality --config configs/monitoring_smoke.yaml
```

This command reads the latest smoke outputs across event-log, retrieval, ranking, reranking, and experiment analysis stages, and writes an offline monitoring bundle with:

- `config.yaml`
- `summary.json`
- `diagnostics.json`

For one-command repo health validation, run:

```bash
bash scripts/ci_smoke.sh
```

This smoke script keeps the project honest by running the lightweight import test plus the full smoke pipeline chain from scaffold validation through monitoring, local API replay, and the final reporting bundle. It is meant to be the default “did we break anything important?” command after incremental changes.

The optional local ranked-feed API can be smoke-tested with:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.serve_ranked_feed --config configs/local_api.yaml
```

This command builds a small FastAPI app backed by the latest local reranked smoke outputs, issues a fixture-compatible request through the API itself, and writes:

- `config.yaml`
- `summary.json`
- `smoke_response.json`
- `openapi_snapshot.json`

If you want to launch the server interactively instead of running the smoke check, use:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.serve_ranked_feed --config configs/local_api.yaml --serve
```

The portfolio-facing reporting bundle can be generated with:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.build_portfolio_report --config configs/portfolio_report_smoke.yaml
```

This command reuses the latest smoke artifacts and writes:

- `config.yaml`
- `executive_summary.json`
- `report_payload.json`
- `portfolio_report.md`
- `architecture_note.md`

## Positioning

This project should be read as a production-style offline system prototype:

- the raw source is public, but the engineering work is in the system built around it
- retrieval, ranking, reranking, experimentation, monitoring, delivery, and reporting are all explicit stages
- the repo is strongest as evidence of system design, pipeline engineering, and decision-oriented ML workflow design

The target outcome is a credible feed-personalization project that shows engineering depth across offline ranking, policy constraints, experiment structure, monitoring, and delivery.

## Likely Next Extensions

If this repo continues beyond the current portfolio baseline, the highest-value next investments are:

- richer smoke and validation data so offline metrics are less tiny-sample constrained
- stronger retrieval and ranking features beyond the first interpretable baseline
- more targeted tests for stage-level logic, not just import and smoke coverage
- optional GitHub Actions or tighter dependency pinning for cleaner environment reproducibility
