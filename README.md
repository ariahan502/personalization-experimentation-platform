# Personalization Experimentation Platform

This repository is a reproducible, config-driven personalization and experimentation project for a content platform feed.

The project is designed around a realistic product question:

- given a user session and a candidate set of content items, which items should the feed rank and show
- how should ranking interact with freshness, diversity, fatigue, and creator-spread constraints
- how should offline model changes be validated before and after an A/B experiment

The intended system is not a generic recommender demo. It is a practical offline-first personalization stack that mirrors how a real team would operate:

- event and session preparation
- candidate generation
- ranking and reranking
- experiment assignment
- metric attribution and A/B analysis
- reporting, monitoring, and run artifacts

The initial dataset choice is `MIND`, the Microsoft News Dataset, used as the foundation for a semi-synthetic event-log workflow that better resembles real feed requests than a plain benchmark table.

## Business Story

Imagine a content platform home feed with these recurring problems:

- new and light users do not have enough history for stable personalization
- returning users can be over-served the same topic or creator
- older high-CTR content can crowd out fresher content
- offline ranking metrics can improve while session quality or long-click quality does not
- teams need controlled experiments to know whether a new ranking treatment should ship

This project treats personalization as a decision system rather than a single model. The objective is to optimize session-level engagement under operational constraints, with experimentation as the final validation layer.

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

This project is intended to be an end-to-end, code-first system project rather than a notebook portfolio exercise.

- reusable logic should live under `src/personalization_platform/`
- changeable settings should live under `configs/`
- entrypoints should run from `personalization_platform.pipeline`
- outputs should be written as reproducible artifact bundles under `artifacts/runs/`
- notebooks may still be used for exploration, but they are not the source of truth for production-style logic

The intended end-to-end flow is:

1. build a request-level event-log surface from MIND-derived inputs
2. generate multi-source candidates
3. train and evaluate a baseline ranker
4. apply explicit reranking constraints
5. assign experiments deterministically and analyze treatment results
6. emit monitoring, reporting, and optional local serving artifacts

The repo is not fully end-to-end yet. Today it validates the scaffold and the first schema contract. It becomes truly end-to-end once the event-log, retrieval, ranking, reranking, and experimentation commands all run from config and produce connected artifact bundles.

## System Scope

The target architecture is organized into six modules:

1. `data`
   Converts MIND into request-level, session-aware event tables and semi-synthetic feed logs.
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

- project scaffold aligned with the ads project structure
- README and roadmap for the full system direction
- local execution plan workspace under `doc/`
- package layout under `src/personalization_platform/`
- a minimal config-driven scaffold command for repository validation
- first event-log schema contract for requests, impressions, user state, and item state
- config-driven schema contract validation command that writes an artifact bundle

What comes next:

- Phase 1 minimal event-log build from MIND smoke inputs
- Phase 2 first retrieval-plus-ranking path
- Phase 3 reranking constraints plus experimentation framework and readout
- Phase 4 monitoring, delivery polish, and portfolio-ready reporting

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

The planned data flow starts from MIND and reshapes it into a product-style log surface:

- impression and click events
- inferred request groups
- session boundaries
- user-state snapshots
- item-state snapshots
- semi-synthetic fields for freshness decay, fatigue, and experiment assignment

The goal is not to invent unrealistic behavior. The goal is to anchor the log distribution in a real public dataset and then add operational fields needed for ranking and experimentation workflows.

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

## Quick Validation

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

This command validates the first-pass schema contract that downstream MIND conversion, retrieval, ranking, and experimentation tickets will build against.

A human-readable version of the contract lives in [src/personalization_platform/data/event_log_schema_contract.md](/Users/hanlingjuan/personalization-experimentation-platform/src/personalization_platform/data/event_log_schema_contract.md).

The repo also includes config contracts for the future event-log build:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.validate_event_log_config --config configs/mind_smoke.yaml
PYTHONPATH=src python -m personalization_platform.pipeline.validate_event_log_config --config configs/mind_full.yaml
```

These configs make the event-log input contract explicit:

- `input.source_mode` distinguishes smoke fixtures from raw MIND inputs
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

## Delivery Philosophy

This project should stay honest and useful:

- not just another MIND benchmark leaderboard attempt
- not just another CTR model notebook
- not an experimentation slide deck without operational logic

The target outcome is a credible feed-personalization project that shows system thinking across offline ranking, operational constraints, and A/B validation.
