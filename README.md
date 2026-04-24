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

What comes next:

- Phase 1 baseline event-log build from MIND and a first retrieval-plus-ranking path
- Phase 2 reranking constraints and policy diagnostics
- Phase 3 experimentation framework and experiment readout
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

## Delivery Philosophy

This project should stay honest and useful:

- not just another MIND benchmark leaderboard attempt
- not just another CTR model notebook
- not an experimentation slide deck without operational logic

The target outcome is a credible feed-personalization project that shows system thinking across offline ranking, operational constraints, and A/B validation.
