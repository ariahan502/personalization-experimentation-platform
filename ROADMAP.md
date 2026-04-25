# Personalization Experimentation Platform Roadmap

## Goal

Build and maintain a reproducible offline personalization and experimentation pipeline for a content-platform feed grounded in MIND-derived event logs.

The project will support data preparation, candidate generation, ranking, reranking constraints, replay-style evaluation, experiment assignment, A/B analysis, reporting, and quality checks.

## Principles

- Treat personalization as a decision system, not just a model-training task.
- Keep retrieval, ranking, reranking, experimentation, and reporting clearly separated.
- Prefer reproducible, config-driven workflows over notebook-only analysis.
- Treat offline ranking metrics as screening tools and online experiments as the final validation mechanism.
- Keep operational constraints such as freshness, diversity, fatigue, and creator spread explicit.
- Keep smoke paths self-contained so the repository can be validated without the full raw dataset.

## Current Capabilities

- Project scaffold aligned with a reproducible `src/` Python layout
- Config-driven scaffold validation command
- Config-driven event-log schema contract command
- Roadmap and execution planning workspace
- Package skeleton for data, retrieval, ranking, reranking, experiments, evaluation, and pipeline code
- First code-backed event-log schema contract for requests, impressions, user state, and item state

## Near-Term Roadmap

### 1. MIND Event-Log Foundation

Convert MIND impressions into a request-level, session-aware dataset surface with the smallest useful contract that downstream retrieval, ranking, and experimentation work can consume.

Expected outputs:

- documented raw-to-event-log conversion path
- minimal schema contract for requests, impressions, user state, and item state
- smoke fixture path
- config-backed smoke event-log build command

Validation:

- event-log build runs from config against smoke inputs
- output tables match the documented first-pass schema
- run bundle includes row counts, schema version, and explicit assumptions

### 2. Baseline Personalization Stack

Add candidate generation, a first ranker, and offline ranking diagnostics.

Expected outputs:

- multi-source candidate builder
- baseline ranking pipeline
- ranking evaluation report

Validation:

- baseline training runs from config
- retrieval and ranking metrics are written to a run bundle
- ranker performance is compared against at least one simpler fallback such as trending-only ordering
- evaluation bundle includes at least one top-line ranking metric and one diagnostic view

### 3. Experimentation Layer

Add deterministic experiment assignment and A/B readout workflows.

Expected outputs:

- assignment logic
- experiment manifest and treatment definitions
- A/B analysis artifact bundle

Validation:

- assignment is deterministic and reproducible
- experiment summary, guardrails, and SRM checks are emitted from config
- treatment definitions and analysis assumptions are inspectable in the artifact bundle

### 4. Delivery Polish

Add monitoring, optional local serving, and portfolio-quality reporting.

Expected outputs:

- drift and quality reports
- optional local scoring API
- business-facing summary artifacts

Validation:

- smoke quality command passes
- optional API smoke run returns ranked items from a small fixture input
- reporting bundle explains technical results, business tradeoffs, and offline caveats clearly enough for portfolio use

## Phase Exit Criteria

### Phase 1 Exit

Phase 1 is complete when the repo can build a smoke event-log bundle from config, write request and impression outputs plus minimal user-state and item-state tables, and document the assumptions required because MIND is not a production event log.

### Phase 2 Exit

Phase 2 is complete when the repo can generate multi-source candidates, train a baseline ranker, and emit a comparison bundle showing whether the ranker improves on a simpler baseline using clearly defined offline metrics.

### Phase 3 Exit

Phase 3 is complete when assignment is deterministic, experiment manifests are explicit, and the analysis workflow emits primary metrics, guardrails, and SRM checks in a reproducible artifact bundle.

### Phase 4 Exit

Phase 4 is complete when the project can demo a fixture-sized ranked-feed flow, emit a smoke-quality pass, and produce a concise reporting bundle that is suitable for README or resume-backed project storytelling.

## Operating Notes

- Large raw data, generated working tables, and run artifacts stay out of Git.
- `doc/` is a local planning workspace and should not be relied on as tracked source of truth unless explicitly promoted.
- Run bundles are written under `artifacts/runs/<timestamp>_<run_name>/`.
- MIND remains the anchor dataset, but the project will augment it with realistic semi-synthetic operational fields when needed for experimentation and ranking simulation.
