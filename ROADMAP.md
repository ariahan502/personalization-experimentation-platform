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
- Roadmap and execution planning workspace
- Package skeleton for data, retrieval, ranking, reranking, experiments, evaluation, and pipeline code

## Near-Term Roadmap

### 1. MIND Event-Log Foundation

Convert MIND impressions into a request-level, session-aware dataset surface and define the first schema contracts.

Expected outputs:

- documented raw-to-event-log conversion path
- schema contract for requests, impressions, users, and items
- smoke fixture path

Validation:

- event-log build runs from config
- output tables match documented schema

### 2. Baseline Personalization Stack

Add candidate generation, a first ranker, and offline ranking diagnostics.

Expected outputs:

- multi-source candidate builder
- baseline ranking pipeline
- ranking evaluation report

Validation:

- baseline training runs from config
- retrieval and ranking metrics are written to a run bundle

### 3. Experimentation Layer

Add deterministic experiment assignment and A/B readout workflows.

Expected outputs:

- assignment logic
- experiment manifest and treatment definitions
- A/B analysis artifact bundle

Validation:

- assignment is deterministic and reproducible
- experiment summary and guardrails are emitted from config

### 4. Delivery Polish

Add monitoring, optional local serving, and portfolio-quality reporting.

Expected outputs:

- drift and quality reports
- optional local scoring API
- business-facing summary artifacts

Validation:

- smoke quality command passes
- optional API smoke run returns ranked items from a small fixture input

## Operating Notes

- Large raw data, generated working tables, and run artifacts stay out of Git.
- `doc/` is a local planning workspace and should not be relied on as tracked source of truth unless explicitly promoted.
- Run bundles are written under `artifacts/runs/<timestamp>_<run_name>/`.
- MIND remains the anchor dataset, but the project will augment it with realistic semi-synthetic operational fields when needed for experimentation and ranking simulation.
