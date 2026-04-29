# Personalization Experimentation Platform Roadmap

## Goal

Build and maintain a reproducible offline personalization and experimentation pipeline for a content-platform feed, currently using MIND as the raw interaction source.

The project will support data preparation, candidate generation, ranking, reranking constraints, replay-style evaluation, experiment assignment, A/B analysis, reporting, and quality checks.

## Principles

- Treat personalization as a decision system, not only a model-training task.
- Keep retrieval, ranking, reranking, experimentation, and reporting clearly separated.
- Prefer reproducible, config-driven workflows over notebook-only analysis.
- Treat offline ranking metrics as screening tools and online experiments as the final validation mechanism.
- Keep operational constraints such as freshness, diversity, fatigue, and creator spread explicit.
- Keep smoke paths self-contained so the repository can be validated without the full raw dataset.

## Execution Pattern

The project is built stage by stage:

1. define a minimal contract for the next layer
2. implement reusable package code under `src/personalization_platform/`
3. add a config-backed pipeline entrypoint
4. validate the stage on smoke inputs
5. write artifact bundles that the next stage can consume

This pattern is used across event-log preparation, retrieval, ranking, reranking, experimentation, monitoring, delivery, and reporting.

## Current Capabilities

- Project scaffold aligned with a reproducible `src/` Python layout
- Config-driven event-log schema and input contracts
- Smoke event-log build from fixture-backed feed-interaction inputs
- Medium validation path for richer local ranking evaluation
- Multi-source retrieval with affinity, content, and trending candidate sources
- Baseline ranking dataset, logistic and tree-based rankers, and segmented comparison diagnostics
- Explicit reranking policy for freshness, diversity, and creator spread
- Deterministic experiment assignment plus offline experiment readout with multiple outcomes, guardrails, slices, and SRM
- Offline monitoring bundle for funnel coverage, score stability, and experiment integrity
- Optional local ranked-feed API with replay and contextual scoring modes
- Portfolio-facing reporting bundle and architecture note
- Targeted stage-level tests plus GitHub Actions CI
- One-command smoke validation via `bash scripts/ci_smoke.sh`

## Roadmap Status

The original four-phase roadmap has now been implemented as a smoke-validated end-to-end system.

- Phase 1: complete
- Phase 2: complete
- Phase 3: complete
- Phase 4: complete

The repo now supports a reproducible path from event-log preparation through delivery/demo artifacts using only local configs and fixture-compatible assets.

The extension phase is also complete for the current scope:

- richer validation data
- richer item and creator metadata
- deeper retrieval
- multiple ranker families
- segmented diagnostics
- stronger experiment readout
- engineering hardening
- more realistic serving/demo behavior

## Next Expansion Opportunities

### 1. Data Realism And Scale

Extend the smoke-sized setup into a somewhat richer offline evaluation environment without losing reproducibility.

Expected outputs:

- larger or more varied validation fixtures
- richer creator and content metadata
- stronger valid/test splits for ranking and experimentation

Validation:

- the current smoke path still passes
- expanded fixtures produce stable bundles and more informative metrics

### 2. Modeling Depth

Go beyond the first interpretable baseline while keeping the comparison frame explicit.

Expected outputs:

- stronger ranker variants
- richer retrieval features or sources
- deeper offline evaluation slices

Validation:

- each new model still compares against the simpler fallback path
- metrics remain reproducible from config-backed runs

### 3. Engineering Hardening

Improve maintainability and environment reproducibility.

Expected outputs:

- more targeted tests
- CI automation
- tighter dependency management or isolated environment guidance

Validation:

- smoke command remains the primary high-signal health check
- incremental tests catch stage-level regressions earlier

Status: mostly complete for the current local-scope project; future work would focus on environment packaging and reproducibility polish rather than missing core checks.

### 4. Presentation And Storytelling

Keep improving the clarity of the project story for README, demo, and portfolio use.

Expected outputs:

- cleaner report excerpts for the README
- demo instructions or screenshots
- stronger architecture storytelling around tradeoffs

Validation:

- the repo remains easy to understand for a new reader
- the reporting bundle continues to reflect real artifact outputs

## Phase Exit Criteria

### Phase 1 Exit

Phase 1 is complete when the repo can build a smoke event-log bundle from config, write request and impression outputs plus minimal user-state and item-state tables, and document the assumptions required because MIND is not a production event log.

### Phase 2 Exit

Phase 2 is complete when the repo can generate multi-source candidates, train a baseline ranker, and emit a comparison bundle showing whether the ranker improves on a simpler baseline using clearly defined offline metrics.

### Phase 3 Exit

Phase 3 is complete when assignment is deterministic, experiment manifests are explicit, and the analysis workflow emits primary metrics, guardrails, and SRM checks in a reproducible artifact bundle.

### Phase 4 Exit

Phase 4 is complete when the project can demo a fixture-sized ranked-feed flow, emit a smoke-quality pass, and produce a concise reporting bundle that is suitable for README or resume-backed project storytelling.

Status: complete.

## Operating Notes

- Large raw data, generated working tables, and run artifacts stay out of Git.
- `doc/` is a local planning workspace and should not be relied on as tracked source of truth unless explicitly promoted.
- Run bundles are written under `artifacts/runs/<timestamp>_<run_name>/`.
- MIND remains the current raw source, but the project builds additional request-level and operational fields when the source data does not expose them directly.
