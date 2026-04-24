# AGENTS.md

This file describes how to work effectively in this repository as an engineering agent.

## Project Intent

This repo is being built as a reproducible offline personalization and experimentation project for a content-platform feed.

The goal is not to build the most complex recommender stack immediately. The goal is to move in small, validated slices toward a realistic system that combines:

- MIND-based event-log preparation
- candidate generation
- ranking
- reranking constraints
- experiment assignment and analysis
- reporting and monitoring

## Working Style

When making progress in this repo:

- prefer small end-to-end slices over large speculative refactors
- leave the repo in a runnable state after each change
- favor reusable Python modules under `src/personalization_platform/` over notebook-only logic
- favor config-backed commands over hardcoded scripts
- validate changes with a real command whenever practical
- keep business constraints explicit rather than burying them inside model code

Good progress in this repo looks like:

- one new command
- one new reusable module
- one measurable validation result
- one clear next step

## New Session Bootstrap

If you are starting in a fresh session, do not rely on prior chat context. Rebuild context from the repo itself.

Use this sequence:

1. Read `AGENTS.md`
2. Check `git status`
3. Read `ROADMAP.md`
4. Read local planning docs if they exist:
   - `doc/execution-plan.md`
   - relevant `doc/feature-xxx/` notes
5. Inspect the implemented pipeline surface:
   - `configs/`
   - `src/personalization_platform/pipeline/`
   - `src/personalization_platform/ranking/`
   - `src/personalization_platform/retrieval/`
   - `src/personalization_platform/experiments/`
6. Validate the current baseline commands before starting new work when practical

Before making changes, summarize:

- current repo state
- current highest-value next task
- planned validation command

## Repo Structure

Tracked source-of-truth code should live in:

- `src/personalization_platform/`
- `configs/`
- `scripts/`
- `README.md`
- `ROADMAP.md`

Notebooks are useful, but they are not the long-term source of truth for reusable logic.

Use notebooks for:

- EDA
- narrative analysis
- temporary experimentation

Move logic out of notebooks when it needs to be repeated, validated, or composed with other steps.

## Data Rules

Do not commit the large raw dataset.

Current expectations:

- raw data lives under `data/raw/`
- intermediate working tables live under `data/interim/`
- processed modeling tables live under `data/processed/`
- generated local fixtures live under `data/fixtures/`
- run artifacts live under `artifacts/runs/`

Before introducing any new data dependency, make sure the path is explicit in config or documentation.

## Pipeline Rules

When adding a new workflow:

1. Put reusable logic in `src/personalization_platform/...`
2. Add a config file under `configs/` if the workflow has changeable parameters
3. Add a pipeline entrypoint under `src/personalization_platform/pipeline/`
4. Keep scripts as thin wrappers only if they still provide convenience
5. Validate the workflow by actually running it

Avoid burying operational settings directly inside notebook cells or one-off scripts.

## Artifact Rules

When a pipeline run produces outputs, prefer a stable run bundle shape like:

- `artifacts/runs/<timestamp>_<run_name>/config.yaml`
- `artifacts/runs/<timestamp>_<run_name>/metrics.json`
- `artifacts/runs/<timestamp>_<run_name>/manifest.json`

Generated artifacts should usually stay local unless the user explicitly wants them tracked.

## Planning Rules

The repo uses a local-only planning workspace under `doc/`.

Important:

- `doc/` is intentionally excluded via `.git/info/exclude`
- planning docs are for iterative local execution planning
- do not assume planning docs are meant to be committed unless the user asks

Use:

- `doc/execution-plan.md` for the tracker and overall execution sequence
- `doc/feature-xxx/` folders for breaking large items into agent-sized tasks

## Validation Expectations

Prefer real validation over theoretical claims.

Examples:

- run the scaffold validation command after changing repo plumbing
- run the event-log build command after changing MIND conversion logic
- run the ranking pipeline after changing retrieval or scoring logic
- inspect emitted metrics and artifact paths

If a full run is expensive, do a smoke run first.

If validation is skipped, say so clearly and explain why.

## Default Next Steps

Unless the user redirects, the next highest-value work is:

1. define the MIND-derived event schema
2. build a smoke event-log conversion path
3. add multi-source candidate generation
4. train a first baseline ranker
5. add experiment assignment and readout

If that work turns out to be too large, break it down in `doc/feature-event-log-foundation/` before coding.
