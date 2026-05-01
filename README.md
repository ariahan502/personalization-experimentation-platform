# Personalization Experimentation Platform

This repository is a reproducible, config-driven personalization and experimentation system for a content feed.

The project implements a full offline workflow for feed ranking:

- build request-level event logs from raw interaction inputs
- generate candidates from multiple retrieval sources
- train and compare a baseline ranker
- apply explicit reranking constraints
- assign and analyze experiments
- emit monitoring, API replay, and reporting artifacts

The current raw interaction source is `MIND`, the Microsoft News Dataset. In this repo, it is used to construct a request-level ranking workflow with explicit retrieval, ranking, reranking, experimentation, and delivery stages.

## At A Glance

- completed offline system for the current repo scope: event log -> retrieval -> ranking -> reranking -> experimentation -> monitoring -> local API -> reporting
- extension phase complete for the current repo scope: richer validation data, deeper retrieval, multiple ranker families, segmented diagnostics, stronger experiment readout, targeted tests, CI, and contextual API scoring
- fastest validation path: `bash scripts/ci_smoke.sh`
- richer retrieval-and-ranking validation path: `bash scripts/ci_medium.sh`
- strongest local treatment-and-lifecycle validation path: `bash scripts/ci_simulated_live.sh`
- production boundary: this repo is an offline experimentation platform and local demo surface, not a production recommender stack

## Process

The implementation follows a small-slice pipeline process:

- define a narrow contract for the next stage
- add a config-backed entrypoint
- run the stage against smoke inputs
- write reproducible run bundles
- connect that stage to the next downstream surface

The current end-to-end flow is:

1. build a request-level event-log surface from feed-interaction inputs derived from MIND
2. generate multi-source candidates
3. train and evaluate a baseline ranker
4. apply explicit reranking constraints
5. assign experiments deterministically and analyze treatment results
6. emit monitoring, reporting, and optional local serving artifacts

Each stage is implemented as package code under `src/personalization_platform/`, configured through `configs/`, and validated by writing artifact bundles under `artifacts/runs/`.

Run manifests now carry normalized lineage metadata:

- `run_metadata` records the current run id, timestamp, run name, artifact path, and output path when applicable
- `upstream_runs` records the labeled upstream run ids, run names, timestamps, and paths consumed by the current stage

This makes it possible to inspect stage provenance directly from `manifest.json` files instead of inferring lineage only from directory names.

The current highest-signal local commands are:

```bash
bash scripts/ci_smoke.sh
bash scripts/ci_medium.sh
bash scripts/ci_simulated_live.sh
PYTHONPATH=src python -m personalization_platform.pipeline.serve_ranked_feed --config configs/local_api.yaml
PYTHONPATH=src python -m personalization_platform.pipeline.build_portfolio_report --config configs/portfolio_report_smoke.yaml
```

## Current Repo Status

What exists today:

- request-level event-log build from fixture-backed feed-interaction inputs
- multi-source candidate generation using `affinity`, `content`, and `trending`
- baseline ranking dataset, logistic and tree-based rankers, and segmented comparison bundle
- explicit reranking rules for freshness, topic diversity, and creator spread
- deterministic experiment assignment plus offline readout with guardrails and SRM
- monitoring bundle for funnel health, score stability, and experiment integrity
- local FastAPI demo surface for replaying ranked fixture requests and scoring contextual candidate payloads
- portfolio-facing reporting bundle with a system summary and architecture note
- targeted unit tests for retrieval merge logic, segmented ranking diagnostics, reranking helpers, and experiment assignment
- one-command repo smoke validation via `bash scripts/ci_smoke.sh`
- one-command medium retrieval and ranking validation via `bash scripts/ci_medium.sh`
- GitHub Actions CI that runs the same smoke-quality command on pushes and pull requests

## Completed Extensions

Beyond the first end-to-end baseline, the repo now also includes:

- a medium validation path so retrieval and ranking checks are not limited to the tiniest smoke fixture
- richer item and creator metadata carried into `item_state` and reused by retrieval, reranking, and reporting
- a third retrieval source (`content`) on top of `affinity` and `trending`
- multiple ranker families, with logistic and tree-based baselines compared on the same validation split
- segmented ranking diagnostics by candidate source, provenance, cold-start status, and history depth
- stronger experiment readout with multiple outcomes, richer guardrails, and treatment slices
- targeted stage-level tests plus GitHub Actions CI
- a local API that supports fixture replay, contextual candidate-payload scoring, and request-time candidate assembly from local retrieval sources
- an explicit training-vs-serving feature contract so request-time inputs and offline-only ranker features stay clearly separated
- local serving interaction logs for request, exposure, response, and simulated click events
- lightweight request-time feature hydration from local state and prior serving logs, with explicit fallback behavior
- explicit request-time assembly controls for source budgets, trending-only fallback, and degraded-mode observability
- deterministic request-time experiment assignment with treatment metadata persisted in serving responses and logs
- live-style experiment readout over serving logs, including SRM, concentration, fallback rate, and degraded request rate
- a deterministic simulated-live validation tier that replays balanced treatment traffic from reranked outputs with fixed seeds
- lifecycle evaluation over both local serving smoke logs and stronger simulated-live serving logs

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

For portfolio or resume framing, the honest description is:

- reproducible offline personalization and experimentation platform
- request-level event-log build over MIND
- multi-source retrieval, baseline ranking, reranking constraints, experiment readout, monitoring, and local demo API

The repo should not be framed as a live production recommender system or as evidence of online lift on real traffic.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

For editable local development:

```bash
pip install -r requirements-dev.txt
pip install -e .
```

Commands use the `src/` layout directly:

```bash
PYTHONPATH=src python -m ...
```

The repo now pins the Python package versions used by the local validation workflows in:

- `requirements.txt`
- `requirements-dev.txt`
- `pyproject.toml`

That keeps `pip install -e .`, `pip install -r requirements.txt`, and the packaged runtime aligned on the same dependency set.

## Packaged Runtime

For a reproducible containerized runtime, the repo includes:

- `Dockerfile`
- `.dockerignore`
- `scripts/docker_smoke.sh`
- `scripts/docker_medium.sh`

Build and run the packaged smoke validation with:

```bash
bash scripts/docker_smoke.sh
```

Build and run the packaged medium validation with:

```bash
bash scripts/docker_medium.sh
```

If you prefer the raw Docker commands, they are:

```bash
docker build -t personalization-platform:local .
docker run --rm personalization-platform:local bash scripts/ci_smoke.sh
docker run --rm personalization-platform:local bash scripts/ci_medium.sh
```

The container image is intentionally thin and includes the fixture data plus the repo-local validation commands, but it does not include large raw datasets or previously generated artifacts.

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
- local serving smoke
- live-style experiment readout from serving logs
- lifecycle promotion and rollback evaluation

The default serving smoke requests intentionally span multiple users and both experiment treatments so the live readout and lifecycle bundle exercise a less toy-like control-versus-candidate comparison.

For a stronger synthetic-live validation tier, the repo also includes a deterministic replay simulation that generates larger serving-log bundles directly from `reranked_rows.csv` with fixed seeds and no dependency on prior `local_api_smoke` artifacts.

The reranking policy now also supports a prediction guard margin so freshness and diversity logic can promote plausible alternatives without letting very weak candidates leapfrog obvious high-score winners.

If you changed retrieval, ranking, or evaluation logic and want a richer offline check, run:

```bash
bash scripts/ci_medium.sh
```

This medium validation script focuses on the ranking stack:

- medium event-log config validation
- medium event-log build
- medium candidate generation
- medium ranking dataset build
- medium logistic baseline training
- medium multi-model comparison diagnostics

Use `ci_smoke.sh` as the fastest repo health check and `ci_medium.sh` as the default follow-up after substantive retrieval or ranking changes.

If you changed reranking, serving behavior, live-readout logic, or lifecycle decision thresholds and want the strongest local validation tier, run:

```bash
bash scripts/ci_simulated_live.sh
```

This simulated-live validation script focuses on treatment comparison and decision quality:

- compile and test the repo
- rebuild the smoke ranking and reranking path
- generate deterministic serving logs directly from reranked outputs
- replay balanced treatment traffic with fixed seeds
- run the same live experiment readout used by the serving smoke flow
- evaluate lifecycle promotion, hold, and rollback guidance on the larger simulated sample

The validation tiers are intentionally different:

- `ci_smoke.sh` checks the whole repo quickly, including local API smoke and reporting
- `ci_medium.sh` strengthens offline retrieval and ranking evaluation
- `ci_simulated_live.sh` strengthens treatment comparison and lifecycle evidence with a larger deterministic replay

For a smaller demo flow after the smoke run succeeds:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.serve_ranked_feed --config configs/local_api.yaml
PYTHONPATH=src python -m personalization_platform.pipeline.build_portfolio_report --config configs/portfolio_report_smoke.yaml
```

The first command validates replay mode, contextual scoring mode, and request-time candidate assembly for the local ranked-feed API. The second packages the latest smoke artifacts into a concise system summary and architecture note.

The local API smoke bundle now also writes a `serving_contract.json` artifact that records the accepted request schema, the current contextual feature surface, and the offline ranker features that are not yet supported online.

It also writes append-only local interaction logs:

- `request_events.csv`
- `exposure_events.csv`
- `response_events.csv`
- `click_events.csv`

These files are the first online-style logging surface for the repo. They are still local smoke artifacts, but they make the serving-to-experimentation boundary explicit and reusable.

When a prior local API smoke bundle exists, the serving layer now reuses those interaction logs to hydrate fresh request-time features such as recent topic click share, recent item CTR, and recent user click rate. If no prior serving log bundle exists, those features fall back to explicit zero-valued defaults rather than hidden approximations.

The request-time assembly path now also exposes simple operational controls:

- `max_sources_per_request`
- `fallback_to_trending_only`
- `trending_fallback_candidate_count`

When those controls trigger, the response and smoke summary surface `degraded_modes` so fallback behavior is observable rather than implicit.

The serving flow now also performs deterministic request-time experiment assignment using the same hashing strategy as the offline experiment layer. Local API responses, request logs, exposure logs, response logs, and click logs all persist the treatment actually shown to the user.

The repo now also includes a local live-style experiment readout command over those serving logs:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.analyze_live_experiment --config configs/live_experiment_smoke.yaml
```

That command emits a readout bundle from request-time logs with treatment metrics, SRM, fallback-rate guardrails, degraded-request rates, and sample-size caveats.

The human-readable version of that serving boundary lives in [src/personalization_platform/features/training_vs_serving_contract.md](/Users/hanlingjuan/personalization-experimentation-platform/src/personalization_platform/features/training_vs_serving_contract.md).

The experiment analysis bundle now includes:

- multiple treatment outcomes (`top1_ctr`, `top2_ctr`, and mean exposure label)
- guardrails for rank shift, prediction drift, top-2 repeat rates, and top-1 concentration
- treatment slices by dataset split, candidate source, cold-start status, and history-depth segment

For a richer offline validation slice than the tiny smoke path, the repo also includes a medium fixture path:

```bash
bash scripts/ci_medium.sh
```

This medium path keeps the same local, reproducible workflow but provides more requests and a larger time-ordered validation holdout for retrieval and ranking diagnostics.

If you want to run the stages individually, the script expands to:

```bash
PYTHONPATH=src python -m personalization_platform.pipeline.validate_event_log_config --config configs/mind_medium.yaml
PYTHONPATH=src python -m personalization_platform.pipeline.build_event_log --config configs/mind_medium.yaml
PYTHONPATH=src python -m personalization_platform.pipeline.build_candidates --config configs/candidates_medium.yaml
PYTHONPATH=src python -m personalization_platform.pipeline.build_ranking_dataset --config configs/ranking_dataset_medium.yaml
PYTHONPATH=src python -m personalization_platform.pipeline.train_ranker --config configs/ranker_medium.yaml
PYTHONPATH=src python -m personalization_platform.pipeline.compare_rankers --config configs/ranker_compare_medium.yaml
```

The ranker comparison bundles now include:

- aggregate classification and ranking metrics
- row-level slices by candidate source and multi-source provenance
- request-level slices by cold-start status and history-depth segment
- segment deltas versus retrieval-order fallback

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

For pipeline stages that emit `manifest.json`, the manifest is intended to be both human-readable and machine-parseable:

- use `run_metadata` to identify the current artifact bundle
- use `upstream_runs` to trace which earlier runs produced the inputs
- use `config.yaml` alongside `manifest.json` when comparing repeated runs of the same named workflow

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

This command reads the latest smoke event-log outputs, builds merged multi-source candidates from affinity, content, and trending retrieval, writes request-level candidates under `data/processed/candidates/<run_name>/`, and emits a run bundle with:

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

## Remaining Extensions

Most of the originally planned baseline and extension work is now implemented. If this repo continues further, the most meaningful remaining investments are:

- tighter environment reproducibility through stricter dependency pinning, lockfiles, or a containerized/devcontainer path
- larger and more varied evaluation slices beyond the current smoke and medium fixtures
- stronger model families or richer features if you want to push deeper into recommender-specific modeling work
- additional presentation polish such as screenshots, selected report excerpts, or a short demo walkthrough
