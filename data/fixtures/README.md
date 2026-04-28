# Fixtures

This directory stores small local fixtures used for smoke validation.

Planned event-log fixture path:

- `data/fixtures/mind_smoke/`

The future smoke fixture should include:

- `behaviors.tsv`
- `news.tsv`

The intent is to keep the fixture raw-like enough that the same event-log conversion code can read it without a special-case parsing path.

Current smoke fixture path:

- `data/fixtures/mind_smoke/behaviors.tsv`
- `data/fixtures/mind_smoke/news.tsv`

Current medium validation fixture path:

- `data/fixtures/mind_medium/behaviors.tsv`
- `data/fixtures/mind_medium/news.tsv`

Smoke fixture design:

- raw-like MIND inputs rather than already-converted event-log tables
- 2 users across 6 requests
- 4 inferred sessions using visible timestamp gaps
- 3 candidate impressions per request
- a mix of clicked and non-clicked items
- 6 items spanning multiple topics to exercise diversity and cold-start joins

Medium fixture design:

- raw-like MIND inputs rather than already-converted event-log tables
- 4 users across 20 requests
- repeated topical preferences so retrieval and ranking have richer signal
- 4 candidate impressions per request
- 12 items spanning sports, finance, tech, health, entertainment, and world topics
- later requests that support a larger time-ordered validation holdout

Intended use:

- event-log smoke conversion
- retrieval development on stable tiny inputs
- ranking dataset and schema join smoke tests
- richer offline validation than the tiny smoke path while staying fully local and reproducible

Known limitations:

- timestamps are tiny and hand-authored rather than distribution-matched to full MIND
- URLs and entity fields are placeholders
- publisher must still be derived by downstream code because raw MIND-like rows do not include it here
- engagement is binary click-only
- impression depth, user count, and topic coverage are intentionally too small for metric interpretation
- the medium fixture improves validation depth, but it is still a local portfolio-scale slice rather than a production-like corpus
