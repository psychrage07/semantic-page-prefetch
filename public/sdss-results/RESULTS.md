# Pagewise on a separate SDSS DR7 dataset

Completed local experiment, 27 October 2026. Original project and saved results are unchanged.

## What was run

The dataset is real astronomy data from [SDSS Data Release 7](https://classic.sdss.org/dr7/), used in [SeLeP, PVLDB 2024](https://www.vldb.org/pvldb/vol17/p2064-zirak.pdf). SeLeP appears in both project reviews: Review 2 reference 3 and Review 3 reference 1.

We downloaded 50,000 matched photometry/spectrum records, imported 49,999 distinct PhotoObj rows and 50,000 SpecObj rows, and built a separate 15,790,080-byte SQLite database (3,855 pages). These are actual survey measurements. This is the first 50,000 matching spectra ordered by ID, not a random or representative sample.

The SQL workload is newly generated from six templates adapted from the [official SDSS sample queries](https://skyserver.sdss.org/dr7/en/help/docs/realquery.asp). It is not SeLeP's original workload. Pagewise traces actual SQLite reads with APSW. SeLeP used a different PostgreSQL database, model, layout, task, and metrics; its published hit ratios cannot be compared directly with these next-page accuracies.

The same selected neural-hybrid architecture was trained from new weights on this dataset. No original checkpoint was overwritten or reused. New models, traces, data, and outputs live entirely in this sibling experiment directory.

## Held-out results

Training: 600 queries / 25,445 observed transitions. Validation: 200 queries / 8,088 observed transitions. Test: 300 queries / 12,639 observed transitions, across five workload seeds. Training and validation sample at most 16 transitions per query; the table below scores all 12,639 test transitions without sampling.

| Predictor | Query-average Top-1 | Transition Top-1 | Transition Top-5 |
|---|---:|---:|---:|
| Our neural hybrid (semantics + Markov) | 35.32% | 27.67% | 38.52% |
| Markov-2 | 27.32% | 25.74% | 37.34% |
| Markov-1 | 18.38% | 14.16% | 31.42% |
| Hybrid without semantics | 31.26% | 26.01% | 39.53% |
| Semantic Model B | 14.78% | 5.89% | 15.38% |
| History-only Model A | 8.15% | 2.52% | 6.85% |
| Sequential | 4.77% | 0.65% | 0.65% |

Our hybrid minus Markov-2 in query-average Top-1: **+7.99 percentage points**. The descriptive hierarchical bootstrap 95% interval is [+6.90, +9.06] points (5,000 resamples of workload seeds and queries). This is a descriptive comparison for this bounded dataset and workload, not a claim of general superiority.

Hybrid candidate recall: **55.99%**. Targets absent from the candidate pool remain incorrect. Macro gives every query equal weight; micro gives every transition equal weight. Report both because long join queries otherwise dominate.

## Training details

Fixed selected architecture: B2 features, hidden width 64, listwise loss, Markov log-prior weight 1, AdamW learning rate 0.001, up to 16 epochs, training seed 101. Candidate generation: existing layout-aware generator, maximum 128 candidates. Checkpoint epoch is chosen on validation macro Top-1 only. All model configurations were fixed before test scoring.

| Model | Validation-selected epoch | Validation macro Top-1 |
|---|---:|---:|
| Our neural hybrid (semantics + Markov) | 16 | 35.62% |
| Hybrid without semantics | 4 | 31.85% |
| Semantic Model B | 16 | 14.73% |
| History-only Model A | 2 | 8.41% |

All four selected checkpoints have a learned neural component (selected epoch above zero). The no-semantics control disables semantic features while retaining history and Markov features. Candidate proposal and transition statistics use training executions only. Full static B-tree metadata is allowed from the read-only database and is not filtered by future test membership.

The exact experiment script and all checkpoint hashes were sealed before scoring the held-out test. See `models/evaluation-seal.json`, `models/freeze.json`, and `scripts/frozen-experiment.py`.

## Results by query template

| Query template | Hybrid macro Top-1 | Markov-2 macro Top-1 |
|---|---:|---:|
| sdss_field_summary | 29.40% | 23.48% |
| sdss_magnitudes | 16.39% | 16.41% |
| sdss_object | 57.00% | 32.00% |
| sdss_plate_join | 59.43% | 51.63% |
| sdss_redshift_join | 24.82% | 24.77% |
| sdss_sky_window | 24.85% | 15.65% |

## Actual runtime smoke check

Twenty additional paired queries per policy ran through the existing APSW tracing and asynchronous prefetch runtime, using the newly fitted checkpoint. All workers shut down cleanly with zero prefetch read errors. The check compares result row counts and first-row previews across policies; it is not a full result-stream checksum check.

| Policy | Mean foreground query ms | Median query ms | Prefetched pages | Useful prefetches | Scored jobs | Read errors |
|---|---:|---:|---:|---:|---:|---:|
| none | 1.158 | 0.803 | 0 | 0 | n/a | 0 |
| markov2 | 1.796 | 1.232 | 33 | 1 | n/a | 0 |
| Our neural hybrid | 2.171 | 1.533 | 2 | 0 | n/a | 0 |

Policy order was randomized once with seed 501 and recorded. SQLite pager cache was 128 KiB; the Pagewise cache was 512 pages, with one worker, degree 1, confidence threshold 0.7, prediction on every fourth miss, stale-job dropping, and no artificial delay. The OS cache was not reset. This small smoke check establishes operational completion and preserves timing diagnostics; it does not establish a statistically supported speedup.

## Checks and limits

- Actual database integrity and foreign-key checks passed; exported IDs retain exact integer precision.
- SQL template/parameter combinations do not overlap between training, validation, and test. Templates and some data pages do overlap. This is not an unseen-template or unseen-object test.
- All predictors were scored on identical held-out transitions; every eligible test transition is included.
- Accuracy aggregates were independently recomputed from per-query hit counts.
- Candidate index and checkpoint transition counts were rebuilt from training only and matched saved artifacts.
- Offline and runtime predictor rankings matched on recorded samples.
- Source snapshot and frozen checkpoint hashes match; separate SQLite database full SHA256 remains unchanged.
- 124 protected original paths passed before/after checks. Tracked files and original checkpoint are hashed fully; the original 8.62 GB database is checked by size, modification time, and header hash, not a full-file hash.
- This is one bounded ordered SDSS subset, one SQLite layout, one host, and generated SQL parameters. It is a separate external-data experiment, not a replacement for the existing research study.

## Where to inspect outputs

- `data/provenance.json`: exact service requests, SQL, raw batch hashes, and dataset origin.
- `data/sdss-dr7.sqlite`: the real, separately imported database.
- `models/hybrid_semantics.pt`: the newly fitted neural hybrid.
- `models/freeze.json`: training histories, validation selection, configurations, and checkpoint hashes.
- `results/accuracy.csv` / `accuracy.json`: full comparison, including controls.
- `results/per-query.csv`: every query/policy's hit counts.
- `results/prediction-samples.jsonl`: actual next-page predictions and targets.
- `traces/`: actual VFS reads and SQL execution records.
- `results/runtime/`: real runtime traces, predictions, timings, and counters.
- `results/verification.json` and `results/independent-audit.json`: verification evidence.

See `README.md` for reproduction commands and [SDSS acknowledgements](https://classic.sdss.org/dr7/credits.html) for data attribution.
