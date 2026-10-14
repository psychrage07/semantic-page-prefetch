# Implementation status (honest verification record)

This file records what was and was not verified in the build environment used to prepare this repository.

## Verified here

- Python modules compile.
- CLI help and environment check run.
- The SQLite database generator creates a small deterministic database.
- The available stdlib SQLite build does **not** support `dbstat` in this environment, but APSW does; the page metadata builder successfully uses the APSW fallback.
- The generated SQL workload templates execute against a small database.
- Model A and Model B can be trained and evaluated on fixture sequences in automated tests.
- The bounded LRU page-cache accounting unit tests pass.
- Suite aggregation, paired per-query comparisons, runtime queue accounting, and the semantic query-position feature have tests.
- APSW is available here (`APSW 3.54.0.0`, SQLite `3.54.0`), and the APSW VFS integration test passes.
- Current test result: `27 passed`.
- A development smoke pipeline completed end to end with real VFS tracing, query-to-trace association, sequence building, matched Model A/Model B training, and held-out prediction evaluation.
- A quick latency-injected suite completed for `stock`, `none`, and `model_b`, producing run-level and paired per-query comparison summaries.
- A fuller latency-injected development suite completed for all strategies. After pruning cached/pending pages before neural scoring, mean neural inference cost dropped substantially in the development suite and paired per-query deltas versus the custom-cache `none` baseline became negative for both neural models. Treat this as a scale-limited engineering signal, not a final research claim.
- Runtime scheduling is now configurable with `predict_on`, `predict_every`, and `queue_size`. A miss-only scheduling suite reduced dropped neural jobs but did not improve the development-scale neural latency deltas versus the prior all-observation scheduler with cached-candidate pruning.
- Suite summaries now include scheduler-pressure counters such as prediction requests, queued/completed/dropped jobs, skipped observations, coalesced jobs, and cached/pending candidate pruning.
- A `runtime-sweep` command now runs a grid of cache/scheduler settings, stores normal suite artifacts for each combination, and writes a ranked sensitivity report. The first quick sweep suggests small-cache configurations can show stronger prefetch effects, while larger custom-cache baselines can make neural prefetching neutral or negative at smoke scale.
- Model B now supports inference-time and retraining-based semantic ablations for `page_type`, `depth`, `object`, `plan`, `template`, `workload`, `position`, plus `page_metadata` and `query_context` aliases. The first smoke ablation evaluations suggest page metadata/object identity carries most of Model B's offline advantage on the held-out smoke split. Retrained no-query-context and no-position variants did not degrade in this small run, so broader seeds are needed before drawing conclusions.
- `train-sweep` now supports retrained semantic ablations across model seeds and reports `delta_vs_model_b`. A three-seed smoke sweep preserved the page-metadata signal: retrained `no_page_metadata` averaged `-6.8783` top-1 points and `-22.2227` top-5 points versus full Model B on the fixed smoke split. Retrained `no_query_context` was roughly neutral on top-1 and negative on top-5, so it needs larger workload validation.
- A tuned small-cache runtime suite compared all strategies under `cache_pages=32`, `degree=2`, `predict_on=miss`. In this smoke-scale latency-injected run, Model A had the best paired delta versus `none` (`-1.099838 ms` mean), ahead of Markov-1 (`-0.513829 ms`) and Markov-2 (`-0.378009 ms`). Model B was only modestly better than `none` (`-0.229559 ms`) and had higher inference cost.
- A `report` command now writes concise Markdown tables from suite summaries, training sweeps, and runtime sensitivity reports.
- `run-prefetch` now accepts `--model-path` so explicit neural checkpoints, including retrained ablation checkpoints, can be evaluated in the runtime. Runtime stats record the checkpoint path and disabled semantic groups. A smoke run with `model_b_retrained_no_page_metadata.pt` under the tuned small-cache setting completed successfully; for seed 101 it improved over full Model B's same-seed runtime despite weaker offline accuracy.
- **NEW**: `max_score_candidates` parameter added to PredictiveRuntime. When set >0, candidates are pre-ranked by Markov successor order and only the top N are scored neurally. This reduces per-job inference cost by limiting forward passes. Runtime stats now track `candidates_pre_ranked_out`, `inference_ms_max`, and `inference_ms_per_job`.
- **NEW**: A medium-scale experiment (2000 customers, 500 products, 15000 orders, 800-page database, 300 queries traced, 12 epochs) confirmed that semantic features improve offline prediction (Model B 28.1% top-1 vs Model A 17.2%) but that Markov baselines dominate offline accuracy (Markov-2 53.1%, Markov-1 48.6%). At runtime with 0.2ms injected delay, all prefetch strategies were slower than `none` due to inference overhead. The `max_score_candidates=12` optimization reduced Model A inference by 34% (693→458ms) and Model B by 22% (899→703ms), bringing Model A to break-even vs `none`.
- **NEW**: `research-report` CLI command generates a consolidated Markdown report combining predictor metrics, suite results, ablation tables, and optimized-suite comparisons from an experiment directory.
- **NEW**: `build_consolidated_report()` in `reporting.py` auto-discovers `evaluation/predictor_metrics.json`, `suite-*/suite_summary.json`, and `suite-model-b-optimized/suite_summary.json` under a run directory.
- **NEW**: `dashboard-export` CLI command writes a compact `pagewise-dashboard.json` snapshot for the React/Vite UI. The root dashboard loads measured predictor/runtime summaries, per-query latency series, and capped trace/prediction samples while clearly labelling the remaining fixture tools.
- **NEW**: Runtime shutdown now waits for accepted jobs and records drain, shutdown, and end-to-end wall-clock timing. A `hybrid` policy uses a training-derived Markov-2 transition when present and semantic Model B only for unseen second-order contexts. In the two-seed, 60-query-per-seed latency-injected development suite, hybrid reduced mean inference from Model B's 236.545 ms to 92.795 ms and had a paired mean delta of `-1.648996 ms` versus `none`; this is development evidence only.
- **NEW**: A recovered three-seed validation suite (200 matched queries per seed, 0.2 ms injected miss delay) completed all 12 policy runs with clean worker shutdown. Hybrid used 1644 Markov jobs and 570.333 semantic jobs per run on average, reduced Model B inference from 979.928 ms to 403.089 ms, and improved paired mean query latency by `-3.186441 ms` versus `none` across 600 matched queries. Markov-2 remained slightly faster at `-3.421255 ms`; both results are latency-injected evidence only.
- **NEW**: Completed the full predeclared final evaluation protocol (`FINAL_EVALUATION_PROTOCOL.md`). Three categories, each with seeds 901/902/903 and 100 queries per seed per strategy, all 8 strategies (stock, none, sequential, markov1, markov2, model_a, model_b, hybrid), predeclared cache/scheduler settings:
  - **Category A (0.2 ms delay):** Markov-2 best at `-4.1821 ms` paired delta (85.7% improved, CI excludes zero). Sequential also significant at `-3.8113 ms`. Model A significant at `-1.3709 ms`. Model B and hybrid CIs cross zero.
  - **Category A (1.0 ms delay):** No strategy significantly beat `none`; Markov-2 closest at `+0.5160 ms` (CI crosses zero). Higher delay amplified both prefetch benefit and scheduling overhead at this cache/queue config.
  - **Category B (real-storage, 0.0 ms):** All prefetching strategies slower than `none` (all CIs positive). The custom bounded-cache `none` is already slower than `stock` SQLite; the 3.2 MB database fits comfortably in the Windows OS page cache so VFS reads are sub-millisecond.
- **NEW**: `build_consolidated_report()` now auto-discovers `final-*` suite directories, separates them from `suite-*` development suites, and generates data-driven Key Findings (cross-condition comparison table, significance markers, conclusion) instead of a placeholder.
- **NEW**: `build_dashboard_snapshot()` now includes `final-*` suites alongside `suite-*` suites.
- **NEW (Benchmark Audit & Repair Milestone)**:
  - Audited and repaired queue staleness: discovered that queue delays caused 71–74% of queued prediction jobs to execute for queries that had already completed. Added active query synchronization (`set_active_query`) and configurable `--drop-stale-jobs`, cutting queue wait time by 52.7% and reducing neural inference overhead by 76.6%.
  - Investigated storage channel contention: independent delays simulated infinite parallel storage bandwidth. Implemented `SimulatedStorageDevice` with `--storage-model shared` (versus default `independent`), serializing device access and accurately accounting for demand miss delay caused by speculative I/O.
  - Verified lock boundaries: verified that artificial delay injection is strictly executed outside cache locks (`BoundedPageCache._lock`) and runtime locks in both VFS demand reads and speculative prefetch reads.
  - Timestamped end-to-end diagnostics: added monotonic timestamps tracing prediction request, worker dequeue, neural scoring, prefetch read, and subsequent demand hit with lead time calculation. Traces are emitted to `prefetch_uses.jsonl`, `predictions.jsonl`, and `runtime_stats.json`.
  - Consistent prefetch rate denominators: standardized useful and wasted prefetch rates to share the identical denominator $N_{PF} = \text{pages\_prefetched}$, ensuring $\text{Useful Prefetch \%} + \text{Total Wasted Prefetch \%} = 100.0\%$.
  - Added regression test suite [test_benchmark_audit_and_repair.py](file:///d:/download/sqlite-semantic-page-prefetching-updated-v2/backend/tests/test_benchmark_audit_and_repair.py). Total test suite expanded to 32 passed tests.
  - Diagnostic experiments executed in `backend/runs/diagnostic-audit-001/`, preserving all frozen final evaluation results in `backend/runs/experiment-latest/`.

## Not verified here

- The completed smoke, medium-scale, and quick-suite runs are development-scale validation artifacts, not final research evidence.
- The final evaluation protocol has been completed (Category A and B). However, Category B (real-storage) does not reset OS page cache state and must not be described as measured SSD I/O. The 3.2 MB database is small enough to fit entirely in RAM cache.
- The root React/Vite dashboard loads measured output through `public/pagewise-dashboard.json` for overview/comparison metrics, query latency curves, and trace samples. Replay controls, the break-even calculator, and the ablation preview remain labelled illustrative values. The optional Streamlit viewer can display actual generated JSON/JSONL artifacts after the backend runs.
- A VFS `xRead` is not proof of physical storage I/O. Real-storage performance requires careful cache control and separate Windows I/O instrumentation; injected-latency results must remain labelled as emulated.

## Latest validation artifacts

- Smoke run: `backend/runs/smoke-20261013-221152`
- Medium-scale experiment: `backend/runs/experiment-latest`
- Medium-scale predictor metrics: `backend/runs/experiment-latest/evaluation/predictor_metrics.json`
- Medium-scale tuned suite: `backend/runs/experiment-latest/suite-tuned-all/suite_summary.json`
- Medium-scale optimized suite: `backend/runs/experiment-latest/suite-model-b-optimized/suite_summary.json`
- Hybrid validation suite: `backend/runs/experiment-latest/suite-hybrid-validation/suite_summary.json`
- **Final protocol Category A (0.2ms):** `backend/runs/experiment-latest/final-latency-0p2ms/suite_summary.json`
- **Final protocol Category A (1.0ms):** `backend/runs/experiment-latest/final-latency-1p0ms/suite_summary.json`
- **Final protocol Category B (real-storage):** `backend/runs/experiment-latest/final-real-storage/suite_summary.json`
- Consolidated research report (with Key Findings): `backend/runs/experiment-latest/research_report.md`
- React dashboard snapshot: `public/pagewise-dashboard.json`
- Smoke predictor metrics: `backend/runs/smoke-20261013-221152/evaluation/predictor_metrics.json`
- Quick paired suite: `backend/runs/smoke-20261013-221152/suite-quick-paired`
- Quick paired suite summary: `backend/runs/smoke-20261013-221152/suite-quick-paired/suite_summary.json`
- Full development suite before runtime pruning: `backend/runs/smoke-20261013-221152/suite-all-latency-0p2ms`
- Full development suite after runtime pruning: `backend/runs/smoke-20261013-221152/suite-all-latency-0p2ms-runtime-prune`
- Resummarized full development suite after runtime pruning: `backend/runs/smoke-20261013-221152/suite-all-latency-0p2ms-runtime-prune-resummary`
- Full development suite with miss-only scheduling: `backend/runs/smoke-20261013-221152/suite-all-latency-0p2ms-predict-miss`
- Resummarized miss-only scheduling suite: `backend/runs/smoke-20261013-221152/suite-all-latency-0p2ms-predict-miss-resummary`
- Single-run queue-size probe: `backend/runs/smoke-20261013-221152/runtime-model-b-queue256`
- Runtime sensitivity quick sweep: `backend/runs/smoke-20261013-221152/runtime-sweep-quick`
- Runtime sensitivity ranking: `backend/runs/smoke-20261013-221152/runtime-sweep-quick/runtime_sensitivity_ranking.csv`
- Semantic ablation evaluation: `backend/runs/smoke-20261013-221152/evaluation-ablations`
- Retrained semantic ablation models: `backend/runs/smoke-20261013-221152/models-retrained-ablations`
- Retrained semantic ablation evaluation: `backend/runs/smoke-20261013-221152/evaluation-retrained-ablations`
- Multi-seed ablation sweep: `backend/runs/smoke-20261013-221152/ablation-sweep-seeds`
- Multi-seed ablation sweep report: `backend/runs/smoke-20261013-221152/ablation-sweep-seeds/training_sweep_report.json`
- Tuned small-cache Markov comparison suite: `backend/runs/smoke-20261013-221152/suite-small-cache-tuned-vs-markov`
- Tuned small-cache report: `backend/runs/smoke-20261013-221152/suite-small-cache-tuned-vs-markov/suite_report.md`
- Ablation sweep Markdown report: `backend/runs/smoke-20261013-221152/ablation-sweep-seeds/training_sweep_report.md`
- Runtime sensitivity Markdown report: `backend/runs/smoke-20261013-221152/runtime-sweep-quick/runtime_sensitivity_report.md`
- Explicit retrained-ablation runtime run: `backend/runs/smoke-20261013-221152/runtime-model-b-retrained-no-page-metadata`

## Required validation before final research claims

1. Create a virtual environment in `backend` using `python -m venv .venv` (this does not require the Windows `py` launcher). Python 3.11 is also supported by the package metadata.
2. Install `-e ".[test]"` and confirm `pagewise check-env` reports APSW available.
3. Run `pytest -q` and verify the APSW integration test is **passed**, not skipped.
4. Run `.\scripts\run_smoke.ps1 -Queries 150` from PowerShell in `backend`.
5. Review the generated `trace.jsonl`, query results, predictor metrics, and any errors before running multi-strategy performance experiments.
6. Run a full `run-suite` with all strategies and multiple seeds, then inspect both run-level summaries and `paired_query_comparisons`.

