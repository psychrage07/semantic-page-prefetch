# Pagewise: A Semantics-Aware Neural Prefetching Framework for SQLite

## 1. Project Overview & Research Question

**Title**: Pagewise — Semantics-Aware Neural Page Prefetching for Irregular Memory Access Patterns in SQLite  
**Repository**: `sqlite-semantic-page-prefetching-updated-v2`  
**Location**: `d:\download\sqlite-semantic-page-prefetching-updated-v2`

### Central Research Question
> **Can application-level semantic information about SQLite database pages (table/index identity, page type, B-tree depth, and query-plan context) improve learned next-page prefetch predictions over history-only predictors, and does any prediction accuracy gain translate into real wall-clock latency improvements under a controlled cache budget?**

---

## 2. Executive Summary of Research Findings

1. **Offline Accuracy (Model B vs Model A)**:
   - On held-out query executions, semantic features significantly improve neural next-page prediction:
     - **Model B (Semantic MLP)**: ~28.1% top-1 accuracy, ~60.0% top-5 accuracy.
     - **Model A (History-only MLP)**: ~17.2% top-1 accuracy, ~40.0% top-5 accuracy.
   - **Markov Baselines Dominance**: Simple statistical baselines outperform both MLPs in offline accuracy:
     - **Markov-2 (Second-order transitions)**: ~53.1% top-1 accuracy.
     - **Markov-1 (First-order transitions)**: ~48.6% top-1 accuracy.
   - **Semantic Ablations**: Multi-seed ablation sweeps demonstrate that **page metadata** (specifically table/index object identity and page type) accounts for the vast majority of Model B's offline gain. Query-plan context and position features provide marginal additional signal.

2. **Online Runtime Latency Under Controlled Evaluation**:
   - **Real Storage (No Injected Delay, 3.2 MB DB)**: In Windows environments, a 3.2 MB SQLite database fits entirely into the OS page cache. All prefetching strategies (heuristics and neural) add CPU scheduling and inference overhead that makes them slower than the custom cache `none` baseline (and custom cache `none` is slower than stock SQLite).
   - **Latency-Injected Mode (0.2 ms / miss)**:
     - **Markov-2** achieved the best paired latency improvement: **`-4.18 ms`** paired delta (85.7% of queries improved).
     - **Sequential prefetching** achieved **`-3.81 ms`** paired delta.
     - **Model A** achieved **`-1.37 ms`** paired delta.
     - **Model B** raw neural inference was too computationally expensive (60–100 ms total inference per run), negating prefetch gains.
     - **Hybrid Predictor** (Markov-2 for known contexts, Model B for unseen contexts) cut Model B inference overhead by over 50% and achieved statistically significant paired latency deltas.

3. **Benchmark Audit & Repairs (Latest Milestone)**:
   - **Queue Staleness**: Discovered that queuing delays caused **71%–74%** of worker prediction jobs to execute for queries that had already finished. Implemented active query synchronization and `--drop-stale-jobs`, cutting queue wait time by **52.7%** and reducing neural inference overhead by **76.6%**.
   - **Storage Channel Contention**: Independent delay injection (`time.sleep`) modeled infinite storage bandwidth. Implemented `SimulatedStorageDevice` with `--storage-model shared` to accurately serialize storage access and account for demand misses delayed by speculative I/O.
   - **Lock Boundaries**: Formally verified that artificial delay injection is strictly executed outside cache locks (`BoundedPageCache._lock`) and runtime locks.
   - **Timestamped Diagnostics**: Added full-lifecycle monotonic timestamps (`request_ns`, `worker_start_ns`, `inference_start_ns`, `inference_done_ns`, `page_read_start_ns`, `page_read_done_ns`, `demand_use_ns`, `lead_time_ms`) logged to `prefetch_uses.jsonl` and `predictions.jsonl`.
   - **Consistent Denominators**: Standardized prefetch efficiency metrics to share the consistent denominator $N_{PF} = \text{pages\_prefetched}$, ensuring $\text{Useful Prefetch \%} + \text{Total Wasted Prefetch \%} = 100.0\%$.

---

## 3. Repository Architecture & Directory Map

```
sqlite-semantic-page-prefetching-updated-v2/
├── PROJECT_CONTEXT.md               # Master context and handoff document
├── README.md                        # High-level repository introduction
├── package.json / tsconfig.json     # Frontend dependencies and TypeScript config
├── vite.config.ts                   # Vite bundler configuration
├── src/                             # Interactive React/TypeScript dashboard
│   ├── components/                  # Dashboard cards, charts, and metrics tables
│   ├── types.ts                     # TypeScript data interfaces
│   └── App.tsx                      # Dashboard root application
├── public/
│   └── pagewise-dashboard.json      # Exported measured results snapshot for frontend
├── backend/
│   ├── pyproject.toml               # Python package configuration (Python 3.11/3.12)
│   ├── requirements.txt             # PyTorch, APSW, NumPy, pytest dependencies
│   ├── FINAL_EVALUATION_PROTOCOL.md # Predeclared evaluation specification
│   ├── IMPLEMENTATION_STATUS.md     # Honest verification and progress record
│   ├── pagewise/                    # Core Python package
│   │   ├── cli.py                   # Unified CLI entry point
│   │   ├── database.py              # Synthetic e-commerce database generator
│   │   ├── workloads.py             # 7 SQL workload templates (point, range, join, agg, etc.)
│   │   ├── metadata.py              # SQLite B-tree & page map extractor (APSW/dbstat)
│   │   ├── tracing.py               # APSW VFS interception layer (TraceVFS, TraceRecorder)
│   │   ├── dataset.py               # Feature vectorization and sequence building
│   │   ├── models.py                # MLPScorer, CandidateIndex, Model A, Model B, Hybrid
│   │   ├── evaluate.py              # Offline ranker evaluation metrics (Top-k, MRR)
│   │   ├── runtime.py               # BoundedPageCache, SimulatedStorageDevice, PredictiveRuntime
│   │   ├── runner.py                # Workload execution harness with VFS instrumentation
│   │   ├── suite.py                 # Repeated multi-seed experiment runner & recovery
│   │   ├── sweep.py                 # Multi-seed training sweeps & semantic ablations
│   │   ├── reporting.py             # Consolidated Markdown reports & dashboard export
│   │   └── dashboard.py             # Optional Streamlit viewer for measured artifacts
│   ├── tests/                       # Automated test suite (32 passed tests)
│   │   ├── test_benchmark_audit_and_repair.py # Audit diagnostics & storage device tests
│   │   ├── test_database_and_workloads.py     # Schema and query execution tests
│   │   ├── test_dataset_and_model.py          # Feature extraction & MLP training tests
│   │   ├── test_metadata_fallback.py          # APSW dbstat metadata tests
│   │   ├── test_reporting.py                  # Markdown reporting & dashboard export tests
│   │   ├── test_runtime_cache.py              # LRU cache, worker queue, & hybrid tests
│   │   ├── test_suite_summary.py              # Suite aggregation & delta bootstrapping tests
│   │   └── test_vfs_integration.py            # Real APSW VFS interception tests
│   ├── scripts/                     # PowerShell automated execution scripts
│   │   ├── run_smoke.ps1            # Quick end-to-end smoke pipeline
│   │   └── run_medium_experiment.ps1# 800-page medium-scale experiment pipeline
│   └── runs/                        # Experimental artifacts and results
│       ├── experiment-latest/       # Frozen primary experimental dataset
│       │   ├── pagewise-medium.sqlite       # Benchmark database snapshot (3.2 MB, 800 pages)
│       │   ├── models/                      # Candidate index, Model A, Model B checkpoints
│       │   ├── research_report.md           # Consolidated research report
│       │   ├── final-latency-0p2ms/         # Final Protocol Category A (0.2ms delay)
│       │   ├── final-latency-1p0ms/         # Final Protocol Category A (1.0ms delay)
│       │   └── final-real-storage/          # Final Protocol Category B (real storage)
│       └── diagnostic-audit-001/    # Diagnostic benchmark audit comparison runs
```

---

## 4. Complete CLI Command Reference

All backend commands are run via `python -m pagewise.cli <command>` from the `backend/` directory:

| Command | Key Arguments | Purpose |
| --- | --- | --- |
| `check-env` | — | Verify Python, SQLite, APSW, PyTorch, and dbstat support |
| `generate-db` | `--db PATH --customers N --products N --orders N` | Generate reproducible synthetic e-commerce SQLite DB |
| `trace` | `--db PATH --out DIR --queries N --seed S` | Trace VFS `xRead` requests using APSW |
| `build-sequences` | `--trace FILE --page-map FILE --out FILE` | Convert raw VFS read trace to per-query page sequences |
| `train` | `--sequences FILE --page-map FILE --out DIR` | Train matched Model A (history) and Model B (semantic) |
| `evaluate` | `--sequences FILE --candidate-index FILE --model-a FILE --model-b FILE` | Evaluate offline prediction accuracy across predictors |
| `run-prefetch` | `--db PATH --model-dir DIR --strategy [model_b\|hybrid\|markov2] --out DIR` | Run workload with bounded custom cache and async prefetcher |
| `run-suite` | `--db PATH --model-dir DIR --out DIR --seeds S1,S2,S3 --strategies STRATS` | Run repeated multi-seed benchmark across all strategies |
| `recover-suite` | `--out DIR` | Rebuild suite manifest and summary after interrupted run |
| `report` | `--input FILE.json --out FILE.md` | Generate formatted Markdown table report from JSON summary |
| `research-report` | `--run-dir DIR --out FILE.md` | Generate consolidated research report with findings |
| `dashboard-export` | `--run-dir DIR --out public/pagewise-dashboard.json` | Export compact JSON snapshot for React frontend |

---

## 5. Quickstart & Verification Guide

### Prerequisites
- Python 3.11 or 3.12 (with `apsw`, `torch`, `pytest`)
- Node.js 18+ (for frontend dashboard)

### 1. Run Backend Automated Test Suite
From `backend/`:
```powershell
python -m pytest tests
```
*Expected result: 32 passed in ~6.5 seconds.*

### 2. Run a Quick Benchmark Audit Diagnostic
```powershell
python -m pagewise.cli run-suite `
  --db runs/experiment-latest/pagewise-medium.sqlite `
  --model-dir runs/experiment-latest/models `
  --out runs/diagnostic-audit-001/quick-test `
  --queries 50 `
  --seeds 101 `
  --strategies none,markov2,hybrid `
  --cache-pages 256 `
  --degree 2 `
  --miss-delay-ms 0.2 `
  --storage-model independent `
  --drop-stale-jobs
```

### 3. Launch Frontend React Dashboard
From repository root:
```powershell
npm install
npm run dev
```
Open `http://localhost:5173` in your browser. The dashboard automatically loads measured results from `public/pagewise-dashboard.json`.
