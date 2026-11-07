# Idle neural hybrid: implementation and measured pilot

The idle-time approach is now implemented for Pagewise's semantic neural-Markov model as **P4**. Instead of predicting repeatedly during a query, it uses completed-query history to rank likely early misses of the next query. A bounded worker reads those pages during the idle interval. New query arrival cancels further speculative admission without waiting for inference to finish. Cached/pending pages are excluded before scoring; an eight-page unused-prefetch quota and hot-page protection limit speculative cache occupancy.

This is a newly trained query-level adaptation using the existing B2 semantic features and width-64 MLP, combined with a query-level Markov prior. It has its own checkpoints. The original selected hybrid, database, frozen core source and frontend research results remain unchanged. P4 has not been wired into the browser's demo selector in this phase.

## Results

The comparison condition was declared before evaluation: **50 ms idle gaps, 0.2 ms artificial shared-storage service, 32-page cache, familiar query values**, using three workload seeds (9001-9003). Large trials have eight queries per policy/seed; small trials have twelve. Mean latency averages the three run means. The p95 column averages the three run p95 estimates; it is not a pooled percentile. Useful counts are pooled over the three trials. A useful page is a prefetched insertion actually requested later.

### Original 8.62 GB database

| Configuration | Mean query time | Average run p95 | Useful / prefetched pages | Useful fraction |
|---|---:|---:|---:|---:|
| P0: same cache, no prefetch | 35.899 ms | 86.502 ms | 0 / 0 | n/a |
| P1: current Markov-2 | 49.278 ms | 117.468 ms | 19 / 497 | 3.82% |
| P2: original selected neural hybrid | 41.346 ms | 101.744 ms | 1 / 29 | 3.45% |
| P3: idle Markov, full page sets | 34.118 ms | 84.438 ms | 83 / 95 | 87.37% |
| P3 control: idle Markov, P4's early-miss targets | 34.395 ms | 84.393 ms | 73 / 84 | 86.90% |
| **P4: idle semantic neural hybrid** | **34.062 ms** | **83.608 ms** | **84 / 96** | **87.50%** |
| P4 control: retrained without semantics | 35.071 ms | 85.687 ms | 85 / 105 | 80.95% |

P4's mean is 17.62% lower than the original neural hybrid and 5.12% lower than P0 in this condition. Compared with full-page-set idle Markov, the mean differs by only 0.17%; these short pilots do not establish neural superiority. P4's mean is 2.88% below the no-semantics control here, but the small database shows the opposite ordering. No universal semantic benefit is established.

### Small development database

| Configuration | Mean query time | Average run p95 | Useful / prefetched pages | Useful fraction |
|---|---:|---:|---:|---:|
| P0: same cache, no prefetch | 6.865 ms | 13.827 ms | 0 / 0 | n/a |
| P1: current Markov-2 | 9.243 ms | 20.246 ms | 27 / 117 | 23.08% |
| P2: existing small neural hybrid | 9.337 ms | 19.299 ms | 3 / 10 | 30.00% |
| P3: idle Markov, full page sets | 5.827 ms | 16.530 ms | 100 / 116 | 86.21% |
| P3 control: idle Markov, P4's early-miss targets | 5.078 ms | 11.389 ms | 107 / 139 | 76.98% |
| **P4: idle semantic neural hybrid** | **5.128 ms** | **11.302 ms** | **101 / 128** | **78.91%** |
| P4 control: retrained without semantics | 4.852 ms | 11.156 ms | 113 / 147 | 76.87% |

P4's mean is 45.08% below the existing small hybrid and 25.30% below P0 in this condition. Matched-target idle Markov and the no-semantics neural control have lower means than semantic P4. Improvements therefore cannot be attributed solely to semantic features.

## Training and validation

Each database has its own 108 training queries and 48 validation queries. The validation parameter pool differs from training; both accepted protocols recorded zero identical SQL-and-parameter pairs across those pools. Repeated evaluation deliberately reuses familiar values. Additional conditions exercise query-level unfamiliar templates; some may be familiar to the original frozen within-query model.

Targets are the next training query's first 16 distinct demand misses. Only completed-query templates and observed page history are supplied online; next-query SQL, parameters, template and pages are unavailable. Candidate generation and transition counts use training records only. Neural logits rerank at most 64 candidates with a log empirical Markov prior; the 0.25 admission threshold is an empirical training frequency, not a calibrated neural confidence.

| Variant | Epochs run / maximum | Selected epoch | Validation early-miss recall |
|---|---:|---:|---:|
| Small P4 semantic | 6 / 16 | 1 | 63.92% |
| Small P4 without semantics | 8 / 16 | 3 | 63.92% |
| Large P4 semantic | 6 / 16 | 1 | 57.29% |
| Large P4 without semantics | 6 / 16 | 1 | 56.61% |

These validation recall scores are not the original model's held-out Top-1 accuracy. They measure coverage of next-query early-miss targets at a 16-page proposal budget. Every selected checkpoint has nonzero trained residual weights and was saved and reloaded before runtime evaluation.

## Evidence and limits

- **154 matched trials, 1,540 evaluation query executions**, plus separate training and validation traces. Seven policies, two workload patterns, zero/50 ms gaps and zero/0.2 ms artificial service were tested on each database; the declared focus had three additional workload seeds.
- All SQL row counts and row-stream hashes matched across policies. All useful idle insertions completed before their demand query arrived. Zero-gap idle trials inserted zero pages. Full accounting, all conditions, paired query deltas, CPU, reads and pollution proxies remain available.
- Gains are from controlled pilots with familiar workloads and imposed idle windows. They do not establish physical-storage or general speedup. OS cache was not reset. Novel-template and zero-gap conditions do not consistently improve over P0.
- CPU overhead remains unresolved: large P4 used about 2,990 ms of process CPU per focused trial versus about 47 ms for full-page-set idle Markov, while their wall-clock query latencies were almost equal. Process CPU includes background threads. No causal native-threading explanation is claimed.
- The pilot source snapshot is preserved. A later diagnostic-only fix makes `IdleRuntime.close()` publish its actual clean shutdown return value into the inherited shutdown flag; the captured pilot's flag retained its default false even though the close return and `worker_alive=false` verified completion. A fresh real-SQL P4 smoke test and the full suite validate the fix; raw results were not rewritten.
- **160 backend tests pass.** Original frozen model/source/database compatibility was independently rechecked through `live_hybrid.assets()`.

Local evidence:

- [Complete report](runs/idle-neural-20261106/final-report-001/RESULTS.md)
- [All trial CSV](runs/idle-neural-20261106/final-report-001/all_trials.csv)
- [Paired query deltas](runs/idle-neural-20261106/final-report-001/paired_queries.csv)
- [Audit](runs/idle-neural-20261106/final-report-001/audit.json)
- [Release verification](runs/idle-neural-20261106/final-report-001/release-verification.json)
- [Small training/protocol](runs/idle-neural-20261106/small-pilot-001/protocol.json)
- [Large training/protocol](runs/idle-neural-20261106/large-pilot-001/protocol.json)

Generated evidence, databases and weights are ignored by Git. This document and implementation retain the reproducible protocol in source control; the local artifacts are required to repeat the original-database tests.

## Run locally

From `backend/`, choose new output directories. Existing outputs are never overwritten.

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=runs/pytest-idle-new
.\.venv\Scripts\python.exe -u idle_neural_experiment.py --out runs/idle-neural-new/small --queries 12 --seed 8001 --epochs 16
.\.venv\Scripts\python.exe -u idle_neural_experiment.py --database runs/large-branch-20261015/pagewise-large.sqlite --model-dir runs/neural-improvement-20261020/frozen-final/params --out runs/idle-neural-new/large --queries 8 --seed 8002 --epochs 16
.\.venv\Scripts\python.exe summarize_idle_neural.py runs/idle-neural-new/small runs/idle-neural-new/large --out runs/idle-neural-new/report
```

Implementation: `idle_neural.py`, `idle_neural_experiment.py`, shared `idle_markov.py` / `idle_runtime.py` / `idle_experiment.py`, and `summarize_idle_neural.py`. No frozen `pagewise/` or `backend/scripts/` source was modified.
