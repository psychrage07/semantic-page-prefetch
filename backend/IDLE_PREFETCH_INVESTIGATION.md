# Query-level idle-prefetching investigation

The first phase followed the requested first three priorities: inspect SeLeP, locate real query lifecycle events, and implement a simple query-level Markov idle baseline. It did not retrain the frozen research model or change the frontend results. The follow-up P4 query-level neural adaptation is now implemented and evaluated separately; see [idle neural results and reproduction](IDLE_NEURAL_RESULTS.md).

## Source study

SeLeP's paper treats completed query result-block sets as a sequence and predicts the next query's partitions. It states that prefetching occurs between user requests and terminates on a new query; its reported approximately 64 ms SDSS prediction/prefetch overhead is workload-specific, not a guaranteed idle window for Pagewise. The inspected public `selep_main.py` evaluation path runs queries and then calls a synchronous cache-prefetch helper, reporting query and prefetch time separately. It does not expose an actual arrival-driven cancellation loop. The provided LRU simulation evicts its oldest entry when capacity is exceeded; it has no separate unused-prefetch eviction priority. These implementation observations are limited to the inspected evaluation path.

Sources: [SeLeP VLDB paper, p. 10](https://ai-db-uom.github.io/data/2024_vldb_selep.pdf), [authors' source](https://github.com/fzirak/SeLeP/blob/0b896c4ff6bb3ec04b8a9a1d4454061cfbfb062d/selep_main.py), [cache](https://github.com/fzirak/SeLeP/blob/0b896c4ff6bb3ec04b8a9a1d4454061cfbfb062d/Backend/Database/LRUCache.py), [PostgreSQL helpers](https://github.com/fzirak/SeLeP/blob/0b896c4ff6bb3ec04b8a9a1d4454061cfbfb062d/Backend/Database/helpers.py). Read-only reference copies are under the isolated run's `references/` folder. No downloaded source is executed.

PredictIO separates user-space prediction from a bounded Linux kernel executor. Its B-tree policy receives upcoming child-page hints from an engine adapter; the executor validates ranges and filters cached/in-flight pages. Pagewise can adopt bounded admission, duplicate suppression and prediction/execution separation in APSW, but its VFS cache is not that kernel backend. Parsing an already-read B-tree page exposes children, not the engine cursor's exact future traversal. This phase uses the portable scheduling principles rather than claiming a PredictIO reproduction.

Source: [PredictIO authors' preprint, sections 3.4, 4.2 and 5.3](https://arxiv.org/html/2610.12971v1).

## Current problem: what is actually measured

The 2,228 prefetched / 2 useful figure is verified in `NEURAL_IMPROVEMENT_RESULTS.md`, zero-delay scenario `k1-delay0.0`, and `public/pagewise-neural-improvement.json`. Its useful fraction is 0.090%; scoring is 1.266 ms per scored job, mean queue wait is 27.874 ms per dequeued job, and there are 4,466 stale drops. Those are original-study aggregates, not this new idle experiment. The scoring counter combines candidates, features and model execution; it cannot identify their individual contributions retrospectively.

New P2 trials time candidate generation, semantic feature construction and the existing MLP forward call separately. Shadow-LRU instrumentation tracks actual demand misses where a same-capacity demand-only cache would hit. Prefetch insertions, useful uses, unused evictions/residency, VFS read bytes, CPU, queue time and initialization/shutdown costs are exported. These diagnostics support investigation; they do not make every latency difference a causal attribution.

## Lifecycle and implementation

`pagewise/runner.py` already marks a query active before executing SQL and clears it in `finally`; `workspace.py` has equivalent boundaries. `tracing.py` calls the runtime on VFS reads. The existing runner immediately proceeds to the next query after pager cleanup: it has no measured human think-time workload and its traces do not contain exact historical query completion/arrival pairs. Consequently, old xRead timestamps cannot prove that natural 25/50/100 ms idle windows were available.

The new opt-in files live outside `pagewise/` and `scripts/` so the original frozen source hash remains valid:

- `idle_markov.py`: first/second-order prediction from completed query-template history or a completed-page-set fingerprint. Counts, probabilities and candidate pages come only from training sessions. Session boundaries reset history. Predictions do not read the next query's SQL, parameters, template or result pages.
- `idle_runtime.py`: a bounded single worker starts after completion and admits reads only while idle. Arrival invalidates the generation and cancels artificial service without joining the worker. Cached/pending pages are skipped; a fixed confidence/budget filter bounds proposals. A speculative cache quota and hot-page protection bound pollution, and unused idle pages expire after the next query. The process cannot preempt an already-admitted real OS read.
- `idle_experiment.py`: actual read-only APSW SQL through the existing TraceVFS and cache. Every policy runs identical SQL/results; order is randomized. Query arrival/end, actual gap, proposals, read admissions, cache insertions and uses are recorded. Query-level models are fixed before evaluation.
- `summarize_idle.py`: audits SQL correctness, delay-device wiring, admission event boundaries, usefulness accounting and protected artifacts; exports all trial CSV/JSON and an unfiltered report.

The main P3 configuration is declared in advance as Markov-2 with template history. Markov-1 and completed-footprint states are ablations. P0 has the same bounded custom cache with prefetching disabled, to avoid confusing cache-capacity changes with prefetch benefit. P1/P2 retain the existing within-query Markov/neural policies and their declared scheduling settings. A separate stock-SQLite baseline would answer an additional practical-overhead question.

## Reproduction

From `backend/`, use a new output directory on each attempt:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_idle_prefetch.py -q -p no:cacheprovider --basetemp=runs/pytest-idle-new
.\.venv\Scripts\python.exe -u idle_experiment.py --out runs/idle-phase-new/small --queries 12 --seed 6002
.\.venv\Scripts\python.exe -u idle_experiment.py --out runs/idle-phase-new/large --database runs/large-branch-20261015/pagewise-large.sqlite --model-dir runs/neural-improvement-20261020/frozen-final/params --queries 8 --seed 7002 --quick
.\.venv\Scripts\python.exe summarize_idle.py runs/idle-phase-new/small runs/idle-phase-new/large --out runs/idle-phase-new/report
```

The default small database/checkpoint belongs to demo `13d383a70d3c`; these ignored artifacts must exist locally. The full small matrix covers requested gaps 0/25/50/100 ms, 16/128 cache pages, no/0.2 ms artificial service delay, and familiar/novel query-level templates. The large pilot is deliberately narrower. Gaps are controlled conditions, not observed natural user idle times. Repeated workloads intentionally reuse familiar values; novel templates are absent from query-level training but may occur in the frozen neural baseline's older corpus. Initial attempts with IDs `smoke-001` and `small-matrix-001` are explicitly excluded because of speculative turnover and incorrect demand-delay wiring; they are preserved for audit.

## What follows this baseline

P4 must be an explicitly trained query-level neural residual using only completed-query information. Reusing the old within-query checkpoint and calling its next-page output a next-query page-set prediction would be invalid. Establish which P3 proposals arrive before demand, quantify waste/pollution, and replicate on new development sessions before adding neural execution cost. Then freeze a new query-level model and scheduler and evaluate P0-P4 on untouched sessions with repeated workload seeds. Preserve the full no-prefetch controls in experimental evidence regardless of frontend display preferences.

## Accepted results from this phase

Accepted outputs are `runs/idle-phase-20261106/small-matrix-002/` (224 trials, 2,688 evaluated query executions) and `large-pilot-002/` (56 trials, 448 evaluated query executions on the original 8.62 GB snapshot). Training traces are separate. The audited report and complete CSV/JSON are under `runs/idle-phase-20261106/final-report-002/`. All SQL row streams matched across policies, workers were clean, and database fingerprints, model checksums and frozen source hashes were unchanged. The full backend suite passed 156 tests, including nine new idle-prediction, cancellation, admission, cache-accounting and real shared-delay wiring tests.

The following condition was declared for the report before examining outcomes: familiar query pattern, 50 ms requested gap, 0.2 ms artificial shared-service delay, 32-page cache, eight queries on the large snapshot. This is a small single-seed pilot, not a confirmed speedup.

| Policy | Mean query ms | p95 query ms | Prefetch insertions | Useful pages |
|---|---:|---:|---:|---:|
| P0: cache-matched no prefetch | 37.669 | 84.112 | 0 | 0 |
| P1: current Markov-2 | 51.698 | 125.874 | 156 | 7 |
| P2: current neural-Markov | 42.244 | 95.595 | 9 | 0 |
| P3: idle Markov-2, template history | 36.133 | 82.178 | 23 | 21 |

All 21 useful P3 insertions in that condition completed before the next query's recorded arrival. This table represents one condition; the full accepted matrix has 40 conditions. P3's mean was higher than P0 in 13 of them. Requested zero-gap idle trials inserted zero pages and had zero useful idle prefetches. Across all four P3 ablations and accepted conditions, 4,753 pages were inserted, 1,952 were used, and all 1,952 useful insertions completed before recorded query arrival. The shadow-cache probe still found 14 pollution misses across those ablations. These are mechanistic and exploratory observations, not population estimates or a replacement for the original study.

The new stage probe in the large table's P2 trial measured approximately 0.041 ms for candidates, 0.364 ms for semantic features and 0.258 ms for the MLP forward call per respective invocation; queue wait averaged 0.109 ms per job. These are newly measured components under this pilot, not a decomposition of the old 1.266 ms aggregate. Process CPU was much higher for P2 than for P3, and native threading/spin behavior remains a profiling question; `threading-diagnostic.json` records the local PyTorch thread settings without attributing the excess CPU to an unmeasured cause.

The next step is replication on additional sessions and a native-SQLite practical baseline, followed by a query-level P4 model trained to prioritize useful early misses within the same bounded scheduler. Keep full-page-set recall distinct from early-miss coverage: a 16-page budget cannot cover a large scan's thousands of distinct pages, regardless of ranking accuracy.
