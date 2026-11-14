# Idle neural hybrid: CPU optimization and expanded evaluation

The saved P4 semantic neural hybrid now has an opt-in optimized execution path: passive native-thread waiting, bounded feature/metadata caches, and a separately labeled selective neural gate. The existing P4 weights, original selected research model, 8.62 GB database and frozen `pagewise/` / `backend/scripts/` source were preserved. No new neural accuracy or retraining claim is made.

## What was improved

Fresh-process diagnosis reproduced high CPU consumption without retraining or prior policy execution. Two 24-query small-database P4 runs used 3,031/3,062 ms of process CPU under inherited waiting settings. With `OMP_WAIT_POLICY=PASSIVE` and `KMP_BLOCKTIME=0`, the corresponding diagnostic runs used 78/31 ms. Additional runs with a preceding P2 workload and single-thread settings are preserved. These switches strongly implicate avoidable native-thread waiting; the experiment changes both waiting variables together and does not identify each variable's separate contribution.

The runtime applies these settings to new child processes before importing numerical libraries. It does not change the user's global environment or the original model's files. The original idle-neural and original within-query neural controls retain inherited settings.

Bounded caches reuse training-derived Markov proposals, immutable page semantics, completed-query context and history/candidate feature rows. The full optimized policy still runs the same trained semantic MLP. Its scores and rankings matched the original on **184 validation/exclusion cases** across both databases, within `1e-6` for logits. Feature rows are capped at 4,096; metadata rows at 4,096; prior and query-context memo tables at 128 each. The default full-neural optimized path does not skip semantic scoring when candidates exist.

## Expanded comparison

There are **182 accepted trials and 4,704 evaluation query executions**, using a fresh process for every policy/condition, randomized policy order and identical SQL across matched policies. This work is separate from the November 6 pilot and original frozen study.

The following two conditions each use five seeds, 24 queries per policy/seed, 50 ms controlled gaps and a 32-page cache:

- **Familiar, delayed:** familiar training values; 0.2 ms requested artificial shared-storage service.
- **Fresh, no delay:** SQL/parameter pairs absent from P4 training; no artificial service delay. Query templates and the session pattern remain familiar.

CPU is the mean process CPU time per 24-query trial, including process threads. Query time is the mean elapsed SQL time. They are different measurements.

| Database / condition | Original idle neural CPU | Optimized full-neural CPU | CPU reduction | Original idle neural query mean | Optimized query mean |
|---|---:|---:|---:|---:|---:|
| Small / familiar, delayed | 2,693.75 ms | 62.50 ms | 97.68% | 5.404 ms | 5.189 ms |
| Small / fresh, no delay | 2,846.88 ms | 56.25 ms | 98.02% | 1.489 ms | 1.466 ms |
| 8.62 GB / familiar, delayed | 2,865.63 ms | 265.63 ms | 90.73% | 39.590 ms | 40.499 ms |
| 8.62 GB / fresh, no delay | 3,153.13 ms | 131.25 ms | 95.84% | 5.221 ms | 5.705 ms |

The clear improvement is CPU efficiency while preserving full-neural inference. Query-time differences vary in sign. All four exploratory paired five-seed bootstrap intervals for optimized minus original idle-neural mean latency include zero; no consistent additional latency speedup is established.

| Database / condition | Original within-query neural mean | Idle Markov mean | Optimized idle neural mean | Optimized useful / inserted |
|---|---:|---:|---:|---:|
| Small / familiar, delayed | 8.621 ms | 5.073 ms | 5.189 ms | 224 / 305 |
| Small / fresh, no delay | 3.312 ms | 1.411 ms | 1.466 ms | 250 / 380 |
| 8.62 GB / familiar, delayed | 48.565 ms | 40.432 ms | 40.499 ms | 419 / 497 |
| 8.62 GB / fresh, no delay | 8.683 ms | 3.977 ms | 5.705 ms | 525 / 904 |

All controls, including cache-matched no prefetch, and every other condition remain in the full report and exports. Additional single-seed tests cover controlled gap sequences of 0/5/25/50/100 ms, unfamiliar query-level templates with a 128-page cache, and zero gaps with a 16-page cache.

OS cache is not reset; first-policy/cold-page effects and scheduling variation remain possible despite randomized order. Artificial delay is not measured physical-disk latency. Low CPU counters have Windows timer granularity. These are expanded development measurements, not a final general-speedup confirmation.

## Selective neural gate

The gate was selected before evaluation using the existing validation records only. It uses a minimum support of three and screens empirical confidence margins of 0.1/0.25/0.5 at an eight-page budget. Selection minimizes neural calls subject to losing at most 0.5 percentage points of validation macro early-miss recall. Both databases selected margin 0.5 with no measured validation recall loss.

The gate uses Markov ordering when all candidates fit the admission budget or the empirical prior clearly separates its cutoff. Otherwise it invokes the saved semantic MLP. It logs its route explicitly. A Markov-only job is never reported as a neural inference.

| Comparison | Full-neural calls | Selective neural calls |
|---|---:|---:|
| Small validation | 46 | 36 |
| Large validation | 46 | 46 |
| Small familiar-delayed evaluation, five seeds | 107 | 49 |
| Small fresh-no-delay evaluation, five seeds | 112 | 72 |
| Large familiar-delayed evaluation, five seeds | 115 | 115 |
| Large fresh-no-delay evaluation, five seeds | 115 | 115 |

The gate saves calls in some small-model workloads but has not shown a consistent latency advantage. It stays an explicit experimental comparison. The full cached-neural policy is the primary optimized neural result.

## Local frontend

**Live demonstration -> Test idle neural hybrid** now provides:

- Small local or original 8.62 GB database selection; 10/20/30 queries per policy.
- Saved P4 weights with checksum verification; no fitting in the browser run.
- Fresh matching SQL workloads, real APSW/VFS traces, prefetch counts, process CPU, mean query latency and actual neural-call counts.
- A visible live Python terminal, cancellation and downloadable complete results.
- A controlled 50 ms pause and no artificial storage delay, disclosed on the page.
- Comparisons with the original within-query neural model, idle Markov, original idle neural, optimized full-neural and selective execution. The no-prefetch control remains in raw downloads; existing frontend presentation preferences are preserved.

The live demo is separate from the saved research study. It requires the matching local artifacts and prepared optimization plans. Their absence disables the run with a reason. Model weights and generated experiment data remain ignored by Git.

## Verification

- All 182 trials passed SQL result hashing, cache accounting, worker shutdown, protected-artifact and idle-admission audits. All useful idle insertions preceded their demand query's arrival.
- **166 backend tests passed**, including score/ranking parity with eviction and exclusions, metadata/context cache separation, selective routing, variable query gaps and local-job validation.
- TypeScript checking and the production frontend build passed.
- `verify_idle_live.py` exercises the real frontend/API proxy, both database options, generated jobs, artifact downloads, input rejection and cancellation. Its saved verification file records the completed outcome.
- Visual browser QA is unavailable in this session: the computer-use inventory has no connected browser, and the in-app browser is unavailable. HTTP/API checks and compilation do not substitute for visual browser inspection.

Accepted outputs:

```text
backend/runs/idle-optimization-20261114/
    cpu-diagnostic-001/
    small-selection-001/optimization-plan.json
    large-selection-001/optimization-plan.json
    small-evaluation-002/
    large-evaluation-001/
    final-report-001/
    api-e2e-001/
```

`small-evaluation-001` is excluded: a development import error stopped it before any completed trial. Its exclusion record is preserved. Accepted trial implementations were hashed and unchanged throughout evaluation.

Full local evidence: [all-condition report](runs/idle-optimization-20261114/final-report-001/RESULTS.md), [all trials CSV](runs/idle-optimization-20261114/final-report-001/all_trials.csv), [paired-seed comparisons](runs/idle-optimization-20261114/final-report-001/paired_seed_comparisons.json), [audit](runs/idle-optimization-20261114/final-report-001/audit.json), [API verification](runs/idle-optimization-20261114/api-e2e-001/verification.json).

## Reproduce

From `backend/`, use new output directories and retain the original pilot artifacts needed for the saved model and validation-only selection:

```powershell
.\.venv\Scripts\python.exe -u profile_idle_cpu.py --out runs/idle-optimization-new/diagnosis
.\.venv\Scripts\python.exe -u optimize_idle.py prepare --dataset small --out runs/idle-optimization-new/small-selection
.\.venv\Scripts\python.exe -u optimize_idle.py prepare --dataset large --out runs/idle-optimization-new/large-selection
.\.venv\Scripts\python.exe -u optimize_idle.py run --plan runs/idle-optimization-new/small-selection/optimization-plan.json --out runs/idle-optimization-new/small-evaluation --queries 24
.\.venv\Scripts\python.exe -u optimize_idle.py run --plan runs/idle-optimization-new/large-selection/optimization-plan.json --out runs/idle-optimization-new/large-evaluation --queries 24
.\.venv\Scripts\python.exe summarize_idle_optimization.py runs/idle-optimization-new/small-evaluation runs/idle-optimization-new/large-evaluation --out runs/idle-optimization-new/report
```

The browser helper currently uses the accepted November 14 plan locations listed above. Start the normal local application from the repository root with `npm.cmd run build` followed by `npm.cmd run preview:local`.

Implementation is in `profile_idle_cpu.py`, `idle_optimized.py`, `optimize_idle.py`, `idle_live.py`, `summarize_idle_optimization.py` and `verify_idle_live.py`, with integration in `local_lab.py`, `LocalLab.tsx` and `IdleHybridLab.tsx`.

References for interpreting native settings: [LLVM OpenMP runtime waiting behavior](https://openmp.llvm.org/design/Runtimes.html), [Intel supported environment variables](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2025-0/supported-environment-variables.html), [PyTorch threading environment](https://docs.pytorch.org/docs/stable/threading_environment_variables.html).

Remaining research work is to measure naturally occurring idle opportunities, investigate unfamiliar-template candidate coverage and establish independent-device/longer-workload benefits. Neither cached inference nor passive waiting changes the original prediction accuracy or proves semantic superiority over idle Markov.
