# Selective Pagewise prefetching handoff

Date: 27 October 2026. Work is on `main`. The requested queue-optimization branch, prior handoff, H1 audit and queue experiment harness were unavailable locally and on GitHub. At the user's direction the missing work was implemented from the existing repository; pasted H1 numbers were not treated as verified results. No branch was created and nothing was pushed.

## Implemented

- `backend/selective_context.py` keeps Markov-2 for known second-order histories, permits neural fallback only for development-supported families, blocks recursive fallback by default, and otherwise uses Markov-1 or skips.
- `backend/selective_deadline.py` checks a development-estimated deadline before work can proceed.
- `backend/selective_adaptive.py` implements an interpretable net-benefit heuristic using measured prediction accuracy, usefulness, read/scoring/queue costs and current queue/cache pressure. Missing evidence causes a skip.
- `backend/selective_prefetch.py` integrates these rules into the real APSW prefetch runtime as an opt-in subclass. It bounds per-query requests, coalesces in-flight jobs, filters cached/pending/duplicate pages, cancels completed-query and deadline-expired work, logs lifecycle timestamps and reasons, and tracks cost/late-read counters. Speculative simulated-device access is nonblocking when demand is already waiting or the device is busy. An operation already in progress and actual OS reads cannot be preempted, so demand priority is best-effort.
- `backend/queue_experiments.py`, `backend/calibrate_selective.py`, `backend/summarize_selective.py`, `backend/enrich_selective_evidence.py` and `backend/write_selective_report.py` provide isolated trials, development calibration, frozen policy checks, paired analysis and reporting. The opt-in runtime adapter is installed in each trial's own Python process; it is not safe as an unguarded global monkeypatch in a concurrently shared process.
- Focused tests cover context selection, deadline boundaries, missing or unprofitable estimates, queue limits, in-flight duplicate coalescing, completed-query cancellation, candidate filtering, device contention, real VFS execution, and actual/approximate reporting alignment.

The new files live outside `backend/pagewise/` and `backend/scripts/` so the original frozen predictor source digest remains `67df304d39d11bc4fb14bd5679d5f5e7e4dd926f2370760820a831b1a122afd2`. The existing trained-hybrid live demo still verifies its original source and checkpoint. The new experiment is available through its local Python harness; no new dashboard controls were added.

## Commits

| Commit | Change |
|---|---|
| `014028779fb9650e34b311343c83d956c20021af` | Isolated baseline harness |
| `cd33a7fdfe0f9f902ab1c50cf7dd142ba8196cd6` | Context and fallback rules, tests |
| `84998dc006b6ece3ed8a6b595ce18ffa5cab30b4` | Deadline checks, tests |
| `d64bb8a8a5ddc8e4d01375164410c86f2f9608e7` | Interpretable benefit controller, tests |
| `cb6e610f2924dea4eefc00a8b3146f0659a119b0` | Opt-in runtime integration and worker tests |
| `07ae5f83fa84206746c5592d29bc184d1fd0ded9` | Development calibration, ablation and held-out harness |
| `3d7e7e6e52994f587718760997f2b5c36f97c218` | Per-family analysis and reporting tests |
| `002ebb62fa0d5aa44c110b27f76761f1b0907798` | Frozen policy, protocol, aggregate evidence and report |

The handoff itself is committed immediately after the evidence commit. `git log -1 --format=%H` identifies its commit. Pre-existing uncommitted SDSS/frontend changes remain outside these commits.

## Frozen experiment

- Development seeds: **1101 and 1102**. Held-out seeds: **4401–4405**. No matching seed appeared among 561 available runtime metadata files; unavailable prior queue artifacts cannot be checked.
- Two scenarios: zero injected delay, and a **0.2 ms requested artificial shared-channel delay**. Five seeds × 30 matched queries × seven policies × two scenarios = **70 held-out trials / 2,100 policy-query executions**. Diagnostic logging was enabled for every policy; each trial used a fresh process, a 512-page Pagewise cache, a 4096 KiB SQLite pager and one worker. The OS cache was not reset. Policies were randomized per seed and scenario.
- Frozen checkpoint SHA256: `23bafd4cce8cc40ddc5dacef7fb4990972cfba985b9d5bf779082b7dd31bf973`. Frozen policy SHA256: `cc3ebc9665a1161fa9e3302af9b10b6527277779118bdfb8e9ea247d37250eb1`. See [policy](backend/selective-prefetch-policy.json) and [protocol](backend/selective-prefetch-protocol.json) for full settings, cost estimates, source hashes and seed rules.
- Development accuracy covered all **16,539** reference transitions; exact Markov-2 context occurred in **84.58%**. The family eligibility rule required at least 40 fallback transitions, 10 queries, both development seeds, and a positive lower 95% paired query-bootstrap bound versus Markov-1. **No family qualified**, so neural fallback was not enabled. Indexed lookup had only three fallback transitions; recursive fallback remained disabled.
- Deadline budgets are per-family median observed intervals between consecutive development VFS reads (approximately 0.023–0.040 ms), not guaranteed actual target deadlines. No uncalibrated neural-score confidence threshold was added. Net benefit uses development-estimated accuracy, usefulness, queue wait, scoring/read costs, miss cost and live pressure; Markov-1 cost uses Markov-2 as a proxy. The per-query request cap is eight.
- A preliminary development context run overlapped an instrumentation change. It was preserved under `development-excluded-runtime-changed/` and excluded. All reported development ablations ran again under fixed code. The policy and source hashes were frozen before any held-out run. Per-family accuracy confidence intervals were added in a later analysis pass without changing policy or measured outcomes.

## Results and hypothesis verdicts

The primary goal—repeatable lower query latency versus no prefetching—**was not achieved**. Paired latency deltas are candidate minus no prefetch; intervals resample seeds and queries within seeds and are descriptive, unadjusted for multiple comparisons.

| Condition | None mean | Combined mean | Delta vs none | 95% paired CI | Verdict |
|---|---:|---:|---:|---|---|
| No injected delay | 9.321 ms | 8.804 ms | −0.517 ms | [−3.692, 1.853] ms | Inconclusive |
| Requested 0.2 ms delay | 94.755 ms | 95.496 ms | +0.741 ms | [0.173, 1.464] ms | Higher latency in this interval |

In zero-delay trials the existing Markov-2 and neural ranker averaged 15.929 ms and 21.510 ms; combined was lower than both, but also prefetched **zero pages**. In controlled-delay trials they averaged 114.328 ms and 112.198 ms; combined again prefetched zero pages. These comparisons do not show a benefit over no prefetching.

- **Context:** no neural fallback was validated, and the held-out context policy scored zero neural jobs. Offline query-average Top-1 was **43.21%** versus **45.98%** for the existing neural hybrid. It did not preserve the measured neural prediction advantage. Context still issued Markov work and inserted 213/235 prefetched pages in zero/controlled-delay trials, of which 15/54 were used.
- **Deadline:** rejected all 668 otherwise-eligible requests before enqueue in each condition and inserted no pages. Its latency reduction versus context accompanies avoiding speculative work; it does not establish timely-prefetch benefit.
- **Adaptive:** rejected all otherwise-eligible actions in both conditions. Its estimated benefit rule did not find a profitable opportunity in this workload.
- **Combined:** likewise inserted no pages and made no actual ranked predictions. Offline scoring that counts skips as errors is 0%; the conditional actual-job accuracy is `n/a`, because there were no scored jobs. This distinction is explicit in the report.
- **Resource cost:** combined zero-delay peak process RSS was **382.8 MB** versus **303.3 MB** for none; total CPU across five trials was **12.92 s** versus **3.92 s**. The adapter still eagerly loads the neural checkpoint even though no fallback was enabled. Scheduling CPU cost was not separately isolated.

[Full results](backend/SELECTIVE_PREFETCH_RESULTS.md) contain every policy's mean/p95, paired latency intervals, per-family latency and prediction metrics with confidence intervals, runtime job alignment, usefulness/waste, cancellation/drop counters and limits. [Machine-readable results](backend/selective-prefetch-results.json) contain all per-family comparisons and resource counters. Raw queries, traces, predictions and logs remain ignored in `backend/runs/selective-prefetch-20261027/`.

## Verification and limitations

- Full backend suite: **145 passed**. After the separate family-interval analysis was added, its three focused reporting tests also passed.
- Pinned TypeScript check `node_modules/.bin/tsc --noEmit` and production `npm run build` passed.
- All **70 held-out trials** had matching workload hashes, clean worker shutdown, zero prefetch read errors and recorded peak RSS. Frozen source and actual live-demo assets still verify.
- Offline policy accuracy is a common-reference approximation with zero queue/cache pressure; it excludes actual per-query caps and counts skipped predictions as errors. Actual runtime job accuracy is separately calculated only for recorded, aligned scoring jobs; ambiguous legacy histories are excluded. Logged candidate lists are truncated and are not a full candidate-recall measure.
- Family samples are small; there are only five workload-seed clusters, multiple comparisons are not adjusted, parameter values may overlap earlier training, and OS caching is uncontrolled. The requested delay and channel are artificial. The code cannot guarantee that an already-started speculative operation never delays a demand read.
- A useful next independent cycle would lazily load the neural model only when development evidence enables fallback and evaluate a less restrictive observable demand horizon. Any changes need fresh development calibration and new untouched test seeds; do not tune to the held-out results above.

## Reproduction

The frozen checkpoint and approximately 8.62 GB database are ignored local artifacts, not included in the repository. A fresh checkout needs the matching frozen artifacts and the same Python dependencies. The source-hash check is byte-exact, including line endings.

From `backend/`, with the existing output directory safely preserved under a different name so `initialize` cannot overwrite it:

```powershell
.\.venv\Scripts\python.exe queue_experiments.py initialize
.\.venv\Scripts\python.exe queue_experiments.py baseline
.\.venv\Scripts\python.exe calibrate_selective.py
.\.venv\Scripts\python.exe queue_experiments.py development --policy context
.\.venv\Scripts\python.exe queue_experiments.py development --policy deadline
.\.venv\Scripts\python.exe queue_experiments.py development --policy adaptive
.\.venv\Scripts\python.exe queue_experiments.py development --policy combined
.\.venv\Scripts\python.exe queue_experiments.py freeze
.\.venv\Scripts\python.exe queue_experiments.py heldout
.\.venv\Scripts\python.exe summarize_selective.py
.\.venv\Scripts\python.exe enrich_selective_evidence.py
.\.venv\Scripts\python.exe write_selective_report.py
```

These commands repeat the declared experiment; they are not fresh independent confirmation because they reuse the same seeds. New research must predeclare different seeds and preserve the current results.
