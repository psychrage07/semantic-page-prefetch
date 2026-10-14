# Pagewise Final Evaluation Protocol v1

## Status and scope

This protocol is frozen before the final-suite commands are run. Earlier suites,
including `suite-hybrid-dev` and `suite-hybrid-validation`, are tuning and pilot
artifacts. They are not pooled with the final results.

## Hypotheses and endpoint

The primary endpoint is the paired mean query-latency difference against the
custom bounded-cache `none` policy. Negative values favor the candidate policy.
The independent replication unit is a workload seed; paired query observations
are retained for effect-size estimates and a query-level bootstrap confidence
interval. A result is reported as inconclusive when its interval crosses zero.

Secondary endpoints are run-level p50/p95, total query time, worker-inclusive
wall time, cache hit/miss/eviction counts, useful and wasted prefetches,
prefetch bytes, inference time, queue pressure, and worker drain time.

## Fixed implementation and data

- Database: `runs/experiment-latest/pagewise-medium.sqlite`.
- Model directory: `runs/experiment-latest/models`.
- Workload: the existing mixed generator, which samples indexed point lookups,
  joins, scans, and traversal templates; query family is retained per row.
- Strategies: `stock`, `none`, `sequential`, `markov1`, `markov2`, `model_a`,
  `model_b`, and `hybrid`.
- Custom-cache settings: 4096 KiB SQLite cache, 32 custom cache pages, degree 2,
  queue size 32, miss-only scheduling, every eligible observation, candidate cap 12.
- Seeds: 901, 902, and 903; 100 queries per seed and strategy.
- Strategy order: rotated by `run-suite` for every seed.

The hybrid gate is frozen: use a training-only exact Markov-2 transition when it
exists, otherwise score Model B candidates. No held-out final result selects a
threshold, model, cache size, degree, or queue size.

## Categories

Category A runs the fixed setup with injected miss delays of 0.2 ms and 1.0 ms.
These results are labelled `latency-injected` and are not evidence of physical
storage performance.

Category B runs the same setup with no injected delay. It is labelled
`real-storage-no-injected-delay`. Windows OS-cache state is not reset or proven
cold, and VFS reads may be satisfied by that cache. It must not be described as
measured SSD I/O.

## Analysis and exclusions

All completed matched queries with a shared seed and query ID are included.
Malformed rows, failed database reads, non-clean worker shutdowns, and missing
query result files invalidate the corresponding run and are reported rather
than silently excluded. Confidence intervals use the existing 2,000-resample
paired query bootstrap. Stock results are retained as a practical reference,
not a same-cache causal comparison.

## Reproduction

Run `scripts/run_final_evaluation.ps1` from the `backend` directory. If a
terminal interruption occurs after individual runs finish, use
`python -m pagewise.cli recover-suite --out <suite directory>` before analysis.
