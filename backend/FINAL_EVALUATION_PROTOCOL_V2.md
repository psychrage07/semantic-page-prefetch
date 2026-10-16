# Pagewise Final Evaluation Protocol v2

Frozen **before** any v2 tuning or final run. After freezing, this file may change
only by **appending dated amendments** at the bottom. v1 (`FINAL_EVALUATION_PROTOCOL.md`)
is retained unchanged as the historical pre-registration.

## Scope

The primary research question: do semantic features (object, page type, B-tree depth,
query-plan node type, query template) improve learned page prefetching over a
history-only MLP, and does any prediction gain survive inference and I/O overhead as a
net latency win?

## Storage model and delays

- Primary storage model: **shared** (`--storage-model shared`), a single serialized
  device. `independent` is retained only for calibration/diagnostic comparisons.
- Device service time is the **measured** achieved delay, not the labelled one. On this
  host `time.sleep` overshoots sub-millisecond requests (0.2 ms requested -> ~0.56 ms
  measured; 1.0 ms -> ~1.54 ms). Every report must print the measured sleep error and
  caveat any earlier result labelled 0.2 ms / 1.0 ms.
- Spin-waits are **not** used in the threaded runtime (they hold the GIL and starve the
  worker thread). Delay accuracy is instead achieved by choosing delay values at/above
  the timer floor and reporting the residual error.

## Strategies (all reported in every table)

`none`, `stock`, `sequential`, `markov1`, `markov2`, `model_a`, `model_b`, `hybrid`,
`null`, `oracle_next` (the original trace-next oracle, relabelled a naive control), plus
new v2 rows: `oracle_miss`, `markov_ctx`, `model_b_table`, `model_b_gated`.

No strategy is ever removed from a table because it performs badly. Metric definitions
are unchanged from v1.

## Metrics

- **Primary:** total workload time (`median_total_query_ms`, and run totals) under
  `--storage-model shared`.
- **Secondary:** p50 and p95 query latency, top-1 / top-k accuracy, useful-prefetch %,
  prefetch lead time, queue wait, inference \(\mu s\)/prediction.

## Hypotheses (verbatim)

Primary metric: total workload time under `--storage-model shared`. Secondary: p50 and
p95 query latency, top-1/top-k accuracy, useful-prefetch %.

- **H1:** On held-out query parameters, Model B has higher top-1 accuracy than Model A.
- **H2:** Model B beats `markov_ctx` (context-keyed Markov table, same key as the semantic
  features, no neural network) and `markov2` in top-1 accuracy.
- **H3:** The best Model B variant has total time below `none`.
- **H4:** The best Model B variant has lower total time than `sequential`, `markov2`,
  `markov_ctx` and `model_a`.
- **H5:** H1 and H3 still hold on held-out query templates.

Each outcome is one of: **supported**, **not supported**, or **inconclusive** (CI
includes zero).

## Success rule (win definition)

A strategy counts as a win over another only if the **95% paired bootstrap CI** of the
paired difference (paired by seed **and** query id) **excludes zero in its favor**, under
`--storage-model shared`. The bootstrap uses **10,000 resamples**. Negative deltas favor
the candidate (candidate latency − baseline latency).

For the H3/H4 comparison families, apply **Holm correction** across the family's tests and
report both the corrected decisions and the raw CIs and effect sizes (mean paired delta,
median paired delta, improved-query %).

## Primary configuration rule (chosen before the final run)

The primary configuration is the cell of the headroom grid (cache pages ∈ {32, 64, 128,
256, 512} × delay ∈ {0.1, 0.2, 0.5, 1.0} ms) that satisfies **both**:

1. cache is **much smaller than the working set** (cache pages ≤ working-set pages / 4), and
2. `oracle_miss` shows the largest **headroom** = \((none\_time - oracle\_time)/none\_time\),
   measured first in the virtual-time simulator (M3) and confirmed in the real runtime.

If `oracle_miss` headroom is **< 10%** in every configuration with a cache much smaller
than the working set, the run enters the **large-database branch** (M4): build a database
of `min(8 GB, 40% free disk)` (≥ 2 GB), regenerate traces and models, and recompute. If
headroom is still < 10% everywhere, set **NO_HEADROOM** and let the claim mapping report
the bound instead of a speedup.

## Splits

Two held-out splits, split by **query execution** (never by individual reads):

- **Held-out query parameters** — train on some parameter sets, test on unseen parameters
  of the same templates.
- **Held-out query templates** — whole templates excluded from training and tuning.

## Training and tuning budget

- Tuning only on development seeds **101–110**. Never 901–903.
- Final protocol runs **once** on fresh seeds **911–915**, after code/configs are frozen.
- Model A and Model B receive the **same** hyperparameter-search budget.
- No cherry-picking: every configuration tried is reported, including failures.

## Claim mapping (implemented in the report generator, not written by hand)

The headline sentence is selected mechanically from the H1–H5 outcomes:

- H1 only: "Semantic features improve offline page prediction, but this did not translate
  into a runtime benefit."
- H1 and H2 supported, H3/H4 not: "A neural model extracts more value from semantic
  context than simple tables, but inference cost outweighs it at these latencies."
- H1 supported, H2 not: "Semantic context helps, but a cheap context-keyed table captures
  it, so a neural network is not required."
- H3 supported, H4 not: "Semantic prefetching beats no prefetching, but does not beat
  simpler prefetchers."
- H1–H4 supported: "Model B improves runtime performance over all tested baselines under
  the stated conditions."
- No headroom: "Under the tested conditions the achievable gain from any prefetcher is
  bounded by X% (oracle), so no model can show a meaningful speedup here."

No claim stronger than the data supports may be written.

## Oracle definition (upper bound)

`oracle_miss` prefetches the next **K would-be-miss** pages from the `none` run of the same
workload/cache, skipping pages already cached, in flight, or already demanded. K is
configurable (1, 2, 4, 8). `oracle_next` is the old trace-next oracle, kept as a naive
control. The **headroom bound is `oracle_miss`**, not `oracle_next`.

## Analysis and exclusions

All completed matched queries with a shared seed and query id are included. Malformed rows,
failed reads, non-clean worker shutdowns and missing result files invalidate the run and are
reported rather than silently dropped. Primary numbers come from the virtual-time simulator
(M3); the real runtime validates the primary configuration on fewer seeds.

## Amendments

### 2026-10-15 — large-database and NO_HEADROOM branch

The small development database had no cache cell satisfying cache pages <= observed
working-set pages / 4. The preregistered large-database branch was therefore executed on
an 8.62 GB database. To make generation tractable while expanding workload-relevant leaf
pages, each `order_items` row contains a deterministic 900-byte payload. Large-database
training traces store `dbstat` metadata only for pages observed in that trace; the same
frozen sparse map is bundled with the trained models and used at runtime.

The large-database shared-device grid produced less than 10% `oracle_miss` headroom in
every eligible cell, and its best cell was confirmed in the real runtime. This triggers
`NO_HEADROOM`. Under this branch, M5/M6 neural runtime optimization stops. The final
runtime confirmation uses fresh seeds 911-915 for `none`, `null`, and `oracle_miss` at the
best grid cell. H3/H4 are not supported if the paired oracle-vs-none interval remains
strictly slower; Holm correction is then not applicable because no family of final neural
runtime comparisons is formed. If the final oracle result unexpectedly shows at least
10% headroom, runtime tuning and the full declared strategy comparison must reopen.

Offline H1/H2/H5 evaluation remains unchanged and uses final model seeds 911-915.
