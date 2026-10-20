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

## Amendments

### 2026-10-20 — v2.1 scientific-validity amendment (oracle semantics, ML
### variants, statistics, reporting)

This amendment appends to protocol v2 without modifying any frozen rule above.
All v2 final results and artifacts remain frozen and untouched. New v2.1
outputs are produced under new `runs/v2.1*` directories, and every run attempt
(failed, interrupted, exploratory, included) is appended to an append-only
hash-chained run ledger (`pagewise.runledger`, `run_log.jsonl`).

**A. Oracle semantics and simulator validity.**

1. `oracle_miss` is reclassified as a **future-aware diagnostic policy**. It
   may perform worse than no prefetching because speculative I/O creates
   contention; its measured results are preserved exactly as measured and are
   never forced to satisfy any theoretical property.
2. The correctly defined offline policy is **`oracle_optimal`** (new). It
   knows the full future demand order, respects what is cached or in flight,
   and optimizes admission/degree over the declared closed action family
   `{no-prefetch} ∪ {prefetch-next-would-be-misses(K), K <= degree}`, where
   every action is evaluated under the exact modeled constraints (same device
   kind and channel count, same cache capacity and worker model, same CPU
   cost model). Because the no-prefetch action is a member evaluated on the
   identical model, the invariant cost(oracle) <= cost(none) holds and is
   asserted empirically by tests.
3. Optimality scope: the optimization is exact *within the declared action
   family*. A globally optimal joint scheduling/admission/eviction policy is
   NP-hard and is **not claimed**; oracle_optimal headroom is a **certified
   lower bound** on achievable headroom. Belady-style cache replacement is
   implemented and tested separately (`_BeladyCache`) and is reported as a
   distinct cache-policy evaluation, not merged with the scheduling policy.
4. Added invariants (tested): oracle_optimal with no-prefetch action never
   costs more than the no-prefetch baseline; for fixed inputs, increasing
   positive service time never reduces absolute completion time beyond
   numerical tolerance; the measured *benefit relative to no prefetching* is
   explicitly allowed to be non-monotonic in delay (shared-device contention).
5. CPU accounting: every simulator run reports `cpu_simulated_ms`,
   `cpu_fraction_of_total`, and `amdahl_max_reduction_ms` (Amdahl bound for
   removing all simulated CPU overhead), accumulated separately from device
   holds and waiting time so no double counting is possible. This is a
   harness property and is reported separately from the oracle's latency
   headroom.
6. Storage model: an abstract **k-channel** model (`KChannelStorageDevice`
   in the threaded runtime; generalized `_Device` in the simulator) with k
   concurrent FIFO service slots, evaluated at k = 1, 4, 16. `k = 1`
   reproduces the v2 `shared` serialized channel exactly (the v1 label is
   retained for frozen-config reproducibility); `independent` remains the
   contention-free diagnostic endpoint; `shared` and `independent` are
   both still accepted everywhere. No HDD/NVMe label may be attached to any
   of these without a device-specific calibration on the target hardware;
   until then all concurrency results are labeled *simulated concurrency
   scenarios*.
7. The two simulator-validation configurations that missed the declared 15%
   target are diagnosed, not removed: `pagewise.cli sim-validate-cells`
   reports configuration, error value, and dominant modeled cause per cell,
   and marks cells whose demand reference is missing as `needs_reference`
   (never silently skipped). Proven defects must be fixed on the experiment
   host or the residual error must be documented as a bounded validity range.
8. The v3 `NO_HEADROOM` decision scope is amended: it stands only when the
   correctly defined `oracle_optimal` shows < 10% headroom in **every
   eligible k >= 4 configuration** of the headroom grid. Any eligible
   configuration with >= 10% headroom reopens the runtime comparison for
   that configuration (`pagewise.cli headroom-v21` implements this rule and
   reports `decision`, `reopen_configurations`, and per-cell results).

**B. ML variants, feature availability and fairness.**

1. Model B is preserved as a separate baseline; nothing in its training or
   feature path changed.
2. **Model B2** = Model B + training-derived Markov-2 successor
   probability/count features (6 fixed features documented in
   `pagewise.ml_variants`). The previously discussed output-score bias
   variant, if studied, is reported separately as
   **`model_b_score_bias`** and is never merged with B2.
3. **Model B3** = Model B2 + query-parameter features demonstrably available
   at prediction issue time (the full SQL parameter set is known at query
   start): param-bucket one-hot plus availability flag. Row identifiers from
   index leaves are **not** used because the current VFS demand context does
   not expose index-child observability events; this extension is
   documented as unimplemented rather than silently assumed.
4. Fairness: identical train/validation/test executions, identical
   candidate-generation rules (one common `CandidateIndex`), and identical
   tuning budgets for Model A, Model B, B2 and B3. Candidate indexes,
   transition counts, and priors derive from training executions only; no
   feature uses future page targets, future outcomes, or held-out
   transition statistics. Regression tests enforce train/test leakage
   checks.
5. Both all-read and miss-only training are evaluated. **Miss-only labels**
   use a declared, reproducible reference-cache replay (no-prefetch LRU,
   `MISS_ONLY_REPLAY_CACHE_PAGES = 32`) over training executions; labels
   are never derived from any tested model's own future cache outcomes.
6. Reported separately: candidate-generation recall against the common set
   of held-out transitions; native candidate coverage for Markov-2,
   `markov_ctx`, and the semantic generator; predeclared Top-1 and Top-5;
   **timely coverage** (1-step-lagged Top-K at reference-cache misses — the
   offline analog of "the correct page completed prefetch before the
   demand"); useful-prefetch and wasted-prefetch rates with explicit
   denominators (prefetched pages for rates, demand misses for coverage);
   inference microseconds per prediction and full runtime overhead.
7. Calibration: temperature scaling fit on **validation data only**;
   candidate scores are normalized over the competing candidate set with a
   softmax over temperature-scaled logits. Independent sigmoid outputs are
   explicitly not treated as a categorical distribution (v2 violation
   fixed). Calibration is evaluated separately (ECE, NLL) and the
   *calibrated* estimates feed the break-even gate.
8. The gate uses **expected net benefit** = P(timely useful) × achievable
   demand latency saved − P(waste) × waste cost − contention − inference −
   scheduling costs. Thresholds are fit on development/validation data
   only.
9. A multi-step prediction head or table distillation may be added only
   after the core B2/B3 comparison is correct, and only as another named
   variant (not merged).

**C. Evaluation splits, statistics, and final runs.**

1. Development seeds **101–110** are for tuning only. Architecture,
   feature groups, thresholds, candidate-generation settings, and the
   evaluation configuration are frozen before final testing.
2. H1, H2 and held-out-template comparisons are re-run for Model B2 and B3
   against `markov2` and `markov_ctx` on the held-out parameter and
   held-out template splits (`pagewise.cli offline-study-v21`), with the
   original H1 (Model B vs Model A) preserved in the same family.
3. The statistical family is **declared before evaluation**
   (`V21_FAMILY`): paired comparisons matched by workload seed and query
   id, statistical units clustered within seeds, two-level cluster
   bootstrap, **Holm correction** across the declared family, and raw effect
   sizes/CIs reported alongside corrected decisions with replication units.
   v2's per-observation bootstrap remains available but is superseded for
   v2.1 decisions.
4. Seeds 911–915 were already used for the v2 final study. They may be used
   for a **clearly labeled amended v2.1 paired comparison** only if no
   model or threshold is tuned on their outcomes; they are not fresh
   confirmatory data. Seeds **921–925** are reserved as the new untouched
   confirmatory set.
5. H3/H4 runtime comparisons run only for eligible k-channel
   configurations (cache <= working-set/4) with demonstrated `oracle_optimal`
   headroom. If no eligible configuration has headroom >= 10% everywhere,
   scope of the concluded NO_HEADROOM is limited to the tested grid and
   modeled constraints and must not be read as ruling out every possible
   storage architecture.
6. Every configuration attempted is recorded in the run ledger with source
   hash, seed, outputs, and exclusion reason; poor results are never deleted
   or silently omitted.

**D. Immutable results and reporting.**

1. Frozen result directories and `FROZEN_MANIFEST.json` are not edited;
   its hash chain is re-verified (`scripts/verify_frozen.py`) and any
   missing artifact is reported.
2. v2.1 reports are generated mechanically from verified experiment
   artifacts; every unsupported or inconclusive claim is stated as such.
   Offline prediction improvements, simulated storage results, and physical
   device measurements remain three distinct evidence classes and are never
   conflated: VFS reads and injected delays are never equated with verified
   physical disk I/O.
3. Changes to the oracle definition and the k-channel storage model relative
   to v2 are identified in every v2.1 report.
