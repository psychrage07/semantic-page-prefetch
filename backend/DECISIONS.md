# Pagewise v2 — DECISIONS

One line per decided point. Newest first.

- **D1 (M0):** No git repository exists at the project root, so milestone commits are
  skipped; all work is left in the working tree and tracked via `PROGRESS.md`.
  *Reason: `git rev-parse` fails and the task allows committing only "if git is available".*
- **D2 (M0):** The frozen-run manifest uses SHA-256 over every regular file (1338 files)
  in `runs/experiment-latest/` and `runs/diagnostic-audit-001/`, stored in
  `FROZEN_MANIFEST.json`. *Reason: rule 1 requires an explicit tamper check with exact hashes.*
- **D3 (M0):** Primary-metric unit is the **run**, with paired analysis at the query level
  (paired by seed and query id) and the run total as the headline number.
  *Reason: the protocol's primary metric is total workload time; per-query pairing gives the CI.*
- **D4 (M0):** Until M1 lands, the existing `oracle` is treated as a *naive control*
  (`oracle_next`) and is **not** used as the headroom bound. *Reason: it is only 10-15% useful
  because it ignores cache hits, so it under-estimates headroom.*
- **D5 (M0):** All results previously labelled "0.2 ms / 1.0 ms" are recorded with the
  measured sleep error (0.2 -> ~0.56 ms, 1.0 -> ~1.54 ms) and caveated everywhere.
  *Reason: rule 6; the labels do not match the achieved service time.*
- **D6 (M0):** Spin-waits are forbidden in the threaded runtime. *Reason: rule 6 — they hold
  the GIL and starve the worker thread.*
- **D7 (M1):** The oracle bound is `oracle_miss` (next K would-be-miss pages from a `none`
  reference), not the old trace-next oracle (now `oracle_next`). *Reason: the old oracle
  targeted already-cached pages and under-stated headroom by ~an order of magnitude.*
- **D8 (M1):** Prefetch reads run on a configurable thread pool (default 1, protocol uses
  4), identical across strategies. *Reason: single-threaded prefetch I/O made timely
  prefetching impossible even on a parallel device; the shared device's lock preserves
  serialization.*
- **D9 (M1):** The shared-model oracle loss is recorded as device saturation, not a
  harness bug, and is carried into the M4 headroom decision. *Reason: wasted prefetch
  reads add serialized device load that demand must wait behind.*
- **D10 (M2):** The held-out-parameter split groups executions by a hash of their SQL
  parameters into 6 buckets and holds out whole buckets; the held-out-template split
  holds out whole templates. Both split by execution. *Reason: the workload has no
  repeated parameter tuples, so bucket-level separation is the strongest parameter
  holdout available.*
- **D11 (M2):** `markov_ctx` and `model_b_table` fall back to `markov1` on an unseen key,
  and coverage is reported separately. *Reason: isolates the value of the semantic key
  while keeping both table strategies comparable and non-empty on held-out templates.*
- **D12 (M2):** Model A and Model B use the same seed set, epochs, hidden size and
  candidate cap; no per-model extra budget. *Reason: rule 4 fairness.*
