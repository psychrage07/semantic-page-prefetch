# Pagewise v2 — FINAL SUMMARY (session handoff)

This session ran autonomously through **M0, M1, and the core of M2**. M3–M9 were not
started. No git repository exists, so nothing was committed (see `DECISIONS.md` D1).
The full test suite passes (**46 passed**) and `FROZEN_MANIFEST.json` verifies
(**1338 files unchanged**).

## What was done

### M0 Housekeeping — DONE
- `FROZEN_MANIFEST.json`: SHA-256 of 1338 files under the two frozen dirs; verified
  unchanged at the end by `scripts/verify_frozen.py`.
- `FINAL_EVALUATION_PROTOCOL_V2.md` (pre-registered before any v2 tuning), `PROGRESS.md`,
  `DECISIONS.md` (D1–D9).

### M1 Fix the oracle — DONE
- Implemented **`oracle_miss`** (prefetch the next K *would-be-miss* pages from a `none`
  reference run) and relabelled the old trace-next oracle as **`oracle_next`**.
- Reference recording added (`demand_reference.json` via `record_demand_reference`).
- Root-cause fix: prefetch I/O ran on a single thread and could never stay ahead.
  Added a configurable **prefetch worker pool** (`--prefetch-workers`, protocol uses 4),
  identical across strategies; the shared device's lock still serializes under `shared`.
- Sanity (independent model, medium DB, cache 32, K=4, 4 workers): oracle_miss Δ vs none
  = −11.2 / −17.3 / −28.5 / −65.7 ms/query at 0.5/1.0/2.0/5.0 ms; usefulness 70–75 %.
  **(a) (b) (c) pass.** Under `shared` the oracle still loses (+4.6 … +10.5 ms/query):
  genuine device saturation, not a harness bug.
- `tests/test_controls_null_oracle.py` + `tests/test_oracle_miss_integration.py`.
- Findings: `runs/m1-sanity/M1_FINDINGS.md`.

### M2 Offline prediction study — CORE DONE, ablation NOT RUN
- `pagewise/offline.py`: `build_splits` (parameter-bucket split and whole-template split,
  both by execution), `markov_ctx` (key = current page + template + plan class + depth),
  `model_b_table` (Model B top-k per same key), paired bootstrap (10,000 resamples),
  H1/H2/H5-offline outcomes, `run_ablation`.
- CLI: `offline-study`, `offline-ablation`.
- Ran on `runs/experiment-latest/trace` (300 executions, 5 model seeds, 12 epochs):
  results in `runs/m2-offline/offline_accuracy.json` / `.csv`.

**M2 results (mean top-1 %, params split):**

| policy | top-1 |
|---|---|
| markov1 | 56.78 |
| markov2 | 59.93 |
| markov_ctx | 58.78 |
| model_a | 22.15 |
| model_b | 34.66 |
| model_b_table | 31.84 |

- **H1: supported** (Model B − Model A = +12.52, CI [11.10, 13.96]).
- **H2: not supported** (Model B vs markov_ctx = −24.12, CI [−28.01, −20.20];
  vs markov2 = −25.26, CI [−29.25, −21.30]).
- **H5-offline (H1 on held-out templates): inconclusive** (Δ = +0.13, CI [−0.13, +0.41]);
  template-split markov_ctx coverage = 0 %, so tables fall back to markov1 there.
- Headline mapping implied so far (offline only): **H1 supported, H2 not** →
  "Semantic context helps, but a cheap context-keyed table captures it, so a neural
  network is not required." (Final sentence must come from `hypothesis_report.py` after M7.)

## What was NOT done (with reasons)

| Milestone | Status | Reason |
|---|---|---|
| M2 Model B feature ablation | NOT RUN | `offline-ablation` was interrupted by the user; `runs/m2-offline/ablation/` is empty. Re-runnable: see below. |
| M3 Virtual-time simulator | NOT STARTED | — |
| M4 Headroom decision | NOT STARTED | Depends on M3. |
| M5 Runtime pipeline repair | NOT STARTED | Gated on M4 headroom. |
| M6 Cheaper Model B | NOT STARTED | Gated on M4. |
| M7 Tuning + final protocol | NOT STARTED | Needs frozen code/config + seeds 911–915. |
| M8 Report + paper | NOT STARTED | Needs M7 results. |
| M9 Dashboard + demo | NOT STARTED | — |

Also not done: `hypothesis_report.py` (does not exist yet), headroom heatmap, break-even
plot, RAMMap cache control, large-database branch.

## Resume instructions

From `backend/`:

```bash
# finish M2 ablation (≈8–10 min, 3 seeds)
python -m pagewise.cli offline-ablation \
  --sequences runs/experiment-latest/trace/sequences.jsonl \
  --workload runs/experiment-latest/trace/workload.jsonl \
  --page-map runs/experiment-latest/trace/page_map.json \
  --out runs/m2-offline/ablation --split params --seeds 7,17,27 --epochs 12

python -m pytest -q          # expect 46 passed
python scripts/verify_frozen.py   # expect FROZEN OK
```

Then start **M3** (`backend/pagewise/sim.py`, virtual-time discrete-event simulator).

## Known limitations / caveats

- All delay labels are wrong by the measured sleep error (0.2 ms → ~0.56 ms,
  1.0 ms → ~1.54 ms); every report must print this and caveat earlier results.
- `oracle_miss` headroom is demonstrated only under the **independent** model so far;
  under `shared` it loses. M3/M4 must decide whether any real configuration has ≥10 %
  headroom.
- M2 template-split results are near chance and markov_ctx coverage is 0 % there.
- No git; work is left in the working tree.
