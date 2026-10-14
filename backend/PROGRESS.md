# Pagewise v2 — PROGRESS

Autonomous lead-engineer run. See `DECISIONS.md` for one-line rationale on each call.

## Milestone status

| Milestone | Status | Notes |
|---|---|---|
| M0 Housekeeping | DONE | manifest (1338 files), protocol v2, progress/decisions, baseline 41 tests |
| M1 Fix oracle | DONE | `oracle_miss`, `oracle_next`, prefetch worker pool; sanity passes under independent; findings in `runs/m1-sanity/M1_FINDINGS.md`
| M2 Offline study | DONE | splits, markov_ctx, model_b_table, 5 seeds x 2 splits, CIs, H1/H2/H5-offline, ablation + charts in `runs/m2-offline/` |
| M3 Virtual-time simulator | NOT STARTED | `backend/pagewise/sim.py`, calibration, headroom grid |
| M4 Headroom decision | NOT STARTED | branching: medium DB vs large DB |
| M5 Real-runtime pipeline repair | BLOCKED on M3/M4 | |
| M6 Cheaper Model B | BLOCKED | |
| M7 Tuning + final protocol v2 | BLOCKED | seeds 911-915, frozen configs |
| M8 Report + paper | BLOCKED | |
| M9 Dashboard + demo | BLOCKED | |
| FINAL VERIFICATION | PENDING | tests + frozen hashes + FINAL_SUMMARY.md |

## Next action

Start **M3** (`backend/pagewise/sim.py`, virtual-time discrete-event simulator), then validate
it against the real runtime within 15% on >=3 configurations and produce the oracle_miss
headroom grid heatmap. Findings so far: `runs/m2-offline/M2_FINDINGS.md` and
`runs/m1-sanity/M1_FINDINGS.md`.

## Resume instructions

Everything the run needs lives under `backend/`:

- Frozen inputs: `runs/experiment-latest/`, `runs/diagnostic-audit-001/` — see `FROZEN_MANIFEST.json`.
- Pre-registration: `FINAL_EVALUATION_PROTOCOL_V2.md` (append-only amendments).
- Decisions: `DECISIONS.md`.
- Task 1 controls (already done): `runs/task1-controls/`, `TASK1_FINDINGS.md`.
- Run the suite: `cd backend && python -m pytest -q`.

To resume after interruption: read this file, `DECISIONS.md`, then continue from the
first NOT STARTED / IN PROGRESS milestone. Do not repeat completed work.

## Current blockers / open questions

- No git repository is present, so milestone commits are skipped (see DECISIONS D1).
- The runtime pipeline (deep async queue) is the known bottleneck; M3/M4 decide
  whether any configuration has real headroom before runtime work (M5-M7) proceeds.

## Log

- (M0) Hashed 1338 files under the frozen directories into `FROZEN_MANIFEST.json`.
- (M0) Wrote `FINAL_EVALUATION_PROTOCOL_V2.md`, `PROGRESS.md`, `DECISIONS.md`; baseline 41 tests.
- (M1) Root cause: oracle prefetched already-cached trace pages, and single-threaded
  prefetch I/O could never stay ahead. Fixed with `oracle_miss` (next K would-be-miss
  pages from a `none` reference) and a `--prefetch-workers` pool. Sanity (a)(b)(c) pass
  under the independent model; shared still loses (device saturation). 46 tests pass.
- (M2) Added `pagewise/offline.py` (param-bucket + template splits, `markov_ctx`,
  `model_b_table`, paired bootstrap, H1/H2/H5-offline). Ran 5 seeds x 2 splits on the
  frozen trace: H1 supported, H2 not supported, H5-offline inconclusive. Outputs in
  `runs/m2-offline/`. See `FINAL_SUMMARY.md`.
- (M2) Finished the interrupted ablation: object identity alone is worth 30.7 of 33.5
  top-1 points; removing page metadata costs 10.15 points. Wrote accuracy + ablation
  charts and 7 new tests (suite now 53 passing).
