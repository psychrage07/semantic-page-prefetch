# Pagewise completion roadmap

This is the execution order for completing the project. `backend/PROGRESS.md` records
the live state and evidence for each milestone. A milestone is complete only when its
listed exit criteria are satisfied and its artifacts are present in the current checkout.

**Status (2026-10-15):** the roadmap is complete through final evaluation,
report/dashboard export, and repository verification. See `backend/PROGRESS.md` for
measured outcomes.

## 1. Reproducible baseline

- Reconcile status documents with the current source tree and available artifacts.
- Create an isolated backend environment and verify APSW, PyTorch, `dbstat`, tests, and
  the frontend production build.
- Generate a new canonical baseline because the historical frozen experiment directories
  referenced by `FROZEN_MANIFEST.json` are absent from this checkout.

Exit criteria: one documented environment; all tests and the frontend build pass; a
fresh generate -> trace -> sequence -> train -> evaluate -> runtime smoke pipeline is
stored under `backend/runs/` with its configuration and seeds.

## 2. Simulator validation (M3)

- Calibrate `pagewise/sim.py` from fresh runtime measurements.
- Validate simulated total time against at least three real runtime configurations,
  covering `none`, a simple prefetcher, and a neural prefetcher where practical.
- Measure and report achieved sleep delay, queue cost, inference cost, prefetch service
  time, worker drain, and shutdown time.

Exit criteria: simulator error is within 15% for at least three declared configurations,
or the model is repaired and the remaining error is explicitly bounded.

## 3. Headroom decision (M4)

- Run the declared cache-size x delay `oracle_miss` headroom grid under the shared-device
  model, with the independent model retained as a diagnostic.
- Confirm the best eligible cell in the real runtime.
- If all eligible cells have less than 10% headroom, follow the large-database branch
  within the available disk limit. If headroom remains below 10%, record `NO_HEADROOM`.

Exit criteria: a frozen primary configuration or a documented `NO_HEADROOM` result.

## 4. Runtime and cheaper-policy work (M5-M6)

- Profile queue staleness, inference, speculative I/O contention, and worker drain on the
  selected configuration.
- Compare existing cheap policies (`markov2`, `markov_ctx`, `model_b_table`, `hybrid`) and
  implement `model_b_gated` only if the headroom result justifies more optimization.
- Preserve the same cache, worker, candidate, and storage assumptions across strategies.

Exit criteria: every retained optimization has an ablation and cost/benefit result; the
candidate set for final evaluation is frozen.

## 5. Final evaluation (M7)

- Tune only on development seeds 101-110 with equal search budgets for Models A and B.
- Freeze code and configuration before final runs.
- Run the final protocol once on seeds 911-915 for held-out parameters and held-out
  templates, including every declared baseline and unsuccessful policy.
- Evaluate H1-H5 with paired bootstrap intervals and Holm-corrected H3/H4 decisions.

Exit criteria: complete, machine-readable final results; clean worker shutdowns; no
silently dropped or unmatched observations; mechanically generated hypothesis outcomes.

## 6. Report, dashboard, and submission (M8-M9)

- Generate the headline from the hypothesis outcomes rather than writing it by hand.
- Update the paper, charts, dashboard snapshot, limitations, and reproduction commands.
- Clearly label measured, simulated, latency-injected, and illustrative values.
- Run the complete backend test suite and frontend build from a clean environment.

Exit criteria: paper, dashboard, raw final results, and documentation agree; the repository
contains no unsupported speedup claim and is ready for demonstration or submission.
