# Pagewise — SQLite learned page-prefetch research

Pagewise investigates whether application-level database semantics help an MLP predict future SQLite database pages better than an otherwise equivalent history-only MLP, and whether any prediction improvement survives runtime overhead.

**Research question:** Do table/index identity, page type, B-tree depth, and query-plan context improve learned page-prefetch prediction for irregular database workloads, and can the resulting prefetches reduce query latency under a controlled cache budget?

## Current repository contents

- `src/`: React/Vite research dashboard. It loads measured backend results from `public/pagewise-dashboard.json`; browser-only replay and calculator controls remain explicitly labelled illustrative fixtures.
- `backend/pagewise/`: new Python backend with database generation, query workload generation, APSW VFS tracing, page metadata, feature generation, PyTorch MLP rankers, baseline evaluation, a bounded asynchronous prefetch cache, repeated-run experiments, and an optional dashboard for measured output files.
- `backend/tests/`: automated tests for database/workload generation, trace parsing, ranker training/evaluation, cache accounting, and an APSW-gated VFS integration test.

## Important experimental boundaries

- VFS `xRead` events are not a complete record of every logical SQLite page reference. SQLite's own page cache can satisfy reads before they reach the VFS.
- A VFS read request does not prove that physical disk I/O occurred; Windows or the underlying OS may have served it from cache.
- `run-prefetch` uses a bounded cache below SQLite's pager and an asynchronous worker for a static read-only database. The main experiment is not a general-purpose production database patch and does not support concurrent database updates as a research assumption.
- `--miss-delay-ms` creates an explicitly **latency-injected** experiment. Those numbers must never be presented as measured NVMe, SATA, HDD, or general OS results.
- No performance gain is assumed. A fixed or Markov baseline may equal or beat an MLP, and semantics may add no useful signal.

## Backend quick start (Windows)

Use Python 3.12 if available. Open PowerShell in the `backend` folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pagewise.cli check-env
```

Read `backend/README.md` for the full generate → trace → train → evaluate → prefetch experiment.

## Run the existing React dashboard

From the repository root:

```powershell
npm ci
npm run dev
```

The React dashboard loads the measured backend snapshot in `public/pagewise-dashboard.json` when present, including offline predictor metrics, final runtime suites, latency series, and trace samples. Browser-only replay and calculator controls remain illustrative and must not be cited as measured results. For raw generated artifacts, the optional Streamlit measured-results viewer is documented in `backend/README.md`.

## Testing

```powershell
cd backend
python -m pip install -e ".[test]"
pytest -q
```

The VFS integration test is skipped when APSW is unavailable. Run `pagewise check-env` to verify that APSW, PyTorch, and `dbstat` are available before performing the real trace experiment.

## Recommended first backend check

After extracting the project, open PowerShell in `backend`, create/activate a Python 3.12 virtual environment, install `-e ".[test]"`, and run `python -m pagewise.cli check-env`. Then run `pytest -q` and `./scripts/run_smoke.ps1 -Queries 150`. The VFS test must pass on your experiment machine before treating any page trace or prefetch result as validated.
