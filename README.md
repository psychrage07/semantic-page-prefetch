# Pagewise — SQLite learned page-prefetch research

Project completion is tracked in [ROADMAP.md](ROADMAP.md), with live milestone evidence
in [backend/PROGRESS.md](backend/PROGRESS.md).

## Final experimental outcome

The frozen five-seed evaluation is complete. Semantic Model B outperformed the matched
history-only Model A offline, but simple Markov predictors were more accurate. Under the
tested latency-injected shared-device conditions, even a future-aware oracle was slower
than no prefetching, so the protocol reached `NO_HEADROOM` and did not open neural
runtime tuning. See [backend/FINAL_SUMMARY.md](backend/FINAL_SUMMARY.md) for the results,
[backend/FINAL_CONFIG_V2.json](backend/FINAL_CONFIG_V2.json) for the frozen settings, and
[backend/FINAL_SOURCE_MANIFEST_V2.json](backend/FINAL_SOURCE_MANIFEST_V2.json) for the
source snapshot.

The dashboard shows `n/a` for unavailable measurements in a loaded snapshot and
keeps offline prediction accuracy separate from runtime availability. Missing trace
samples and latency series are shown as empty states. Browser fixtures are used
when no backend snapshot is loaded.

`FINAL_SOURCE_MANIFEST_V2.json` records the source used for evaluation. Later
dashboard availability and `.env` ignore fixes are post-evaluation changes to
`src/App.tsx` and `.gitignore`; the historical source manifest is retained.

Pagewise investigates whether application-level database semantics help an MLP predict future SQLite database pages better than an otherwise equivalent history-only MLP, and whether any prediction improvement survives runtime overhead.

**Research question:** Do table/index identity, page type, B-tree depth, and query-plan context improve learned page-prefetch prediction for irregular database workloads, and can the resulting prefetches reduce query latency under a controlled cache budget?

## Current repository contents

- `src/`: React/Vite research dashboard. It loads measured backend results from `public/pagewise-dashboard.json`, and its Local experiments workspace starts real backend runs through the loopback-only bridge. The overview calculator and its preview controls remain illustrative fixtures.
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
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pagewise.cli check-env
```

Read `backend/README.md` for the full generate → trace → train → evaluate → prefetch experiment.

## Run the local frontend and backend

From the repository root:

```powershell
npm ci
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
cd ..
npm run dev:local
```

Open the Vite URL shown in the terminal. The main walkthrough is **Project overview → How it works → Live demonstration → Research results**. See [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md) for a short faculty demonstration script. The live demo generates a small database, traces real reads, trains and evaluates Model A and Model B, runs a policy suite, and automatically displays its measured results. The final selected neural hybrid is reported separately in Research results. Runs and checkpoints stay under ignored `backend/runs/local-lab/`, so a fresh clone can train its own development models without downloading the historical database.

The default live mode is now **Test trained hybrid**: it loads the existing selected checkpoint and compares fresh runtime with no prefetch and Markov-2 on the matching local 8.62 GB database. It verifies the frozen checkpoint/source/snapshot and never retrains. A separate evaluation stage scores every eligible next-page transition from fresh baseline traces with the semantic checkpoint and native Markov controls. It calculates query-average Top-1, transition Top-1/Top-5 and candidate recall, with exact history/target examples and semantic metadata. These fresh scores automatically populate the overview, research results, comparison and prediction inspector through `/api/selected-results`. The saved held-out study remains separate; fresh parameter values may overlap training. **Build a small demo** selects the A/B training pipeline described above. The trained-mode button requires the ignored frozen model, original database, metadata and policy artifacts; it reports unavailable inputs rather than substituting a different model.

**Advanced research tools** in the live demo retains the 22 CLI commands, 14 research-script entry points, run logs and artifact browser. Historical dashboards and fixture previews are under the collapsed **Research workspace** navigation.

The local bridge binds only to `127.0.0.1:8765` and executes one job at a time. `npm run dev` still starts the static dashboard alone; the local experiment controls require `npm run dev:local`. If Python is installed outside `backend/.venv`, set `PAGEWISE_PYTHON` to its executable path before starting the local command.

For a local production-build preview, run `npm run build` followed by `npm run preview:local`. This serves the built dashboard and the same local backend bridge together.

The **Interactive workspace** at `/?view=workspace` keeps the real SQL editor, database explorer, live Python terminal, page reads, predictions, prefetch monitor, model comparison and files visible. The main actions are **Run query** and **Compare models**; optional parameters are under **More options**. **Create small demo** uses the existing backend pipeline. The small demo and explicit research database use matching models and separate outputs. See [WORKSPACE_GUIDE.md](WORKSPACE_GUIDE.md) for the walkthrough and measured artifacts.

The advanced historical dashboard loads `public/pagewise-dashboard.json`; its fixture preview controls remain illustrative. The main project overview uses only the measured final-study export. For raw generated artifacts, the optional Streamlit viewer is documented in `backend/README.md`.

The **Research results** tab (`/?view=neural`) displays the separate fresh study from `public/pagewise-neural-improvement.json`, with detailed uncertainty and diagnostics under **Full research evidence**. See [the study roadmap](backend/NEURAL_IMPROVEMENT_PLAN.md) and [measured results](backend/NEURAL_IMPROVEMENT_RESULTS.md). Its download contains the measured study. Local frozen checkpoints and raw traces are ignored research artifacts; the historical research snapshot remains separate.

## Testing

```powershell
cd backend
python -m pip install -e ".[test]"
pytest -q
```

The VFS integration test is skipped when APSW is unavailable. Run `pagewise check-env` to verify that APSW, PyTorch, and `dbstat` are available before performing the real trace experiment.

## Recommended first backend check

After extracting the project, open PowerShell in `backend`, create/activate a Python 3.12 virtual environment, install `-e ".[test]"`, and run `python -m pagewise.cli check-env`. Then run `pytest -q` and `./scripts/run_smoke.ps1 -Queries 150`. The VFS test must pass on your experiment machine before treating any page trace or prefetch result as validated.
