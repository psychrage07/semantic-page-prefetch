# Pagewise Python backend

This backend is the experimental engine behind the Pagewise SQLite prefetching research. It intentionally starts with a development-size dataset so the end-to-end data/model path can be checked before attempting a multi-gigabyte database.

The React dashboard's **Live demonstration** tab connects to this backend through `local_lab.py`. From the repository root, `npm run dev:local` starts both the loopback API and Vite frontend. Its command form reflects every Pagewise CLI option and includes the research scripts; the one-click development pipeline runs generate → trace → build sequences → train → evaluate → run suite → dashboard export. Generated files live in `backend/runs/local-lab/` and can be opened from the browser. Use the CLI examples below for large or protocol-frozen experiments.

## 1. Install and check the environment

Use Python 3.11 or 3.12 with available PyTorch and APSW wheels. From PowerShell in this folder, use `python` rather than the optional Windows `py` launcher (which may not be installed):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pagewise.cli check-env
pytest -q
```

The environment command reports Python, SQLite, APSW, PyTorch, CUDA availability, and `dbstat` support. A successful import is not enough: the APSW VFS integration test must also pass on the machine that will be used for measurements.

## 2. Generate a development database

From `backend/`:

```powershell
python -m pagewise.cli generate-db `
  --db ..\data\pagewise-dev.sqlite `
  --customers 10000 --products 1000 --orders 50000 `
  --avg-items 3 --graph-nodes 5000 --edges-per-node 3 --seed 17
```

The generator creates customers, categories, products, orders, order items, indexes, and graph edges. The defaults are a development profile, not a promise of a 5–20 GB database. Scale rows only after the pipeline is verified and check disk space first. Use `--overwrite` only when you intentionally want to replace an existing database.

## 3. Trace actual SQLite VFS read requests

```powershell
python -m pagewise.cli trace `
  --db ..\data\pagewise-dev.sqlite `
  --out ..\runs\trace-001 `
  --queries 500 --seed 1234 --cache-kib 512
```

Outputs include:

- `trace.jsonl`: main database VFS `xRead` requests with byte range, page range, timing, result, query ID, query template, and query-plan context.
- `page_map.json`: `dbstat` mapping from page IDs to table/index, B-tree path, page type, and depth.
- `plans.json`, `workload.jsonl`, `query_results.csv`, and `run_metadata.json`.

The connection disables SQLite memory-mapped I/O and shrinks SQLite's pager cache between query executions to expose more VFS requests under a controlled setup. This is an experimental choice, not a statement that every production SQLite application uses those settings.

The trace is **not** a complete logical page-reference trace and a VFS `xRead` does not prove physical disk I/O. Plan context is derived from the same SQLite runtime where APSW is available. Page metadata first tries the Python `sqlite3` build and then falls back to APSW's SQLite build; `check-env` reports each provider separately. If neither exposes `dbstat`, the tracer stops with an actionable error instead of silently inventing page labels.

## 4. Build per-query page sequences

```powershell
python -m pagewise.cli build-sequences `
  --trace ..\runs\trace-001\trace.jsonl `
  --page-map ..\runs\trace-001\page_map.json `
  --out ..\runs\trace-001\sequences.jsonl
```

This expands each observed VFS read byte range into the database pages it overlaps and groups those pages by query execution. Events without an associated query ID, errors, and non-main-database files are excluded. This reflects observed VFS read requests, not every page hit in SQLite's internal cache.

## 5. Train matched Model A and Model B

```powershell
python -m pagewise.cli train `
  --sequences ..\runs\trace-001\sequences.jsonl `
  --page-map ..\runs\trace-001\page_map.json `
  --out ..\models\run-001 `
  --seed 7 --epochs 20 --candidates 32 --hidden 128
```

Artifacts include `model_a.pt`, `model_b.pt`, `candidate_index.json`, train/validation/test execution splits, and `training_report.json`.

- **Model A:** Last eight observed page IDs, page deltas, and candidate-page features.
- **Model B:** Same base features plus candidate page type/depth/object metadata, query-plan class, template ID, and workload family.
- Both use the same MLP shape and training budget. The model candidate dictionary is built from training executions only.
- Query executions are split as groups to avoid mixing individual reads from the same execution across train and test.
- The candidate ranker has a deliberate limit: it can only select pages made available by its candidate generator. Candidate recall is reported.

To train additional Model B semantic ablation checkpoints on the same split, add named ablation specs:

```powershell
python -m pagewise.cli train `
  --sequences ..\runs\trace-001\sequences.jsonl `
  --page-map ..\runs\trace-001\page_map.json `
  --out ..\models\run-001-ablations `
  --seed 7 --epochs 20 --candidates 32 --hidden 128 `
  --semantic-ablations "no_page_metadata=page_metadata;no_query_context=query_context;no_object=object;no_position=position"
```

This writes checkpoints such as `model_b_retrained_no_object.pt` and records their disabled semantic groups in `training_report.json`.

For a training-seed sensitivity check with a fixed split:

```powershell
python -m pagewise.cli train-sweep `
  --sequences ..\runs\trace-001\sequences.jsonl `
  --page-map ..\runs\trace-001\page_map.json `
  --out ..\models\sweep-001 `
  --seeds 7,17,27 --split-seed 7 --epochs 15
```

This runs several independent training initializations. Add `--semantic-ablations "no_page_metadata=page_metadata;no_query_context=query_context"` to train and evaluate retrained ablation checkpoints for every seed. The resulting `training_sweep_report.json` includes aggregate policy metrics and `delta_vs_model_b` tables. With only a few seeds, intervals remain coarse and describe training-seed variation on a fixed split rather than generalization to every possible database workload.

## 6. Evaluate held-out next-page predictions

```powershell
python -m pagewise.cli evaluate `
  --sequences ..\models\run-001\test_sequences.jsonl `
  --page-map ..\runs\trace-001\page_map.json `
  --candidate-index ..\models\run-001\candidate_index.json `
  --model-a ..\models\run-001\model_a.pt `
  --model-b ..\models\run-001\model_b.pt `
  --out ..\results\prediction-eval-001
```

This reports top-1/top-k accuracy, candidate recall, and prediction logs for history-only and semantic models. No-prefetch is a runtime policy, not a predictor, so it is not assigned an invented zero prediction-accuracy score. The oracle is a perfect-future reference only, not a deployable algorithm.

To run inference-time semantic ablations for Model B, add named ablation specs:

```powershell
python -m pagewise.cli evaluate `
  --sequences ..\models\run-001\test_sequences.jsonl `
  --page-map ..\runs\trace-001\page_map.json `
  --candidate-index ..\models\run-001\candidate_index.json `
  --model-a ..\models\run-001\model_a.pt `
  --model-b ..\models\run-001\model_b.pt `
  --out ..\results\prediction-ablation-001 `
  --semantic-ablations "no_page_metadata=page_metadata;no_query_context=query_context;no_object=object;no_position=position"
```

Supported semantic groups are `page_type`, `depth`, `object`, `plan`, `template`, `workload`, and `position`; aliases are `page_metadata` and `query_context`. These ablations zero feature groups at evaluation time for diagnosis. They do not replace retraining-based ablations for final claims.

To evaluate retrained ablation checkpoints, add `--retrained-ablation-dir ..\models\run-001-ablations`. The output will include policies named `model_b_retrained_<label>` in addition to any inference-time `model_b_ablate_<label>` policies.

## 7. Run actual SQLite queries with the bounded VFS cache

First train the models as above. Then run a different query-parameter seed to evaluate on workload parameters not used for trace training:

```powershell
python -m pagewise.cli run-prefetch `
  --db ..\data\pagewise-dev.sqlite `
  --model-dir ..\models\run-001 `
  --strategy model_b `
  --out ..\runs\runtime-model-b `
  --queries 250 --seed 9876 --cache-kib 512 `
  --cache-pages 256 --degree 2 --threshold 0.0
```

Supported strategies are `none`, `sequential`, `markov1`, `markov2`, `model_a`, and `model_b`. Each uses the same bounded LRU cache and query workload; `none` retains the custom cache but issues no predictions. For context, `trace` without a prefetch config gives a stock/no-custom-cache comparison. That stock baseline differs in cache layering and must be described as such.

To run a specific checkpoint, such as a retrained ablation model, keep the appropriate neural strategy and pass `--model-path`:

```powershell
python -m pagewise.cli run-prefetch `
  --db ..\data\pagewise-dev.sqlite `
  --model-dir ..\models\run-001-ablations `
  --model-path ..\models\run-001-ablations\model_b_retrained_no_page_metadata.pt `
  --strategy model_b `
  --out ..\runs\runtime-model-b-no-page-metadata `
  --queries 250 --seed 9876 --cache-kib 512 `
  --cache-pages 256 --degree 2
```

The runtime stats record both `model_path` and any `disabled_semantic_groups` stored in the checkpoint.

Scheduling controls:

- `--queue-size`: maximum queued prediction jobs for the asynchronous worker.
- `--predict-on all|miss|hit`: schedule after every observed page, only custom-cache misses, or only custom-cache hits.
- `--predict-every N`: schedule at most every Nth eligible observed page.

The `hybrid` strategy uses a training-derived Markov-2 transition whenever an exact second-order history is known, and calls semantic Model B only for unseen second-order contexts. It is a separate policy that keeps all existing baselines intact; it does not use held-out test outcomes to choose its gate.

Runtime artifacts include `runtime_stats.json`, `predictions.jsonl`, `trace.jsonl`, query timings, and cache statistics. The model runs in a background worker; if it cannot keep up, queue drops, queued work, coalesced requests, candidate-pruning counts, and prediction wait time are measured. `runtime_stats.json` also records worker drain and shutdown duration, while `run_metadata.json` records end-to-end wall time after queued prefetch work has completed. Neural runtime scoring skips candidate pages that are already cached or pending because those pages cannot produce a useful additional prefetch. The prefetcher is limited to static, read-only SQLite data for the main experiment.

### Latency-injected mode

To experiment with a controlled delay at cache misses, add `--miss-delay-ms 1.0` (for example). The worker waits for the same configured delay before each simulated prefetch read; demand misses in the custom cache also incur delay. The stock baseline adds delay per main-file VFS read range. Because the injection units differ slightly for multi-page requests, treat this as an emulated experiment and inspect the trace. Never describe these results as actual hardware latency or a real NVMe/HDD measurement.

## 8. Run repeated policy comparisons

```powershell
python -m pagewise.cli run-suite `
  --db ..\data\pagewise-dev.sqlite `
  --model-dir ..\models\run-001 `
  --out ..\results\suite-real `
  --queries 250 --seeds 101,202,303 `
  --strategies stock,none,sequential,markov1,markov2,model_a,model_b,hybrid `
  --cache-kib 512 --cache-pages 256 --degree 2
```

The query parameter seed is paired across strategies within each trial, and strategy ordering rotates between trials. On a Windows machine, the OS cache may remain warm between runs; this is recorded as a real-storage run without injected delay, not a guaranteed cold-cache run. Use a documented RAMMap/cache-reset protocol only when you are prepared to control and report it.

Outputs:

- `suite_runs.json`: full run metadata
- `suite_summary.json` and `suite_summary.csv`: run-level p50/p95 aggregation, cache/prefetch counts, scheduler-pressure counts, and bootstrap intervals over run p50 values
- `suite_summary.json` also includes `paired_query_comparisons`, which compares matched query IDs for paired seeds against `stock` and/or `none` baselines when the corresponding `query_results.csv` files are available

Repeat with `--miss-delay-ms 1.0 --out ..\results\suite-latency-1ms` to keep latency-injected results separate.

If a machine or terminal interrupts after per-policy run directories were written, recover the completed suite without rerunning it:

```powershell
python -m pagewise.cli recover-suite --out ..\results\suite-real
```

## 9. Runtime sensitivity sweeps

Use `runtime-sweep` to run a small grid of cache and scheduler settings while retaining normal `run-suite` artifacts for every combination:

```powershell
python -m pagewise.cli runtime-sweep `
  --db ..\data\pagewise-dev.sqlite `
  --model-dir ..\models\run-001 `
  --out ..\results\runtime-sweep-001 `
  --queries 150 --seeds 101,202 `
  --strategies none,model_a,model_b --baseline none `
  --cache-kib 512 --cache-pages 128,256 `
  --degrees 1,2 --queue-sizes 64 `
  --predict-on all,miss --predict-every 1 `
  --miss-delay-ms 1.0
```

Outputs include `runtime_sensitivity_report.json`, `runtime_sensitivity_ranking.csv`, and one normal suite directory per parameter combination. Rank rows use paired query deltas against the selected baseline. Treat this as a tuning and diagnosis tool; final claims still require a predeclared full experiment.

## 10. Concise artifact reports

Use `report` to turn large JSON outputs into compact Markdown tables:

```powershell
python -m pagewise.cli report `
  --input ..\results\suite-real\suite_summary.json `
  --out ..\results\suite-real\suite_report.md

python -m pagewise.cli report `
  --input ..\models\sweep-001\training_sweep_report.json `
  --out ..\models\sweep-001\training_sweep_report.md

python -m pagewise.cli report `
  --input ..\results\runtime-sweep-001\runtime_sensitivity_report.json `
  --out ..\results\runtime-sweep-001\runtime_sensitivity_report.md
```

The report command supports suite summaries, training sweeps, and runtime sensitivity reports. It summarizes existing measured artifacts; it does not compute new metrics or fill missing values.

## 11. Measured-results dashboard

The React dashboard in the repository root can load a compact measured-results snapshot from Vite's `public` directory. Export one from a completed experiment run:

```powershell
python -m pagewise.cli dashboard-export `
  --run-dir runs\experiment-latest `
  --out ..\public\pagewise-dashboard.json
```

The dashboard reads `/pagewise-dashboard.json` at startup and shows measured predictor accuracy, candidate recall, suite p50 values, paired runtime deltas, per-query latency series, and capped trace/prediction samples. The replay controls, break-even calculator, and ablation preview remain explicitly labelled as illustrative browser fixtures.

To view actual backend outputs, install the optional dashboard dependency:

```powershell
python -m pip install -r requirements-dashboard.txt
streamlit run pagewise/dashboard.py
```

Upload `suite_summary.json`, `predictor_metrics.json`, `run_metadata.json`, `runtime_stats.json`, and/or `predictions.jsonl` through the sidebar. The page displays only uploaded measurements and does not invent numbers.

## 12. Tests

```powershell
pytest -q
```

The VFS integration test uses `pytest.importorskip("apsw")` and will be skipped if APSW is missing. Do not treat a skipped VFS test as proof that tracing or runtime prefetching works. Before collecting final results, run `check-env` and make sure the VFS integration test passes on the experiment machine.

## Fast Windows smoke run

After installing dependencies and confirming APSW, run this from PowerShell in the `backend` directory (or use the path to the script):

```powershell
.\scripts\run_smoke.ps1 -Queries 150
```

If PowerShell blocks the script because it is not digitally signed, run it with a process-scoped bypass instead of changing machine policy:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_smoke.ps1 -Queries 150
```

The script generates a small database, verifies actual VFS tracing by collecting a workload, builds query-level page sequences, trains both MLPs, and evaluates held-out predictions. It stores everything under a timestamped `backend/runs/smoke-*` folder. This is a pipeline smoke test, not the final performance experiment. The script deliberately stops if APSW or PyTorch is unavailable.

The default **Test trained hybrid** mode uses `live_hybrid.py` and `live_evaluation.py` to execute the frozen semantic checkpoint on the matching local database. Its nine stages include fresh next-page evaluation, separate from timed runtime. `/api/selected-results` serves the latest completed evaluated run to the frontend. Evaluation counts all eligible transitions, including targets missing from the candidate set; no fitting or tuning occurs. Fresh workload values may overlap training, so these scores remain separate from the final held-out study.

The **Interactive workspace** at `/?view=workspace` exposes read-only SQL, actual schema/row browsing, live VFS traces, worker predictions/counters and measured SQL benchmarks through `workspace.py`. It reuses the existing tracing, models and runtime, the local job queue and cancellation. Every execution saves isolated artifacts. See [the workspace guide](../WORKSPACE_GUIDE.md) for APIs, measurement scope and the browser walkthrough.
