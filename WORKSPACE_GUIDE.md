# Pagewise interactive workspace

From the repository root, run `npm run build` followed by `npm run preview:local`.
Open **http://localhost:4173/?view=workspace** and keep the terminal running.
For development, use `npm run dev:local` and the Vite URL printed by the launcher.

## Simple controls, visible workspace

The database explorer, SQL editor, live terminal and all output tabs are visible on the main workspace page. Choose an example query, click **Run query**, and watch the actual Python output. Click **Compare models** to run the same SQL with the default comparison methods.

Only optional controls are collapsed under **More options**: query values, model, repetition count, row limit and comparison model choices. The rest of the application navigation remains unchanged.

## Faculty walkthrough

1. Click **Create small demo**. This invokes the existing seven-stage CLI pipeline: generate SQLite, trace actual queries, build sequences, train A/B models, evaluate, benchmark and export. The terminal shows captured Python stdout/stderr. Every build has a new run ID and database file.
2. In **Database**, select a table. Browse its actual schema, columns, primary keys, indexes, page size/count and up to 25 rows. **Download SQLite file** downloads the generated database.
3. Load an existing workload query, or enter read-only SQL. Supply `?` parameters as a JSON array. Choose a prediction policy, repetition count and result-row limit. Click **Run query**.
4. **SQL results** shows returned rows and the actual `EXPLAIN QUERY PLAN`. The browser previews up to 100 rows; CSV contains rows up to the configured cap. Long browser cells use bounded previews with original lengths; BLOB values show byte length and a 64-byte hex preview.
5. **Page reads** shows the latest persisted APSW VFS `xRead` events: query ID, page range, bytes, duration, cache state and plan class. It refreshes during execution; events can arrive in batches. Fast queries can finish before the next poll.
6. **Predictions** shows actual background jobs, observed history, ranked candidates, scores, queue/inference times and runtime actions. Empty output means no prediction jobs were recorded. Ranking scores are not measured correctness percentages.
7. **Prefetch monitor** shows real demand/cache/prediction/prefetch counters and recorded uses. A useful entry was fetched early and later served from the Pagewise cache. Correct predictions alone do not guarantee useful or faster execution.
8. Choose **Models to compare**, then click **Compare models**. Each policy runs the same SQL, parameters, result cap and repetitions with a fresh Pagewise cache. Order is randomized and recorded. Results report measured mean/median query time and row counts, without assuming prefetch wins.
9. **Files** downloads real rows, timing CSVs, traces, predictions, counters, plans, requests, benchmark CSV and combined JSON. Reopen earlier work from **Local run**, including after restarting the service.

## Separate database/model scopes

**Small development demo** creates and trains its own database/models. Model B is its semantic model. Standard hybrid is the existing Markov-2/Model-B policy. The 8.62 GB checkpoint is never applied to this unrelated small database.

**Research database** explicitly selects the original 8.62 GB snapshot and frozen selected neural hybrid. Matching local checkpoint, source and database/metadata identities are verified. Ignored research artifacts are required. All new SQL outputs go into new local-lab directories; frozen results and committed exports are preserved. Custom SQL does not constitute a new held-out accuracy evaluation.

The existing guided demo and research pages remain. The runtime table previously removed from those pages stays removed; the workspace has a separate SQL benchmark view.

## What is measured

- `backend/workspace.py` adapts the SQL editor to existing `TraceRecorder`, `TraceVFS`, `PredictiveRuntime` and models. It does not implement a replacement predictor or storage engine.
- Connections open read-only, disable mmap and enable `query_only`. An authorizer rejects writes, attachment, unsafe file functions and configuration statements. Only one statement returning rows is accepted. Preparation and runtime errors are shown.
- SQL has a 10-second progress-handler limit per execution, a 1–5000 result-row cap and a 1 MB SQLite value-size limit. Capped results are marked. A row cap does not bound the work needed to produce those rows.
- SQLite cache is 128 KiB and is released between repetitions. Pagewise cache is 128 pages for small runs, 512 for research. The selected checkpoint retains its frozen scheduling settings. Every policy starts a new Pagewise cache; OS caches are not reset.
- Timings include SQL execution and row fetching/serialization up to the result cap. Model loading, writing output files and draining workers are outside the query timing. Diagnostic tracing/monitoring adds overhead. No artificial delay is inserted.
- VFS reads can hit the OS cache; they do not prove physical disk I/O. Interactive benchmarks do not replace the frozen research study or establish general speedup.
- Queries without `ORDER BY`, or using nondeterministic functions, can return different row streams. Results record whether row order/values matched across executions.
- Live tabs show bounded recent samples. Complete persisted JSONL files remain downloadable. Cancelled runs may leave partial files and their last counters; the job status identifies cancellation.

## Generated files

Each interactive job uses `backend/runs/local-lab/<job-id>/`:

```text
request.json
state.json
workspace-result.json
benchmark.csv
model/                         # run-local metadata/model copies
<policy>/plan.json
<policy>/trace.jsonl
<policy>/predictions.jsonl      # when jobs were recorded
<policy>/prefetch_uses.jsonl
<policy>/runtime_stats.json
<policy>/query-result.json
<policy>/query-results.csv
<policy>/query-timings.csv
```

`backend/runs/local-lab/<job-id>.log` captures process output. Generated databases, models and runs stay ignored by Git; source, tests and guides remain trackable.

## Local API

- `GET /api/workspace/databases`: generated databases, research option, existing query templates and policies.
- `GET /api/workspace/schema?database=<id>&table=<optional-table>`: schema and bounded table preview.
- `POST /api/jobs` with `preset: "workspace"`, database, SQL, params, policies, repeats and row limit: asynchronous SQL/benchmark job using the existing queue and cancellation.
- `GET /api/workspace/run?id=<job-id>&policy=<optional-policy>`: live snapshots, recent trace/prediction/use samples, artifacts and completed result.
- `GET /api/download?path=<artifact-path>`: real file download.

Only known generated databases and the explicit matching research snapshot are selectable. Jobs run one at a time and the API remains on loopback.
