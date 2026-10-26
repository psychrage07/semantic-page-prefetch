import { useEffect, useRef, useState } from "react";

type Database = { id: string; label: string; scope: "small" | "research"; bytes: number; policies: string[] };
type Template = { id: string; sql: string; params: unknown[]; family: string };
type Catalog = { databases: Database[]; templates: Template[]; policies: Record<string, string> };
type Table = { name: string; type: string; sql: string; columns: { name: string; type: string; primary_key: boolean }[]; indexes: { name: string; unique: boolean }[] };
type Schema = { path: string; bytes: number; page_count: number; page_size: number; tables: Table[]; preview?: { table: string; columns: string[]; rows: unknown[][] } };
type Job = { id: string; status: string; step: number; total_steps: number; preset?: string; started_at: string; root: string; error?: string };
type Trace = { seq: number; query_id?: string; first_page: number; last_page: number; amount: number; duration_us: number; cache_status: string; source: string; plan_class: string };
type Prediction = { prediction_id: number; query_id: string; history: number[]; inference_ms?: number; queue_wait_ms?: number; ranked_candidates?: { page: number; score: number }[]; selected?: { page: number; status: string }[] };
type QueryResult = { policy: string; label: string; columns: string[]; preview: unknown[][]; mean_ms: number; median_ms: number; template: string; plan: string[]; runtime: Record<string, number | boolean>; measurements: { query_id: string; elapsed_ms: number; rows: number; truncated: boolean; result_sha256: string }[] };
type WorkAccuracy = { available: boolean; reason?: string; eligible_transitions: number; scored_transitions?: number; sampled: boolean; max_transitions_per_query: number; scope: string; policies: Record<string, { queries: number; transitions: number; top1_macro_pct: number; top1_micro_pct: number; top5_micro_pct: number; candidate_recall_pct: number }> };
type Run = { state: { status?: string; phase?: string; policy?: string; repeat?: number; error?: string; trace_events?: number; stats?: Record<string, number | boolean> }; policy?: string; request: { database: string; policies: string[]; sql: string; params: unknown[]; repeats: number; row_limit: number }; trace: Trace[]; predictions: Prediction[]; uses: { page: number; prediction_id: number; demand_query_id: string; lead_time_ms: number }[]; artifacts: { name: string; path: string; bytes: number }[]; result?: { evaluation?: WorkAccuracy; results: QueryResult[]; order: string[]; notes: string[]; database_unchanged: boolean; identical_row_order_and_values: boolean; checkpoint_sha256?: string; request: { scope: string } } };
type Tab = "database" | "results" | "trace" | "predictions" | "prefetch" | "benchmark" | "files";
const tabs: [Tab, string][] = [["database", "Database"], ["results", "Results"], ["trace", "Page reads"], ["predictions", "Predictions"], ["prefetch", "Prefetch"], ["benchmark", "Comparison"], ["files", "Files"]];
const active = (job?: Job | null) => job?.status === "running" || job?.status === "cancelling";
const display = (value: unknown): string => value === null ? "NULL" : typeof value === "object" ? JSON.stringify(value) : String(value);
const fixed = (value: unknown, digits = 2) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "n/a";
const download = (path: string) => `/api/download?path=${encodeURIComponent(path)}`;
async function api<T>(path: string, body?: object): Promise<T> {
  const response = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : { cache: "no-store" });
  const data = await response.json().catch(() => { throw new Error("Start npm run preview:local to connect the Python service."); });
  if (!response.ok) throw new Error(data.error || "Local request failed");
  return data as T;
}

function DataTable({ columns, rows }: { columns: string[]; rows: unknown[][] }) {
  return <div className="workspace-table"><table className="faculty-table"><thead><tr>{columns.map((column, i) => <th key={i}>{column}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i}>{row.map((value, j) => <td key={j} title={display(value)}>{display(value).slice(0, 500)}</td>)}</tr>)}</tbody></table>{rows.length === 0 && <p className="workspace-empty">The query returned no rows.</p>}</div>;
}

export default function InteractiveWorkspace() {
  const [catalog, setCatalog] = useState<Catalog>({ databases: [], templates: [], policies: {} });
  const [database, setDatabase] = useState("");
  const [schema, setSchema] = useState<Schema | null>(null);
  const [table, setTable] = useState("");
  const [sql, setSql] = useState("SELECT order_id, customer_id, total_cents FROM orders ORDER BY order_id LIMIT 25");
  const [params, setParams] = useState("[]");
  const [policy, setPolicy] = useState("model_b");
  const [benchPolicies, setBenchPolicies] = useState<string[]>(["none", "markov2", "model_b"]);
  const [repeats, setRepeats] = useState(3);
  const [rowLimit, setRowLimit] = useState(1000);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [jobId, setJobId] = useState(() => localStorage.getItem("pagewise-workspace-job") || "");
  const [job, setJob] = useState<Job | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [viewPolicy, setViewPolicy] = useState("");
  const [tab, setTab] = useState<Tab>("database");
  const [error, setError] = useState("");
  const [online, setOnline] = useState(false);
  const [busy, setBusy] = useState(false);
  const [log, setLog] = useState("");
  const [follow, setFollow] = useState(true);
  const terminal = useRef<HTMLPreElement>(null);
  const initialized = useRef("");
  const completed = useRef("");
  const selected = catalog.databases.find(item => item.id === database);
  const running = jobs.some(active) || active(job);
  const chosen = run?.result?.results.find(row => row.policy === (viewPolicy || run.policy)) ?? run?.result?.results[0];
  const stats = run?.state.policy === run?.policy ? run?.state.stats : chosen?.runtime;

  const evaluation = run?.result?.evaluation;
  const own = run?.request.database === "research" ? "model_ranker" : "model_b";
  const ownAccuracy = evaluation?.policies[own];
  const accuracyPolicy = chosen?.policy && chosen.policy !== "none" ? chosen.policy : own;
  const selectedAccuracy = evaluation?.policies[accuracyPolicy];

  async function refreshCatalog(preferred?: string) {
    const data = await api<Catalog>("/api/workspace/databases");
    setCatalog(data); setOnline(true);
    setDatabase(previous => data.databases.some(item => item.id === (preferred || previous)) ? preferred || previous : data.databases.find(item => item.scope === "small")?.id || data.databases[0]?.id || "");
  }
  useEffect(() => {
    void refreshCatalog().catch(problem => setError((problem as Error).message));
    const refresh = () => void api<{ jobs: Job[] }>("/api/jobs").then(data => {
      setJobs([...data.jobs].sort((a, b) => Date.parse(b.started_at) - Date.parse(a.started_at))); setOnline(true);
    }).catch(() => setOnline(false));
    refresh(); const timer = window.setInterval(refresh, 1500);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => {
    if (!database) return;
    let disposed = false;
    setSchema(null); setTable("");
    void api<Schema>(`/api/workspace/schema?database=${encodeURIComponent(database)}`).then(data => { if (!disposed) setSchema(data); }).catch(problem => { if (!disposed) setError((problem as Error).message); });
    const selectedPolicy = database === "research" ? "model_ranker" : "model_b";
    setPolicy(selectedPolicy); setBenchPolicies(["none", "markov2", selectedPolicy]);
    return () => { disposed = true; };
  }, [database]);
  useEffect(() => {
    if (!jobId) return;
    let disposed = false;
    localStorage.setItem("pagewise-workspace-job", jobId);
    const refresh = async () => {
      try {
        const data = await api<{ job: Job; log: string }>(`/api/jobs/${jobId}`);
        if (disposed) return;
        setJob(data.job); setLog(data.log); setOnline(true);
        setJobs(previous => previous.map(item => item.id === data.job.id ? data.job : item));
        if (data.job.preset === "workspace") {
          const output = await api<Run>(`/api/workspace/run?id=${jobId}${viewPolicy ? `&policy=${encodeURIComponent(viewPolicy)}` : ""}`);
          if (disposed) return;
          setRun(output);
          if (initialized.current !== jobId) {
            initialized.current = jobId; setDatabase(output.request.database); setSql(output.request.sql); setParams(JSON.stringify(output.request.params)); setRepeats(output.request.repeats); setRowLimit(output.request.row_limit);
          }
          if (data.job.status === "failed") setError(output.state.error || data.job.error || "SQL execution failed. Inspect the actual terminal output.");
        }
        if (data.job.preset === "demo" && data.job.status === "completed" && completed.current !== jobId) {
          completed.current = jobId; await refreshCatalog(jobId); setTab("database");
        }
      } catch (problem) { if (!disposed) setError((problem as Error).message); }
    };
    void refresh(); const timer = window.setInterval(() => void refresh(), 750);
    return () => { disposed = true; window.clearInterval(timer); };
  }, [jobId, viewPolicy]);
  useEffect(() => { if (follow && terminal.current) terminal.current.scrollTop = terminal.current.scrollHeight; }, [log, follow]);

  async function launch(build = false, benchmark = false) {
    setBusy(true); setError("");
    try {
      const values = build ? [] : JSON.parse(params);
      const payload = build ? { preset: "demo", queries: 100 } : { preset: "workspace", database, sql, params: values, repeats, row_limit: rowLimit, policies: benchmark ? benchPolicies : [policy] };
      const data = await api<{ job: Job }>("/api/jobs", payload);
      setJobId(data.job.id); setJob(data.job); setRun(null); setViewPolicy(""); setLog("");
      setJobs(previous => [data.job, ...previous]);
      if (!build) setTab(benchmark ? "benchmark" : "results");
    } catch (problem) { setError((problem as Error).message); }
    finally { setBusy(false); }
  }
  async function chooseTable(name: string) {
    try {
      const result = await api<Schema>(`/api/workspace/schema?database=${encodeURIComponent(database)}&table=${encodeURIComponent(name)}`);
      setSchema(result); setTable(name);
    } catch (problem) { setError((problem as Error).message); }
  }
  async function cancel() {
    try { if (job) { const data = await api<{ job: Job }>(`/api/jobs/${job.id}/cancel`, {}); setJob(data.job); } }
    catch (problem) { setError((problem as Error).message); }
  }

  return <div className="faculty-page interactive-workspace">
    <div className="faculty-title with-status"><div><p className="eyebrow">REAL SQLITE · LIVE VFS TRACE · PYTHON OUTPUT</p><h1>Interactive workspace</h1><p>Run a query, watch the terminal, and compare our model with other methods.</p></div><span className={`demo-service ${online ? "online" : "offline"}`}>{online ? "Python connected" : "Python offline"}</span></div>
    {error && <div className="local-error" role="alert">{error}<button className="local-secondary" onClick={() => setError("")}>Dismiss</button></div>}
    {!online && <p className="faculty-callout">Start <code>npm run preview:local</code> from the project directory, then reload this page.</p>}
    <section className="faculty-section workspace-database-bar"><label htmlFor="workspace-database">Database<select id="workspace-database" value={database} disabled={running || busy} onChange={event => { setDatabase(event.target.value); setError(""); }}>{catalog.databases.length === 0 && <option value="">Build a small demo to begin</option>}{catalog.databases.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label><div className="workspace-scope"><strong>{selected?.scope === "research" ? "RESEARCH DATABASE" : "SMALL DEVELOPMENT DEMO"}</strong><span>{selected ? `${(selected.bytes / 1024 / 1024).toFixed(2)} MiB · read-only SQL` : "No generated database selected"}</span></div><button className="faculty-primary" disabled={!online || running || busy} onClick={() => void launch(true)}>Create small demo</button><p className="workspace-note">Build uses the existing seven-stage generate → trace → train → evaluate → benchmark pipeline. Every build gets its own SQLite file and output directory.</p></section>
    <div className="workspace-layout"><div className="workspace-main">
      <section className="faculty-section sql-panel"><div className="faculty-section-heading"><div><p className="eyebrow">RUN A REAL DATABASE QUERY</p><h2>Your query</h2></div><select aria-label="Load a query template" defaultValue="" disabled={running || busy} onChange={event => { const template = catalog.templates.find(item => item.id === event.target.value); if (template) { setSql(template.sql); setParams(JSON.stringify(template.params)); } }}><option value="">Load an existing workload query…</option>{catalog.templates.map(item => <option key={item.id} value={item.id}>{item.id.replaceAll("_", " ")}</option>)}</select></div><label className="sr-only" htmlFor="workspace-sql">SQL statement</label><textarea id="workspace-sql" className="sql-editor" spellCheck={false} value={sql} disabled={running || busy} onChange={event => setSql(event.target.value)} /><details className="workspace-options workspace-settings"><summary>More options</summary><div className="sql-controls"><label htmlFor="workspace-params">Query values (JSON array)<input id="workspace-params" value={params} disabled={running || busy} onChange={event => setParams(event.target.value)} spellCheck={false} /></label><label htmlFor="workspace-policy">Model<select id="workspace-policy" value={policy} disabled={running || busy} onChange={event => setPolicy(event.target.value)}>{selected?.policies.map(value => <option key={value} value={value}>{catalog.policies[value]}</option>)}</select></label><label htmlFor="workspace-repeats">Repeat query<select id="workspace-repeats" value={repeats} disabled={running || busy} onChange={event => setRepeats(Number(event.target.value))}>{[1, 3, 5, 10, 20].map(value => <option key={value} value={value}>{value}</option>)}</select></label><label htmlFor="workspace-limit">Maximum rows<input id="workspace-limit" type="number" min={1} max={5000} value={rowLimit} disabled={running || busy} onChange={event => setRowLimit(Number(event.target.value))} /></label></div><details className="workspace-options"><summary>Models to compare ({benchPolicies.length} selected)</summary><div className="workspace-checks">{selected?.policies.map(value => <label key={value}><input type="checkbox" checked={benchPolicies.includes(value)} disabled={running || busy} onChange={event => setBenchPolicies(previous => event.target.checked ? [...previous, value] : previous.filter(item => item !== value))} />{catalog.policies[value]}</label>)}</div></details></details><div className="sql-actions"><button className="faculty-primary" disabled={!database || !online || running || busy} onClick={() => void launch()}>Run query</button><button className="local-secondary" disabled={!database || !online || running || busy || benchPolicies.length < 2} onClick={() => void launch(false, true)}>Compare models</button>{active(job) && <button className="local-secondary" onClick={() => void cancel()}>Stop run</button>}</div><p className="workspace-note">Runs real queries with the matching model. Open More options only if you want to change the settings.</p></section>
      <section className="faculty-section workspace-output"><div className="workspace-tabs" role="tablist" aria-label="Workspace outputs">{tabs.map(([value, label]) => <button key={value} id={`workspace-tab-${value}`} role="tab" aria-controls="workspace-tab-panel" aria-selected={tab === value} onClick={() => setTab(value)}>{label}{value === "trace" && run ? ` (${run.trace.length})` : ""}</button>)}</div>
        <div role="tabpanel" id="workspace-tab-panel" aria-labelledby={`workspace-tab-${tab}`}>
        {tab === "database" && <>{schema ? <><div className="workspace-db-info"><span>{schema.page_count.toLocaleString()} pages</span><span>{schema.page_size} bytes / page</span><a href={download(schema.path)}>Download SQLite file</a></div><p className="workspace-note">{schema.path}</p><div className="workspace-schema">{schema.tables.map(item => <button key={item.name} className={table === item.name ? "selected" : ""} onClick={() => void chooseTable(item.name)}><strong>{item.name}</strong><small>{item.columns.length} columns · {item.indexes.length} indexes</small></button>)}</div>{table && <><h3>{table}</h3><p className="workspace-note">{schema.tables.find(item => item.name === table)?.columns.map(column => `${column.name} ${column.type}${column.primary_key ? " PRIMARY KEY" : ""}`).join(" · ")}</p><pre className="workspace-plan">{schema.tables.find(item => item.name === table)?.sql}</pre><p className="workspace-note">Indexes: {schema.tables.find(item => item.name === table)?.indexes.map(index => `${index.name}${index.unique ? " (unique)" : ""}`).join(", ") || "None"}</p><button className="faculty-link" disabled={running || busy} onClick={() => { setSql(`SELECT * FROM "${table.replaceAll('"', '""')}" LIMIT 25`); setParams("[]"); }}>Use this table in SQL editor ↑</button></>}{schema.preview && <><h3>Actual rows · up to 25</h3><DataTable columns={schema.preview.columns} rows={schema.preview.rows} /></>}</> : <p className="workspace-empty">{database ? "Reading SQLite schema…" : "Build a small demo to create a real database."}</p>}</>}
        {tab !== "database" && !run && <p className="workspace-empty">Run SQL to produce actual {tabs.find(item => item[0] === tab)?.[1].toLowerCase()}. No example measurements are substituted.</p>}
        {run && tab !== "database" && <>
          <div className="workspace-run-bar"><span>Run <strong>{jobId}</strong> · {run.request.database === "research" ? "research database" : "small demo"}</span><label>Policy output<select aria-label="Policy output" value={viewPolicy || run.policy || ""} onChange={event => setViewPolicy(event.target.value)}>{run.request.policies.map(value => <option key={value} value={value}>{catalog.policies[value] || value}</option>)}</select></label></div>
          {selected && database !== run.request.database && <p className="faculty-callout">The editor database changed. These outputs still belong to run {jobId} on database {run.request.database}.</p>}
          {tab === "results" && <>{chosen ? <><div className="workspace-db-info"><span>{chosen.measurements[0]?.rows} rows {chosen.measurements[0]?.truncated ? "· result cap reached" : "returned"}</span><span>{selectedAccuracy ? `${fixed(selectedAccuracy.top1_micro_pct)}% Top-1 prediction accuracy` : "Accuracy will appear after evaluation"}</span><span>{chosen.label}</span></div><DataTable columns={chosen.columns} rows={chosen.preview} /><p className="workspace-note">Preview contains up to 100 rows; CSV contains up to {run.request.row_limit}. Long cells are clipped in the browser with preview metadata. BLOB values show byte length and a 64-byte hex preview.</p><details className="workspace-options"><summary>How SQLite ran this query</summary><pre className="workspace-plan">{chosen.plan.join("\n")}</pre><p className="workspace-note">Query type: {chosen.template}.</p></details></> : <p className="workspace-empty">SQL is executing. Follow the terminal and page reads while results are prepared.</p>}</>}
          {tab === "trace" && <><p className="workspace-note">Latest {run.trace.length} recorded database reads. Updates while the query runs. These are file reads; some may be served from the OS cache.</p><DataTable columns={["Seq", "Query", "Page range", "Bytes", "Read µs", "Cache", "Source", "Plan"]} rows={run.trace.map(event => [event.seq, event.query_id || "setup", event.first_page === event.last_page ? event.first_page : `${event.first_page}–${event.last_page}`, event.amount, event.duration_us, event.cache_status, event.source, event.plan_class])} /></>}
          {tab === "predictions" && <><p className="workspace-note">Real next-page guesses from the chosen model. A score ranks the possible pages; it is not an accuracy percentage.</p>{run.predictions.map(event => <article className="workspace-prediction" key={event.prediction_id}><div><strong>Prediction #{event.prediction_id} · {event.query_id}</strong><details className="runtime-details"><summary>Job timing</summary><span>{fixed(event.inference_ms, 3)} ms inference | {fixed(event.queue_wait_ms, 3)} ms queue wait</span></details></div><p>History: {event.history?.join(" → ")}</p><DataTable columns={["Rank", "Next page", "Score", "Runtime action"]} rows={(event.ranked_candidates || []).map((candidate, index) => [index + 1, candidate.page, candidate.score, event.selected?.find(row => row.page === candidate.page)?.status || "not selected"])} /></article>)}{run.predictions.length === 0 && <p className="workspace-empty">No recorded prediction jobs for this policy yet.</p>}</>}
          {tab === "prefetch" && <><p className="workspace-note">Watch pages being fetched early. Useful means the query later used a page that was fetched ahead of time.</p><div className="workspace-counters">{[["Demand reads", "demand_requests"], ["Cache hits", "cache_hits"], ["Cache misses", "cache_misses"], ["Scored jobs", "completed_jobs"], ["Queue length", "queue_length"], ["Prefetched entries", "pages_prefetched"], ["Useful entries", "useful_prefetches"], ["Unused entries", "total_wasted_prefetches"], ["Dropped jobs", "dropped_jobs"], ["Stale jobs dropped", "stale_jobs_dropped"], ["Prediction / read errors", "prefetch_read_errors"], ["Cache capacity (pages)", "capacity_pages"]].map(([label, key]) => <article key={key}><span>{label}</span><strong>{stats?.[key] == null ? "n/a" : String(stats[key])}</strong></article>)}</div><h3>Recorded uses of prefetched entries</h3><DataTable columns={["Page", "Prediction ID", "Demand query", "Lead time ms"]} rows={run.uses.map(event => [event.page, event.prediction_id, event.demand_query_id, event.lead_time_ms])} /></>}
          {tab === "benchmark" && <>{run.result ? <>
            <p className="workspace-note">Prediction accuracy on this SQL workload. Every predictor is checked against the same observed next pages.</p>
            {evaluation?.available && ownAccuracy ? <>
              <div className="faculty-metrics"><article><span>{run.request.database === "research" ? "Our neural hybrid" : "Small semantic model"} Top-1</span><strong>{fixed(ownAccuracy.top1_micro_pct)}<small>%</small></strong><p>The first guessed page was correct</p></article><article><span>Top-5 accuracy</span><strong>{fixed(ownAccuracy.top5_micro_pct)}<small>%</small></strong><p>The actual page was among the first five</p></article><article><span>Difference from Markov-2</span><strong>{ownAccuracy.top1_micro_pct - evaluation.policies.markov2.top1_micro_pct >= 0 ? "+" : ""}{fixed(ownAccuracy.top1_micro_pct - evaluation.policies.markov2.top1_micro_pct)}<small> pp</small></strong><p>Same evaluated next-page transitions</p></article></div>
              <DataTable columns={["Predictor", "Top-1 accuracy", "Top-5 accuracy", "Next-page checks"]} rows={Object.entries(evaluation.policies).map(([policy, row]) => [catalog.policies[policy] || policy, `${row.top1_micro_pct.toFixed(2)}%`, `${row.top5_micro_pct.toFixed(2)}%`, row.transitions])} />
              <p className="workspace-note">Both accuracy columns weight each scored transition equally. {evaluation.sampled ? `${evaluation.scored_transitions} of ${evaluation.eligible_transitions} transitions checked, evenly spaced with up to ${evaluation.max_transitions_per_query} per query.` : `All ${evaluation.scored_transitions} eligible transitions checked.`} {evaluation.scope}</p>
            </> : <p className="workspace-empty">{evaluation?.reason || "This older run has no prediction-accuracy evaluation. Click Compare models to compute fresh accuracy."}</p>}
            <details className="runtime-details"><summary>Timing and execution details</summary><p className="workspace-note">The same SQL and parameters ran for every timed policy. Execution order: {run.result.order.map(value => catalog.policies[value] || value).join(" -> ")}.</p><DataTable columns={["Policy", "Executions", "Mean ms", "Median ms", "Rows / execution"]} rows={run.result.results.map(row => [row.label, row.measurements.length, row.mean_ms.toFixed(3), row.median_ms.toFixed(3), row.measurements.map(value => value.rows).join(", ")])} /><p className="workspace-note">Database unchanged: {run.result.database_unchanged ? "verified" : "not verified"}. Returned row stream identical: {run.result.identical_row_order_and_values ? "yes" : "no; ordering and nondeterministic SQL can affect this"}.</p><ul className="workspace-notes">{run.result.notes.map(note => <li key={note}>{note}</li>)}</ul></details>
          </> : <p className="workspace-empty">{run.state.phase === "evaluating_accuracy" ? "Checking predictions against the actual next pages..." : `Running comparison: ${run.state.policy || "preparing"}, repetition ${run.state.repeat || 0}/${run.request.repeats}.`}</p>}</>}
          {tab === "files" && <><p className="workspace-note">Real files from this run. Downloads preserve the measured outputs; historical research files are separate.</p><div className="workspace-artifacts">{run.artifacts.filter(file => !file.name.startsWith("model/")).map(file => <a key={file.path} href={download(file.path)}><strong>{file.name}</strong><span>{file.bytes.toLocaleString()} bytes ↓</span></a>)}</div>{job && <a className="faculty-link" href={download(`runs/local-lab/${job.id}.log`)}>Download terminal log ↓</a>}</>}
        </>}
        </div>
      </section>
    </div><aside className="workspace-terminal"><div className="terminal-heading"><div><span className="terminal-dot" /><strong>Live terminal</strong></div><span>{job?.status || "idle"}</span></div><label className="terminal-history">Local run<select aria-label="Workspace run history" value={jobId} onChange={event => { setJobId(event.target.value); setViewPolicy(""); setRun(null); setError(""); }}>{!jobs.some(item => item.id === jobId) && <option value="">Select a run</option>}{jobs.filter(item => ["workspace", "demo"].includes(item.preset || "")).map(item => <option key={item.id} value={item.id}>{item.id} · {item.preset} · {item.status}</option>)}</select></label><p className="terminal-meta">{job ? `${job.root} · ${job.status === "running" ? `stage ${job.step}/${job.total_steps}` : job.status}` : "Build a small database or execute SQL to start a real process."}</p><pre ref={terminal} className="workspace-log" aria-label="Actual Python process log" aria-live="off">{log || "No process output yet."}</pre><label className="terminal-follow"><input type="checkbox" checked={follow} onChange={event => setFollow(event.target.checked)} />Follow output</label><p className="terminal-footnote">Actual Python output. Open the tabs to see reads, predictions and results.</p></aside></div>
  </div>;
}
