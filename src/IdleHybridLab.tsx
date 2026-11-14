import { useEffect, useState } from "react";

type Job = { id: string; root: string; status: string; preset?: string; started_at: string; error?: string };
type Row = {
  policy: string; label: string; mean_ms: number; process_cpu_ms: number; neural_scoring_calls: number;
  runtime: { pages_prefetched: number; useful_prefetches: number; pollution_misses?: number };
};
type Result = { artifact_type: string; dataset: string; queries_per_policy: number; seed: number; rows: Row[] };
const running = (job: Job | null) => job?.status === "running" || job?.status === "cancelling";
async function api<T>(path: string, body?: object): Promise<T> {
  const response = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : { cache: "no-store" });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || "The local request failed.");
  return value as T;
}

export default function IdleHybridLab() {
  const [dataset, setDataset] = useState("small");
  const [queries, setQueries] = useState(20);
  const [ready, setReady] = useState<{ available: boolean; reason?: string } | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [log, setLog] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let disposed = false;
    setReady(null);
    void api<{ available: boolean; reason?: string }>(`/api/idle-hybrid?dataset=${dataset}`)
      .then(value => { if (!disposed) setReady(value); }).catch(() => { if (!disposed) setReady({ available: false, reason: "Start the local Pagewise service to check the model." }); });
    return () => { disposed = true; };
  }, [dataset]);

  useEffect(() => {
    let disposed = false;
    void api<{ jobs: Job[] }>("/api/jobs").then(value => {
      const previous = value.jobs.filter(item => item.preset === "idle-hybrid")
        .sort((a, b) => Date.parse(b.started_at) - Date.parse(a.started_at))[0];
      if (!disposed && previous) setJob(current => current ?? previous);
    }).catch(() => {});
    return () => { disposed = true; };
  }, []);

  useEffect(() => {
    if (!job?.id) return;
    let disposed = false, opened = false;
    const id = job.id;
    const refresh = async () => {
      try {
        const value = await api<{ job: Job; log: string }>(`/api/jobs/${id}`);
        if (disposed) return;
        setJob(value.job); setLog(value.log);
        if (value.job.status === "failed") setError(value.job.error || "The run failed. Inspect the terminal output.");
        if (value.job.status === "completed" && !opened) {
          const file = await api<{ content: string }>(`/api/file?path=${encodeURIComponent(`${value.job.root}/idle-result.json`)}`);
          const measured = JSON.parse(file.content) as Result;
          if (measured.artifact_type !== "idle_hybrid_live" || !measured.rows.some(row => row.policy === "P4-cached")) throw new Error("This run has no readable idle neural result.");
          if (!disposed) { setResult(measured); opened = true; }
        }
      } catch (problem) { if (!disposed) setError((problem as Error).message); }
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => { disposed = true; window.clearInterval(timer); };
  }, [job?.id]);

  async function start() {
    setBusy(true); setError(""); setResult(null); setLog("");
    try { const value = await api<{ job: Job }>("/api/jobs", { preset: "idle-hybrid", dataset, queries }); setJob(value.job); }
    catch (problem) { setError((problem as Error).message); }
    finally { setBusy(false); }
  }
  async function stop() {
    if (!job) return;
    try { const value = await api<{ job: Job }>(`/api/jobs/${job.id}/cancel`, {}); setJob(value.job); }
    catch (problem) { setError((problem as Error).message); }
  }
  const selected = result?.rows.find(row => row.policy === "P4-cached");
  const earlier = result?.rows.find(row => row.policy === "P4-original");
  const useful = selected?.runtime.pages_prefetched ? 100 * selected.runtime.useful_prefetches / selected.runtime.pages_prefetched : null;
  const visible = result?.rows.filter(row => row.policy !== "P0").sort((a, b) => ["P2", "P3", "P4-original", "P4-cached", "P4-selective"].indexOf(a.policy) - ["P2", "P3", "P4-original", "P4-cached", "P4-selective"].indexOf(b.policy));
  const locked = busy || running(job);
  return <>
    <section className="faculty-section demo-launch">
      <div><p className="eyebrow">SAVED SEMANTIC MODEL · IDLE-TIME PREFETCHING</p><h2>Run the idle neural hybrid</h2>
        <p>After a query finishes, our model predicts pages for the next query and loads them during the pause. New query arrival stops further speculative reads.</p>
        <p className="demo-scope">Uses separately trained P4 weights. This demo adds a controlled 50 ms pause between queries and no artificial storage delay. The original research study stays separate.</p></div>
      <div className="demo-launch-controls">
        <label htmlFor="idle-database">Database<select id="idle-database" value={dataset} disabled={locked} onChange={event => setDataset(event.target.value)}><option value="small">Small local database</option><option value="large">Original 8.62 GB database</option></select></label>
        <label htmlFor="idle-queries">Queries per policy<select id="idle-queries" value={queries} disabled={locked} onChange={event => setQueries(Number(event.target.value))}>{[10, 20, 30].map(value => <option key={value} value={value}>{value} queries</option>)}</select></label>
        <button className="faculty-primary" disabled={!ready?.available || locked} onClick={() => void start()}>{locked ? "Experiment running…" : "Run idle neural hybrid"}</button>
        <small>{ready?.available ? "Saved checkpoint verified · matching SQL across policies" : ready?.reason || "Checking local model…"}</small>
      </div>
    </section>
    {error && <p className="local-error" role="alert">{error}</p>}
    <section className="faculty-section"><div className="faculty-section-heading"><div><p className="eyebrow">ACTUAL PYTHON OUTPUT</p><h2>Live terminal</h2></div><span role="status">{job ? `${job.status} · ${job.id}` : "Ready to run"}</span></div>
      <pre className="local-log" aria-label="Idle experiment terminal">{log.split("\n").filter(line => !line.startsWith("P0:")).join("\n") || "Start a run to see real execution logs here."}</pre>
      {running(job) && <button className="local-secondary" onClick={() => void stop()}>Stop this run</button>}
    </section>
    {result && selected && <section className="faculty-section" id="idle-results">
      <div className="faculty-section-heading"><div><p className="eyebrow">MEASURED ON THIS COMPUTER</p><h2>Idle hybrid results</h2></div><a className="local-secondary" href={`/api/download?path=${encodeURIComponent(`${job?.root}/idle-result.json`)}`}>Download results</a></div>
      <p>{result.dataset === "large" ? "Original 8.62 GB" : "Small local"} database · {result.queries_per_policy} matching queries per policy · 50 ms pauses · no added storage delay.</p>
      <div className="idle-summary-grid">
        <div className="idle-summary-card"><span>Useful prefetched pages</span><strong>{selected.runtime.useful_prefetches} / {selected.runtime.pages_prefetched}</strong><small>{useful === null ? "No pages were prefetched" : `${useful.toFixed(1)}% were requested later`}</small></div>
        <div className="idle-summary-card"><span>Optimized neural CPU time</span><strong>{selected.process_cpu_ms.toFixed(1)} ms</strong><small>{earlier ? `Earlier idle neural: ${earlier.process_cpu_ms.toFixed(1)} ms` : "Process CPU across the workload"}</small></div>
        <div className="idle-summary-card"><span>Mean query time</span><strong>{selected.mean_ms.toFixed(3)} ms</strong><small>Measured SQL execution time</small></div>
      </div>
      <div className="faculty-table-wrap"><table className="faculty-table"><thead><tr><th>Policy</th><th>Mean query time</th><th>Process CPU</th><th>Useful / prefetched</th><th>Neural calls</th></tr></thead><tbody>{visible?.map(row => <tr key={row.policy}><td>{row.policy === "P4-cached" ? <strong>{row.label} · our model</strong> : row.label}</td><td>{row.mean_ms.toFixed(3)} ms</td><td>{row.process_cpu_ms.toFixed(1)} ms</td><td>{row.runtime.useful_prefetches} / {row.runtime.pages_prefetched}</td><td>{row.neural_scoring_calls}</td></tr>)}</tbody></table></div>
      <p className="local-note">Useful means a prefetched page was later requested. CPU time adds work across process threads; query time measures elapsed SQL execution. The selective variant may use Markov alone when candidates fit its budget or the prior clearly separates the cutoff. These are local demonstration measurements; OS cache is not reset.</p>
    </section>}
  </>;
}
