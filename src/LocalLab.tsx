import { useEffect, useRef, useState } from "react";
import AdvancedLab from "./AdvancedLab";
import DemoResults, { type DemoSnapshot } from "./DemoResults";
import TrainedHybridResults, { type TrainedHybridSnapshot } from "./TrainedHybridResults";

type Job = { id: string; status: string; step: number; total_steps: number; command: string; started_at: string; root: string; error?: string; preset?: string; stage_labels?: string[]; result_file?: string };
type Status = "checking" | "online" | "offline";
const stages = [
  ["Create database", "Build a small SQLite dataset."], ["Observe page reads", "Run SQL queries and record reads."],
  ["Prepare examples", "Turn the reads into page sequences."], ["Train models", "Learn history-only and semantic models."],
  ["Test predictions", "Check unseen query executions."], ["Measure query time", "Compare the prefetch policies."], ["Prepare results", "Save the measured dashboard."],
];
const active = (job: Job) => job.status === "running" || job.status === "cancelling";
const time = (job: Job) => new Date(job.started_at).toLocaleString([], { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
async function request<T>(path: string, body?: object): Promise<T> {
  const response = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : { cache: "no-store" });
  let payload;
  try { payload = await response.json(); } catch { throw new Error("The local service is unavailable. Start npm run preview:local and retry."); }
  if (!response.ok) throw new Error(payload.error || "The request could not be completed.");
  return payload as T;
}

export default function LocalLab({ onLoadSnapshot }: { onLoadSnapshot: (snapshot: unknown) => void }) {
  const [connection, setConnection] = useState<Status>("checking");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selected, setSelected] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [queries, setQueries] = useState(100);
  const [hybridQueries, setHybridQueries] = useState(20);
  const [mode, setMode] = useState<"trained-hybrid" | "demo">("trained-hybrid");
  const [ready, setReady] = useState<{ available: boolean; reason?: string } | null>(null);
  const [error, setError] = useState("");
  const [log, setLog] = useState("");
  const [busy, setBusy] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [result, setResult] = useState<DemoSnapshot | null>(null);
  const [trainedResult, setTrainedResult] = useState<TrainedHybridSnapshot | null>(null);
  const [loadingResult, setLoadingResult] = useState(false);
  const selectedRef = useRef("");
  const launchedRef = useRef("");
  const running = jobs.some(active) || Boolean(job && active(job));

  useEffect(() => {
    let disposed = false;
    const refresh = async () => {
      try {
        const history = await request<{ jobs: Job[] }>("/api/jobs");
        if (disposed) return;
        const sorted = [...history.jobs].sort((a, b) => Date.parse(b.started_at) - Date.parse(a.started_at));
        setConnection("online"); setJobs(sorted);
        if (!selectedRef.current) {
          const preferred = sorted.find(active) ?? sorted.find(item => (item.preset === mode || (mode === "demo" && item.total_steps === 7)) && item.status === "completed");
          if (preferred) { selectedRef.current = preferred.id; setSelected(preferred.id); }
        }
      } catch { if (!disposed) setConnection("offline"); }
    };
    void refresh(); const timer = window.setInterval(() => void refresh(), 2000);
    return () => { disposed = true; window.clearInterval(timer); };
  }, [mode]);

  useEffect(() => {
    if (connection !== "online") return;
    let disposed = false;
    void request<{ available: boolean; reason?: string }>("/api/trained-hybrid").then(value => { if (!disposed) setReady(value); }).catch(() => { if (!disposed) setReady({ available: false, reason: "Could not verify the local trained-model artifacts." }); });
    return () => { disposed = true; };
  }, [connection]);

  useEffect(() => {
    if (!selected) return;
    let disposed = false;
    setResult(null); setTrainedResult(null); setLoadingResult(false); setLog("");
    let requestedResult = false;
    const update = async () => {
      try {
        const response = await request<{ job: Job; log: string }>(`/api/jobs/${selected}`);
        if (disposed) return;
        setJob(response.job); setLog(response.log);
        if (response.job.status === "completed" && (response.job.result_file || response.job.total_steps === 7) && !requestedResult) {
          requestedResult = true; setLoadingResult(true);
          try {
            const file = await request<{ content: string }>(`/api/file?path=${encodeURIComponent(`${response.job.root}/${response.job.result_file || "dashboard.json"}`)}`);
            const snapshot = JSON.parse(file.content);
            if (snapshot.artifact_type === "trained_hybrid_live") {
              if (!Array.isArray(snapshot.rows) || !snapshot.rows.some((row: { policy: string }) => row.policy === "model_ranker")) throw new Error("The trained model run has no readable result.");
              if (!disposed) setTrainedResult(snapshot as TrainedHybridSnapshot);
            } else {
              if (snapshot.schema_version !== 1 || !snapshot.prediction?.policies || !snapshot.runtime_suites) throw new Error("This run has no readable dashboard export.");
              if (!disposed) setResult(snapshot as DemoSnapshot);
            }
          } catch (problem) { if (!disposed) setError((problem as Error).message); }
          finally { if (!disposed) setLoadingResult(false); }
        }
      } catch { if (!disposed) setConnection("offline"); }
    };
    void update(); const timer = window.setInterval(() => void update(), 2000);
    return () => { disposed = true; window.clearInterval(timer); };
  }, [selected]);

  async function launch() {
    setBusy(true); setError("");
    try {
      const response = await request<{ job: Job }>("/api/jobs", { preset: mode, queries: mode === "trained-hybrid" ? hybridQueries : queries });
      launchedRef.current = response.job.id; selectedRef.current = response.job.id;
      setSelected(response.job.id); setJob(response.job); setJobs(previous => [response.job, ...previous]); setResult(null); setTrainedResult(null);
    } catch (problem) { setError((problem as Error).message); }
    finally { setBusy(false); }
  }
  async function cancel() {
    if (!job) return;
    try { const response = await request<{ job: Job }>(`/api/jobs/${job.id}/cancel`, {}); setJob(response.job); }
    catch (problem) { setError((problem as Error).message); }
  }
  function changeMode(value: "trained-hybrid" | "demo") {
    setMode(value); selectedRef.current = ""; setSelected(""); setJob(null); setResult(null); setTrainedResult(null); setLog(""); setError("");
  }
  const hybridMode = mode === "trained-hybrid";
  const isGuided = job?.total_steps === 7 || job?.preset === "trained-hybrid";
  const displayedStages = job?.stage_labels?.map(label => [label, "Actual paired query execution with fixed model weights."]) ?? (job?.total_steps === 7 ? stages : hybridMode ? [
    ["Verify trained model", "Check the frozen weights and matching database."], ["Run paired workloads", "Compare no prefetch, Markov-2 and the selected model."], ["Evaluate predictions", "Score every observed next-page transition with the semantic model."], ["Open live results", "Display accuracy, query timings and actual model output."],
  ] : stages);
  const completed = job?.status === "completed";
  const finished = completed ? displayedStages.length : Math.max(0, (job?.step ?? 0) - 1);
  const saved = job?.id !== launchedRef.current;

  return <div className="faculty-page live-demo-page">
    <div className="faculty-title with-status"><div><p className="eyebrow">RUN IT ON THIS COMPUTER</p><h1>Live demonstration</h1><p>Test our trained selected hybrid on its original database, or build a small experiment from scratch.</p></div><span className={`demo-service ${connection}`}>{connection === "online" ? "Local service ready" : connection === "checking" ? "Connecting…" : "Local service offline"}</span></div>
    {connection === "offline" && <div className="faculty-callout" role="status">Start <code>npm run preview:local</code> from the project folder. This screen will reconnect automatically.</div>}
    {error && <div className="local-error" role="alert">{error}</div>}
    <div className="faculty-segment demo-mode" role="group" aria-label="Live demonstration mode"><button aria-pressed={hybridMode} disabled={running || busy} onClick={() => changeMode("trained-hybrid")}>Test trained hybrid</button><button aria-pressed={!hybridMode} disabled={running || busy} onClick={() => changeMode("demo")}>Build a small demo</button></div>
    {hybridMode ? <section className="faculty-section demo-launch"><div><p className="eyebrow">SELECTED NEURAL HYBRID · ORIGINAL 8.62 GB DATABASE</p><h2>Run the model we selected</h2><p>Loads the trained checkpoint and checks fresh next-page predictions against the Markov baselines. Model weights and the selected scheduling policy stay fixed.</p><p className="demo-scope">This executes the actual selected model. It reports fresh Top-1/Top-5 accuracy from actual next-page traces. Timing details remain available separately. The saved held-out study remains separate.</p>{ready?.available === false && <p className="local-error" role="status">{ready.reason}</p>}</div><div className="demo-launch-controls"><label htmlFor="hybrid-queries">Queries per policy / seed<select id="hybrid-queries" value={hybridQueries} disabled={running || busy} onChange={event => setHybridQueries(Number(event.target.value))}>{[10, 20, 30].map(value => <option key={value} value={value}>{value} queries</option>)}</select></label><button className="faculty-primary" onClick={() => void launch()} disabled={connection !== "online" || !ready?.available || busy || running}>{busy ? "Starting…" : running ? "Experiment running…" : "Run trained hybrid"}<span aria-hidden="true">↗</span></button><small>{ready?.available ? "Frozen checkpoint verified · no retraining" : ready ? "Matching local research artifacts required" : "Checking trained-model artifacts…"}<br />{hybridQueries} queries × 3 policies × 2 new seeds.</small></div></section> : <section className="faculty-section demo-launch"><div><p className="eyebrow">SMALL DATABASE · TRAIN FROM SCRATCH</p><h2>Run a fresh development experiment</h2><p>Creates its own data, trains Model A and Model B, and compares eight runtime configurations. No artificial delay is added.</p><p className="demo-scope">The standard hybrid in this small demo is separate from the frozen selected neural hybrid. Use “Test trained hybrid” to run the selected model.</p></div><div className="demo-launch-controls"><label htmlFor="demo-queries">Queries to trace<select id="demo-queries" value={queries} disabled={running || busy} onChange={event => setQueries(Number(event.target.value))}>{[100, 150, 300].map(value => <option key={value} value={value}>{value} queries</option>)}</select></label><button className="faculty-primary" onClick={() => void launch()} disabled={connection !== "online" || busy || running}>{busy ? "Starting…" : running ? "Experiment running…" : "Start live demo"}<span aria-hidden="true">↗</span></button><small>Runtime comparison: 30 queries × 8 policies × 2 seeds.</small></div></section>}
    <section className="faculty-section demo-progress"><div className="faculty-section-heading"><div><p className="eyebrow">{job && saved ? "PREVIOUS LOCAL RUN" : "LIVE PIPELINE"}</p><h2>{!job ? "What will happen" : completed ? "Experiment complete" : job.status === "cancelling" ? "Stopping the experiment" : job.status === "cancelled" ? "Experiment cancelled" : job.status === "failed" ? "Experiment stopped with an error" : job.status === "interrupted" ? "Previous run was interrupted" : `Step ${job.step || 1} of ${job.total_steps}`}</h2></div>{job && <span className="demo-time">{time(job)}</span>}</div>
      {job && !isGuided ? <p className="faculty-description">Another backend command is {job.status}: {job.command}. Open advanced tools for its details.</p> : <ol className="demo-stage-list">{displayedStages.map(([title, description], index) => { const done = Boolean(job) && index < finished; const current = job !== null && active(job) && index === Math.max(0, job.step - 1); return <li key={title} className={done ? "done" : current ? "current" : ""} aria-current={current ? "step" : undefined}><span className="demo-stage-marker">{done ? "✓" : String(index + 1).padStart(2, "0")}</span><div><strong>{title}</strong><small>{description}</small></div>{current && <span className="demo-stage-active">{job?.status === "cancelling" ? "Stopping" : "Running"}</span>}</li>; })}</ol>}
      {job && active(job) && job.status !== "cancelling" && <button className="local-secondary" onClick={() => void cancel()}>Stop this run</button>}
      {loadingResult && <p role="status">Opening the measured results…</p>}
      {completed && (result || trainedResult) && <button className="faculty-primary" onClick={() => document.getElementById("demo-results")?.scrollIntoView({ behavior: "smooth", block: "start" })}>View measured results ↓</button>}
      {job && <details className="demo-log"><summary>Execution details {job.status === "failed" ? "· inspect the error" : ""}</summary><p className="local-note">{job.error || `Run ${job.id} · ${job.root}`}</p><pre className="local-log">{log || "Waiting for backend output…"}</pre></details>}
    </section>
    {result && <DemoResults snapshot={result} onInspect={() => onLoadSnapshot(result)} />}
    {trainedResult && <TrainedHybridResults result={trainedResult} />}
    <details className="faculty-advanced" onToggle={event => setAdvanced(event.currentTarget.open)}><summary><span>Advanced research tools</span><small>Individual commands, earlier runs and raw files</small></summary>{advanced && <AdvancedLab onLoadSnapshot={onLoadSnapshot} />}</details>
  </div>;
}
