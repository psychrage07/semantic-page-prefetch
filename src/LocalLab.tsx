import { useEffect, useMemo, useRef, useState } from "react";

type Option = { name: string; required: boolean; default: unknown; choices: string[] | null; help: string; kind: string; multiple: boolean; path: boolean };
type Command = { name: string; help: string; options: Option[] };
type Job = { id: string; status: string; step: number; total_steps: number; command: string; started_at: string; finished_at: string | null; root: string; log: string; exit_code?: number; error?: string };
type Entry = { name: string; path: string; directory: boolean; bytes: number | null };

async function api<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, { cache: "no-store", ...init });
  let payload: Record<string, unknown>;
  try { payload = await response.json(); } catch { throw new Error("Local backend is unavailable. Start it with npm run dev:local."); }
  if (!response.ok) throw new Error(String(payload.error ?? `Backend returned ${response.status}`));
  return payload as T;
}

function commandGroups(commands: Command[]) {
  const groups: [string, string[]][] = [
    ["Setup and data", ["check-env", "generate-db", "trace", "build-sequences"]],
    ["Models and accuracy", ["train", "train-sweep", "evaluate", "offline-study", "offline-study-v21", "offline-ablation"]],
    ["Runtime", ["run-prefetch", "run-suite", "runtime-sweep", "headroom-v21", "sim-validate-cells"]],
    ["Reports and exports", ["summarize-suite", "recover-suite", "report", "research-report", "report-v21", "dashboard-export", "dashboard-export-v21"]],
    ["Research scripts", commands.filter(command => command.name.startsWith("script:")).map(command => command.name)],
  ];
  return groups.map(([label, names]) => [label, names.filter(name => commands.some(command => command.name === name))] as const);
}

export default function LocalLab({ onLoadSnapshot }: { onLoadSnapshot: (snapshot: unknown) => void }) {
  const [connected, setConnected] = useState(false);
  const [commands, setCommands] = useState<Command[]>([]);
  const [selected, setSelected] = useState("check-env");
  const [values, setValues] = useState<Record<string, string | boolean>>({});
  const [jobs, setJobs] = useState<Job[]>([]);
  const [jobId, setJobId] = useState("");
  const [job, setJob] = useState<Job | null>(null);
  const [log, setLog] = useState("");
  const [directory, setDirectory] = useState("runs/local-lab");
  const [entries, setEntries] = useState<Entry[]>([]);
  const [artifact, setArtifact] = useState<{ path: string; content: string } | null>(null);
  const [demoQueries, setDemoQueries] = useState(100);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const completedRef = useRef("");
  const activeCommand = useMemo(() => commands.find(command => command.name === selected), [commands, selected]);

  async function refresh() {
    try {
      const [health, catalog, history] = await Promise.all([
        api<{ ok: boolean }>("/api/health"), api<{ commands: Command[] }>("/api/commands"), api<{ jobs: Job[] }>("/api/jobs"),
      ]);
      setConnected(health.ok);
      setCommands(catalog.commands);
      setJobs(history.jobs);
      setError("");
      if (!jobId && history.jobs.length) setJobId(history.jobs[0].id);
    } catch (problem) {
      setConnected(false);
      setError((problem as Error).message);
    }
  }

  async function listFiles(path: string) {
    try {
      const result = await api<{ path: string; entries: Entry[] }>(`/api/files?path=${encodeURIComponent(path)}`);
      setDirectory(path);
      setEntries(result.entries);
      setError("");
    } catch (problem) { setError((problem as Error).message); }
  }

  async function viewFile(path: string) {
    try {
      const file = await api<{ path: string; content: string }>(`/api/file?path=${encodeURIComponent(path)}`);
      setArtifact(file);
      setError("");
    } catch (problem) { setError((problem as Error).message); }
  }

  useEffect(() => {
    void refresh();
    void listFiles("runs/local-lab");
    const timer = window.setInterval(() => {
      void api<{ jobs: Job[] }>("/api/jobs").then(result => {
        setConnected(true);
        setJobs(result.jobs);
      }).catch(() => setConnected(false));
    }, 2000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!jobId) return;
    const update = () => {
      void api<{ job: Job; log: string }>(`/api/jobs/${jobId}`).then(result => {
        setJob(result.job);
        setLog(result.log);
        if (result.job.status !== "running" && result.job.status !== "cancelling" && completedRef.current !== result.job.id) {
          completedRef.current = result.job.id;
          void listFiles(result.job.root);
        }
      }).catch(problem => setError((problem as Error).message));
    };
    update();
    const timer = window.setInterval(update, 2000);
    return () => window.clearInterval(timer);
  }, [jobId]);

  function selectCommand(name: string) {
    setSelected(name);
    setValues({});
    setError("");
  }

  async function launch(body: object) {
    setBusy(true);
    setError("");
    try {
      const result = await api<{ job: Job }>("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      setJobId(result.job.id);
      setJob(result.job);
      setLog("");
      setJobs(previous => [result.job, ...previous]);
      setDirectory(result.job.root);
    } catch (problem) { setError((problem as Error).message); }
    finally { setBusy(false); }
  }

  function launchCommand() {
    const options: Record<string, string | boolean | string[]> = {};
    for (const item of activeCommand?.options ?? []) {
      const value = values[item.name];
      if (value === undefined || value === "" || value === false) continue;
      options[item.name] = item.multiple && typeof value === "string" ? value.split(/[,\n]/).map(part => part.trim()).filter(Boolean) : value;
    }
    void launch({ command: selected, options });
  }

  async function cancel() {
    if (!job || job.status !== "running") return;
    try {
      const result = await api<{ job: Job }>(`/api/jobs/${job.id}/cancel`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
      setJob(result.job);
    } catch (problem) { setError((problem as Error).message); }
  }

  function loadSnapshot() {
    if (!artifact) return;
    try {
      const payload = JSON.parse(artifact.content);
      if (payload.schema_version !== 1) throw new Error("This JSON file is not a Pagewise dashboard snapshot.");
      onLoadSnapshot(payload);
      setError("");
    } catch (problem) { setError((problem as Error).message); }
  }

  return <div className="local-lab">
    <div className="local-heading">
      <div><p className="eyebrow">LOCAL BACKEND / REAL SQLITE RUNS</p><h1>Experiment workspace</h1><p>Run every Pagewise backend command from this computer, follow its output, and open its measured artifacts.</p></div>
      <span className={`local-connection ${connected ? "online" : ""}`}>{connected ? "Local backend connected" : "Local backend offline"}</span>
    </div>
    {error && <div className="local-error" role="alert">{error}</div>}
    {!connected && <p className="local-help">From the repository root, run <code>npm run dev:local</code> and open the shown browser address. The backend uses <code>backend/.venv</code> when available. <button className="local-secondary" onClick={() => void refresh()}>Retry connection</button></p>}
    <section className="local-panel local-preset">
      <div><p className="eyebrow">ONE-CLICK DEMO</p><h2>Run the full pipeline</h2><p>Generate a small SQLite database, trace real reads, build sequences, train A and B, evaluate them, compare runtime policies, and export a measured dashboard. This is a development run, not the final research study.</p></div>
      <label>Traced queries <input type="number" min="1" max="2000" value={demoQueries} onChange={event => setDemoQueries(Number(event.target.value))} /></label>
      <button className="local-primary" disabled={!connected || busy || jobs.some(item => item.status === "running")} onClick={() => void launch({ preset: "demo", queries: demoQueries })}>Run local experiment</button>
    </section>
    <div className="local-grid">
      <section className="local-panel">
        <p className="eyebrow">ALL BACKEND FEATURES</p><h2>Run a command</h2>
        <label className="local-field">Command<select value={selected} onChange={event => selectCommand(event.target.value)}>
          {commandGroups(commands).map(([label, names]) => <optgroup label={label} key={label}>{names.map(name => <option key={name} value={name}>{name}</option>)}</optgroup>)}
        </select></label>
        <p className="local-description">{activeCommand?.help}</p>
        <div className="local-options">{activeCommand?.options.map(option => <label className="local-field" key={option.name}>
          <span>--{option.name}{option.required && " *"}</span>
          {option.kind === "boolean" ? <input type="checkbox" checked={values[option.name] === true} onChange={event => setValues(previous => ({ ...previous, [option.name]: event.target.checked }))} />
            : option.choices ? <select value={String(values[option.name] ?? "")} onChange={event => setValues(previous => ({ ...previous, [option.name]: event.target.value }))}><option value="">Default{option.default != null ? ` (${option.default})` : ""}</option>{option.choices.map(choice => <option key={choice} value={choice}>{choice}</option>)}</select>
              : <input type={option.kind === "int" || option.kind === "float" ? "number" : "text"} step={option.kind === "float" ? "any" : undefined} value={String(values[option.name] ?? "")} placeholder={option.default == null ? option.path ? "Path under backend/runs, data, models or results" : "Optional" : String(option.default)} onChange={event => setValues(previous => ({ ...previous, [option.name]: event.target.value }))} />}
          {option.help && <small>{option.help}</small>}
        </label>)}</div>
        <button className="local-primary" disabled={!connected || busy || jobs.some(item => item.status === "running")} onClick={launchCommand}>Run {selected}</button>
        <p className="local-note">Commands run with the installed local Python environment. Paths stay inside the project data and result folders. Output is saved locally.</p>
      </section>
      <section className="local-panel">
        <p className="eyebrow">EXECUTION</p><h2>Run status</h2>
        <label className="local-field">Recent jobs<select value={jobId} onChange={event => setJobId(event.target.value)}><option value="">Select a job</option>{jobs.map(item => <option key={item.id} value={item.id}>{item.id} · {item.status} · {item.command || "pipeline"}</option>)}</select></label>
        {job ? <><div className="local-status"><strong className={`local-state ${job.status}`}>{job.status}</strong><span>Step {job.step}/{job.total_steps}: {job.command || "starting"}</span></div>
          <p className="local-note">Started {new Date(job.started_at).toLocaleString()} · Results in {job.root}</p>
          {job.status === "running" && <button className="local-secondary" onClick={() => void cancel()}>Cancel run</button>}
          {job.error && <p className="local-error">{job.error}</p>}
          <pre className="local-log" aria-label="Backend output">{log || "Waiting for backend output…"}</pre></> : <p className="local-note">Start a run to see live output here.</p>}
      </section>
    </div>
    <section className="local-panel">
      <p className="eyebrow">LOCAL ARTIFACTS</p><h2>Browse measured output</h2>
      <div className="local-browser"><label>Folder <input value={directory} onChange={event => setDirectory(event.target.value)} onKeyDown={event => { if (event.key === "Enter") void listFiles(directory); }} /></label><button className="local-secondary" onClick={() => void listFiles(directory)}>Open folder</button><button className="local-secondary" onClick={() => void listFiles(directory.replace(/[\\/][^\\/]+[\\/]?$/, "") || "runs")}>Up</button></div>
      <div className="local-files">{entries.length ? entries.map(entry => <button key={entry.path} onClick={() => entry.directory ? void listFiles(entry.path) : void viewFile(entry.path)}><span>{entry.directory ? "▣" : "▤"}</span>{entry.name}{entry.bytes != null && <small>{Math.ceil(entry.bytes / 1024)} KB</small>}</button>) : <p>No files in this folder yet.</p>}</div>
      {artifact && <div className="local-artifact"><div><strong>{artifact.path}</strong>{artifact.path.endsWith("dashboard.json") && <button className="local-primary" onClick={loadSnapshot}>Show on overview</button>}</div><pre>{artifact.content}</pre></div>}
    </section>
  </div>;
}
