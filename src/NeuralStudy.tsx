import { useEffect, useState } from "react";

type Metric = { top1_macro_pct: number; top1_micro_pct: number; top5_micro_pct: number; candidate_recall_pct?: number; queries: number; transitions: number };
type Comparison = { mean: number; ci95_low: number; ci95_high: number; holm_adjusted_p: number; holm_decision: boolean };
type RuntimeMetric = { mean_ms: number; p95_ms: number; queries: number };
type Study = {
  schema_version: number; study: string; evidence: string; seeds: number[]; labels: Record<string, string>;
  models: Record<string, string>; splits: Record<string, Record<string, Metric>>;
  development: Record<string, Record<string, { validation_macro_top1_pct: number; config: { variant: string } }>>;
  comparisons: Record<string, Comparison>; runtime_comparisons: Record<string, Comparison>;
  runtime: { label: string; summary: Record<string, RuntimeMetric> }[];
  runtime_telemetry: { scenario: string; pages_prefetched: number; useful_prefetch_pct: number | null; useful_pages_per_demand_request_pct: number | null; inference_ms_per_scored_job: number | null; queue_wait_ms_per_dequeued_job: number | null; stale_jobs_dropped: number }[];
  scope: string[]; source_sha256: string;
  simulator: { decision: string; cells: number; reopened_cells: number; validation_cells: number; validation_failures: number; scope: string; bound: string };
};

const fixed = (n: number | null | undefined, digits = 2) => n == null || !Number.isFinite(n) ? "n/a" : n.toFixed(digits);
const signed = (n: number) => `${n > 0 ? "+" : ""}${fixed(n)}`;

export default function NeuralStudy() {
  const [study, setStudy] = useState<Study | null>(null);
  const [error, setError] = useState(false);
  const [split, setSplit] = useState("params");
  useEffect(() => {
    const controller = new AbortController();
    fetch("/pagewise-neural-improvement.json", { cache: "no-store", signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error("Unavailable"); return response.json(); })
      .then((payload: Study) => { if (payload.schema_version !== 1 || !payload.splits || !payload.runtime) throw new Error("Invalid study"); setStudy(payload); })
      .catch(() => { if (!controller.signal.aborted) setError(true); });
    return () => controller.abort();
  }, []);

  if (!study) return <section className="panel neural-empty"><h1>Neural improvement study</h1><p role="status">{error ? "The measured study export is unavailable. Generate public/pagewise-neural-improvement.json from the completed study artifacts." : "Loading measured study…"}</p></section>;
  const rows = study.splits[split];
  const delta = rows.selected.top1_macro_pct - rows.b2_bce.top1_macro_pct;
  const comparisons = Object.entries(study.comparisons).filter(([name]) => name.startsWith(`${split}:`));
  const b3Validation = Object.values(study.development[split]).filter(row => row.config.variant === "b3").map(row => row.validation_macro_top1_pct);

  return <div className="neural-study">
    <section className="neural-intro">
      <p className="eyebrow">FRESH CONFIRMATION / 8.62 GB SQLITE DATABASE</p>
      <h1>What improved, and where?</h1>
      <p>Validation selected the model before five new workload seeds were collected. Prediction accuracy and measured query latency are reported separately.</p>
      <div className="neural-meta"><span>Seeds {study.seeds.join(" / ")}</span><span>All eligible transitions</span><span>Frozen weights and policy</span></div>
    </section>
    <div className="neural-tabs" role="group" aria-label="Evaluation split">
      <button aria-pressed={split === "params"} onClick={() => setSplit("params")}>Unseen parameters</button>
      <button aria-pressed={split === "templates"} onClick={() => setSplit("templates")}>Unseen templates</button>
    </div>
    <div className="neural-cards">
      <article className="panel"><p className="eyebrow">SELECTED / MACRO TOP-1</p><strong>{fixed(rows.selected.top1_macro_pct)}<small>%</small></strong><p>{study.models[split]}</p></article>
      <article className="panel"><p className="eyebrow">CHANGE FROM B2 BINARY LOSS</p><strong>{signed(delta)}<small> pp</small></strong><p>Same candidate generator and split</p></article>
      <article className="panel"><p className="eyebrow">CANDIDATE RECALL / MICRO</p><strong>{fixed(rows.selected.candidate_recall_pct)}<small>%</small></strong><p>Missing targets count as errors</p></article>
    </div>
    <section className="panel neural-panel">
      <div className="panel-heading"><div><p className="eyebrow">OFFLINE PREDICTION / HIGHER IS BETTER</p><h2>Matched neural controls and native Markov baselines</h2></div></div>
      <p>Macro Top-1 gives each query equal weight. Micro Top-1 and Top-5 count page transitions. {rows.selected.queries} held-out queries / {rows.selected.transitions.toLocaleString()} transitions.</p>
      <div className="neural-table-wrap"><table><thead><tr><th>Predictor</th><th>Macro Top-1</th><th>Micro Top-1</th><th>Micro Top-5</th><th>Candidate recall</th></tr></thead>
        <tbody>{Object.entries(rows).map(([name, row]) => <tr key={name} className={name === "selected" ? "neural-selected" : ""}><th scope="row">{study.labels[name] ?? name}</th><td>{fixed(row.top1_macro_pct)}%</td><td>{fixed(row.top1_micro_pct)}%</td><td>{fixed(row.top5_micro_pct)}%</td><td>{row.candidate_recall_pct === undefined ? "n/a" : `${fixed(row.candidate_recall_pct)}%`}</td></tr>)}</tbody>
      </table></div>
      <p className="neural-note">Model B is the semantic neural model. B2 adds Markov statistics; the selected parameter model learns a semantic neural correction to a Markov prior. An epoch-zero template checkpoint is a Markov fallback with no learned neural gain.</p>
      <p>B3 adds query parameters. Its best screened hybrid reached {fixed(Math.max(...b3Validation))}% on sampled validation data for this split. It was not promoted to final testing; this validation score is separate from the fresh-test table.</p>
    </section>
    <section className="panel neural-panel">
      <p className="eyebrow">PAIRED UNCERTAINTY / FIVE WORKLOAD SEEDS</p><h2>Does the gain survive correction?</h2>
      <p>Positive effects favor the named candidate. Intervals resample workload seeds and paired queries. Holm adjustment covers the full frozen family across both splits.</p>
      <div className="neural-table-wrap"><table><thead><tr><th>Comparison</th><th>Delta pp</th><th>95% interval</th><th>Holm p</th><th>Favorable support</th></tr></thead><tbody>
        {comparisons.map(([name, row]) => <tr key={name}><th scope="row">{name.split(":")[1].replaceAll("_vs_", " vs ")}</th><td>{signed(row.mean)}</td><td>[{fixed(row.ci95_low)}, {fixed(row.ci95_high)}]</td><td>{fixed(row.holm_adjusted_p, 4)}</td><td>{row.holm_decision ? "Supported" : "Not established"}</td></tr>)}
      </tbody></table></div>
    </section>
    <section className="panel neural-panel">
      <p className="eyebrow">REAL APSW QUERY EXECUTION / LOWER IS BETTER</p><h2>End-to-end latency</h2>
      <p>Cache: 512 pages. One prefetch worker. The frozen neural policy predicts every fourth miss at confidence ≥ 0.7, degree 1. Each row contains 300 query executions across five seeds.</p>
      <div className="neural-table-wrap"><table><thead><tr><th>Scenario</th><th>No prefetch mean</th><th>Markov-2 mean</th><th>Neural mean</th><th>Neural p95</th></tr></thead><tbody>
        {study.runtime.map(scenario => <tr key={scenario.label}><th scope="row">{scenario.label}</th><td>{fixed(scenario.summary.none.mean_ms)} ms</td><td>{fixed(scenario.summary.markov2.mean_ms)} ms</td><td>{fixed(scenario.summary.selected_ranker.mean_ms)} ms</td><td>{fixed(scenario.summary.selected_ranker.p95_ms)} ms</td></tr>)}
      </tbody></table></div>
      <p className="neural-note">k is an abstract service-channel count. Delay labels are requested artificial delays; the 0.2 ms sleep calibration achieved 0.5234 ms on this host. These measurements do not establish physical disk performance.</p>
      <div className="neural-table-wrap"><table><thead><tr><th>Neural vs no prefetch</th><th>Delta ms</th><th>95% interval</th><th>Holm p</th><th>Speedup support</th></tr></thead><tbody>
        {Object.entries(study.runtime_comparisons).filter(([name]) => name.includes("selected_ranker")).map(([name, row]) => <tr key={name}><th scope="row">{name.split(":")[0]}</th><td>{signed(row.mean)}</td><td>[{fixed(row.ci95_low)}, {fixed(row.ci95_high)}]</td><td>{fixed(row.holm_adjusted_p, 4)}</td><td>{row.holm_decision ? "Supported" : "Not established"}</td></tr>)}
      </tbody></table></div>
      <h2>Prefetch efficiency and worker cost</h2>
      <div className="neural-table-wrap"><table><thead><tr><th>Scenario</th><th>Pages prefetched</th><th>Useful rate</th><th>Scoring ms/job</th><th>Queue wait ms/job</th><th>Stale jobs dropped</th></tr></thead><tbody>
        {study.runtime_telemetry.map(row => <tr key={row.scenario}><th scope="row">{row.scenario}</th><td>{row.pages_prefetched.toLocaleString()}</td><td>{fixed(row.useful_prefetch_pct)}%</td><td>{fixed(row.inference_ms_per_scored_job, 3)}</td><td>{fixed(row.queue_wait_ms_per_dequeued_job, 3)}</td><td>{row.stale_jobs_dropped.toLocaleString()}</td></tr>)}
      </tbody></table></div>
      <p className="neural-note">Useful rate divides useful pages by pages prefetched. Scoring includes candidate generation and features, excluding jobs dropped before scoring; queue wait includes all dequeued jobs. These counters help diagnose overhead and waste.</p>
    </section>
    <section className="panel neural-panel">
      <p className="eyebrow">MODEL SCOPE / EXPLORATORY SIMULATION</p><h2>Read the limits with the results</h2>
      <p>The simulator explored {study.simulator.cells} cells and reopened {study.simulator.reopened_cells} restricted-family configurations. {study.simulator.validation_failures} of {study.simulator.validation_cells} zero-delay validation cells missed the 15% error target.</p>
      <p>{study.simulator.scope} Its headroom is a {study.simulator.bound}.</p>
      <ul>{study.scope.map(note => <li key={note}>{note}</li>)}</ul>
      <p className="neural-note">Frozen predictor source: <code>{study.source_sha256.slice(0, 16)}</code>. Download the study JSON for exact values and paired effects.</p>
    </section>
  </div>;
}
