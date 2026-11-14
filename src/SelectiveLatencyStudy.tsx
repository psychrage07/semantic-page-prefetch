import { useEffect, useState } from "react";

type Interval = { ci95_low: number; ci95_high: number };
type Policy = { mean_ms: number; queries: number; vs_none?: Interval; counters: { pages_prefetched: number; useful_prefetches: number } };
type Accuracy = { top1_macro_pct: number; candidate_recall_pct: number; queries: number; transitions: number };
type Study = { seeds: number[]; scenarios: Record<string, { runtime: { policies: Record<string, Policy> }; offline_accuracy: { policies: Record<string, Accuracy> } }> };
const fixed = (value: number | undefined) => value == null || !Number.isFinite(value) ? "n/a" : value.toFixed(2);
const labels: Record<string, string> = { none: "No prefetch", markov2: "Markov-2", model_ranker: "Selected neural hybrid", context: "Context-filtered experiment", deadline: "Deadline experiment", adaptive: "Adaptive experiment", combined: "Combined experiment" };

export default function SelectiveLatencyStudy({ compact = false, onDetails }: { compact?: boolean; onDetails?: () => void }) {
  const [study, setStudy] = useState<Study | null>(null);
  const [scenario, setScenario] = useState("zero");
  useEffect(() => {
    const controller = new AbortController();
    fetch("/pagewise-selective-prefetch.json", { signal: controller.signal })
      .then(response => response.ok ? response.json() : Promise.reject())
      .then((data: Study) => setStudy(data)).catch(() => {});
    return () => controller.abort();
  }, []);
  if (!study) return <section className="faculty-section"><h2>Held-out comparison</h2><p>Results unavailable.</p></section>;
  const policies = study.scenarios[scenario].runtime.policies;
  const accuracy = study.scenarios[scenario].offline_accuracy.policies;
  const neural = accuracy.model_ranker, markov = accuracy.markov2;
  const renderRows = (keys: string[]) => keys.map(key => { const row = policies[key]; return row && <tr key={key} className={key === "model_ranker" ? "highlight" : ""}><th scope="row">{labels[key]}</th><td>{fixed(row.mean_ms)} ms</td><td>{row.counters.pages_prefetched.toLocaleString()}</td><td>{row.counters.useful_prefetches.toLocaleString()}</td></tr>; });
  return <section className="faculty-section selective-latency" aria-label="Held-out prediction and runtime results">
    <div className="faculty-section-heading"><div><p className="eyebrow">8.62 GB DATABASE | HELD-OUT COMPARISON</p><h2>Prediction quality and runtime</h2></div>{onDetails && <button className="faculty-link" onClick={onDetails}>See research results</button>}</div>
    <p className="faculty-description">On these held-out queries, the neural hybrid improved prediction quality. Query times below compare prediction policies; complete controls are available in the downloaded report.</p>
    <div className="faculty-table-wrap"><table className="faculty-table"><thead><tr><th>Predictor</th><th>Query-average Top-1</th><th>Candidate coverage</th></tr></thead><tbody><tr><th scope="row">Markov-2</th><td>{fixed(markov?.top1_macro_pct)}%</td><td>{fixed(markov?.candidate_recall_pct)}%</td></tr><tr className="highlight"><th scope="row">Selected neural hybrid</th><td>{fixed(neural?.top1_macro_pct)}%</td><td>{fixed(neural?.candidate_recall_pct)}%</td></tr></tbody></table></div>
    <p className="faculty-source">{neural.queries} held-out queries | {neural.transitions.toLocaleString()} next-page transitions | Top-1 weights each query equally; candidate coverage weights transitions equally. Offline predictions are separate from runtime scheduling.</p>
    <h3>Mean query time</h3>
    <div className="faculty-segment latency-scenarios" role="group" aria-label="Timing condition"><button aria-pressed={scenario === "zero"} onClick={() => setScenario("zero")}>No added delay</button><button aria-pressed={scenario === "controlled-delay"} onClick={() => setScenario("controlled-delay")}>Artificial I/O delay</button></div>
    <div className="faculty-table-wrap"><table className="faculty-table"><thead><tr><th>Policy</th><th>Mean query time</th><th>Pages prefetched</th><th>Useful pages</th></tr></thead><tbody>{renderRows(["markov2", "model_ranker", "context"])}</tbody></table></div>
    <p className="faculty-source">{policies.none.queries} matched queries per policy across {study.seeds.length} seeds. {scenario === "zero" ? "No artificial delay was added; SQLite and OS caching still apply." : "The experiment requests a 0.2 ms artificial service delay on a shared single-channel model; this is not a physical-drive benchmark."} The context experiment did not use neural fallback.</p>
    {!compact && <details className="runtime-details"><summary>Experimental controls: deadline, adaptive and combined</summary><p className="faculty-source">These policies prefetched zero pages in this evaluation. They are ablations, not evidence of a successful prefetch speedup.</p><div className="faculty-table-wrap"><table className="faculty-table"><thead><tr><th>Experiment</th><th>Mean query time</th><th>Pages prefetched</th><th>Useful pages</th></tr></thead><tbody>{renderRows(["deadline", "adaptive", "combined"])}</tbody></table></div></details>}
    <p className="faculty-source">Full uncertainty intervals, CPU measurements and methods: <a href="/selective-prefetch-results.md" download>measured report</a> or <a href="/pagewise-selective-prefetch.json" download>results JSON</a>.</p>
  </section>;
}
