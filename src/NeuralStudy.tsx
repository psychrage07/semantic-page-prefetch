import { useEffect, useState } from "react";
import StudyDetails from "./StudyDetails";
import SemanticEvaluation from "./SemanticEvaluation";
import type { TrainedHybridSnapshot } from "./selectedModel";

type Metric = { top1_macro_pct: number; queries: number; transitions: number; candidate_recall_pct?: number };
type Study = { seeds: number[]; splits: Record<string, Record<string, Metric>>; runtime: { label: string; summary: Record<string, { mean_ms: number }> }[] };
const fixed = (value: number | undefined) => value == null || !Number.isFinite(value) ? "n/a" : value.toFixed(2);
const names: Record<string, string> = { a_bce: "Model A · page history", b_bce: "Model B · history + semantics", b_structured_listwise: "Model B · structured training", b2_bce: "B2 · semantics + Markov statistics", markov1: "Markov-1", markov2: "Markov-2", selected: "Selected neural hybrid" };
const scenarios: Record<string, string> = { "k1-delay0.0": "No injected delay · 1 channel", "k1-delay0.2": "Injected delay · 1 channel", "k4-delay0.2": "Injected delay · 4 channels", "k16-delay0.2": "Injected delay · 16 channels" };

export default function NeuralStudy({ onOpenLab, selectedResult }: { onOpenLab: () => void; selectedResult?: TrainedHybridSnapshot }) {
  const [study, setStudy] = useState<Study | null>(null);
  const [error, setError] = useState(false);
  const [split, setSplit] = useState("params");
  const [details, setDetails] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    fetch("/pagewise-neural-improvement.json", { signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error(); return response.json(); })
      .then((data: Study) => { if (!data.splits?.params?.selected || !Array.isArray(data.runtime)) throw new Error(); setStudy(data); })
      .catch(() => { if (!controller.signal.aborted) setError(true); });
    return () => controller.abort();
  }, []);
  if (!study) return <div className="faculty-page"><div className="faculty-title"><h1>Research results</h1><p role="status">{error ? "The saved study could not be loaded. No example results are substituted." : "Loading measured study…"}</p></div></div>;
  const rows = study.splits[split];
  const delta = rows.selected.top1_macro_pct - rows.markov2.top1_macro_pct;
  return <div className="faculty-page">
    <SemanticEvaluation result={selectedResult} onDemo={onOpenLab} />
    <div className="faculty-title"><p className="eyebrow">FINAL STUDY · SAVED MEASUREMENTS</p><h1>What did we achieve?</h1><p>Compare how accurately each method predicts the next database page. Runtime measurements are available below.</p><div className="study-provenance"><span>8.62 GB SQLite database</span><span>{study.seeds.length} new workload seeds</span><span>Models selected before final testing</span></div></div>
    <section className="faculty-section"><div className="faculty-section-heading"><div><p className="eyebrow">QUESTION 1</p><h2>Can we predict the next page?</h2></div><div className="faculty-segment" role="group" aria-label="Evaluation split"><button aria-pressed={split === "params"} onClick={() => setSplit("params")}>New query values</button><button aria-pressed={split === "templates"} onClick={() => setSplit("templates")}>New query types</button></div></div>
      <p className="faculty-description">{split === "params" ? "The query types are familiar, but their parameter values are held out. Our selected policy combines a Markov prior with a semantic neural correction." : "The SQL query templates are held out. The selected policy for this split is an epoch-zero Markov fallback; it has no learned neural gain."}</p>
      <div className="faculty-metrics"><article><span>{split === "params" ? "Our neural hybrid" : "Selected fallback policy"}</span><strong>{fixed(rows.selected.top1_macro_pct)}<small>%</small></strong><p>Query-average Top-1 accuracy</p></article><article><span>Difference from Markov-2</span><strong>{delta > 0 ? "+" : ""}{fixed(delta)}<small> pp</small></strong><p>Percentage-point difference on this split</p></article><article><span>Held-out evaluation</span><strong>{rows.selected.queries}</strong><p>{rows.selected.transitions.toLocaleString()} observed page transitions</p></article></div>
      <div className="faculty-table-wrap"><table className="faculty-table"><thead><tr><th>Predictor</th><th>Query-average Top-1 accuracy ↑</th></tr></thead><tbody>{Object.entries(names).map(([key, label]) => <tr key={key} className={key === "selected" ? "highlight" : ""}><th scope="row">{key === "selected" && split === "templates" ? "Selected fallback policy" : label}</th><td>{fixed(rows[key]?.top1_macro_pct)}%</td></tr>)}</tbody></table></div>
      <p className="faculty-source">Top-1 checks the first predicted page. Each query receives equal weight. B3 was screened on development data and was not selected for final testing.</p>
    </section>
    <details className="faculty-advanced"><summary><span>Runtime measurements</span><small>Query timings and prefetch overhead</small></summary><section className="faculty-section"><p className="eyebrow">QUESTION 2</p><h2>Did prefetching make queries faster?</h2><p className="faculty-description">Lower query time is better. The selected neural policy was slower than no prefetching in every measured scenario below.</p><div className="faculty-table-wrap"><table className="faculty-table"><thead><tr><th>Runtime scenario</th><th>No prefetch</th><th>Markov-2</th><th>Neural hybrid</th></tr></thead><tbody>{study.runtime.map(row => <tr key={row.label}><th scope="row">{scenarios[row.label] ?? row.label}</th><td>{fixed(row.summary.none.mean_ms)} ms</td><td>{fixed(row.summary.markov2.mean_ms)} ms</td><td>{fixed(row.summary.selected_ranker.mean_ms)} ms</td></tr>)}</tbody></table></div><p className="faculty-source">Mean query time · 300 query executions per policy/scenario · 512-page cache · one prefetch worker. Delayed scenarios request an artificial 0.2 ms delay; channels are abstract service slots, not hardware labels.</p></section></details>
    <div className="faculty-findings"><article><span className="finding-label">WHAT WORKED</span><h3>Semantics added useful signal.</h3><p>The neural hybrid improved held-out-parameter prediction accuracy over Markov-2. Detailed paired uncertainty is available below.</p></article><article><span className="finding-label">EVALUATION SCOPE</span><h3>Query type matters.</h3><p>These gains apply to the tested workloads. Unfamiliar query types remain difficult; runtime measurements have not established a query speedup.</p></article></div>
    <details className="faculty-advanced" onToggle={event => setDetails(event.currentTarget.open)}><summary><span>Full research evidence</span><small>Top-5, candidate recall, uncertainty and runtime diagnostics</small></summary>{details && <StudyDetails onOpenLab={onOpenLab} />}</details>
    <section className="presentation-next"><div><h2>Show the pipeline live.</h2><p>Execute the selected neural hybrid and inspect freshly evaluated prediction accuracy.</p></div><button className="faculty-primary" onClick={onOpenLab}>Open live demonstration →</button></section>
  </div>;
}
