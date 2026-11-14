import LatencySummary from "./LatencySummary";

export type DemoSnapshot = {
  schema_version: number;
  prediction?: { test_executions?: number; policies?: { policy: string; evaluated_next_page_predictions?: number; top1_accuracy_pct?: number | null; top5_accuracy_pct?: number | null; candidate_recall_pct?: number | null }[] };
  runtime_suites?: { strategies?: { strategy?: string; median_of_run_p50_ms?: number | null }[]; queries_per_run?: number; seeds?: number[] }[];
  selected_hybrid?: { neural_residual_trained: boolean; best_epoch: number };
  context_policy?: { scope: string };
};
const labels: Record<string, string> = { none: "No prefetch", model_ranker: "Our neural hybrid", context: "Context-filtered policy", sequential: "Sequential", markov1: "Markov-1", markov2: "Markov-2", model_a: "Model A | history", model_b: "Model B | semantics", hybrid: "Standard hybrid" };
const number = (value: number | null | undefined, suffix: string) => value == null || !Number.isFinite(value) ? "n/a" : `${value.toFixed(2)}${suffix}`;
export default function DemoResults({ snapshot, onInspect }: { snapshot: DemoSnapshot; onInspect: () => void }) {
  const suite = snapshot.runtime_suites?.[0];
  const metric = (policy: string) => snapshot.prediction?.policies?.find(row => row.policy === policy);
  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(snapshot, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = "pagewise-live-demo.json"; link.click(); URL.revokeObjectURL(url);
  }
  return <section id="demo-results" className="faculty-section demo-results" aria-label="Measured demo results"><div className="faculty-section-heading"><div><p className="eyebrow">MEASURED ACCURACY | SMALL LOCAL DATABASE</p><h2>Your demo results</h2></div><button className="local-secondary" onClick={download}>Download results</button></div>
    <p className="faculty-description">These scores use held-out queries. Our neural hybrid uses the selected architecture with new weights trained for this small database; the original research checkpoint remains separate.</p>
    {!snapshot.selected_hybrid && <p className="faculty-source">This earlier run did not train our neural hybrid. Start a new small demo to see its results.</p>}
    {snapshot.selected_hybrid && !snapshot.selected_hybrid.neural_residual_trained && <p className="faculty-source">This run selected the Markov fallback; no neural correction was trained.</p>}
    <div className="faculty-metrics"><article><span>Our hybrid Top-1 accuracy</span><strong>{number(metric("model_ranker")?.top1_accuracy_pct, "%")}</strong><p>The first guessed page was correct</p></article><article><span>Our hybrid Top-5 accuracy</span><strong>{number(metric("model_ranker")?.top5_accuracy_pct, "%")}</strong><p>The actual page was in the first five guesses</p></article><article><span>Our hybrid candidate coverage</span><strong>{number(metric("model_ranker")?.candidate_recall_pct, "%")}</strong><p>The actual next page was among the candidates</p></article></div>
    <div className="faculty-table-wrap"><table className="faculty-table"><thead><tr><th>Predictor</th><th>Top-1 accuracy</th><th>Top-5 accuracy</th></tr></thead><tbody>{Object.entries(labels).filter(([policy]) => policy !== "none").map(([policy, label]) => <tr key={policy} className={policy === "model_ranker" ? "highlight" : ""}><th scope="row">{label}</th><td>{number(metric(policy)?.top1_accuracy_pct, "%")}</td><td>{number(metric(policy)?.top5_accuracy_pct, "%")}</td></tr>)}</tbody></table></div>
    <p className="faculty-source">{snapshot.prediction?.test_executions ?? "n/a"} held-out queries | {metric("model_b")?.evaluated_next_page_predictions ?? "n/a"} evaluated transitions. Both accuracy columns weight each transition equally.</p>
    {snapshot.selected_hybrid && <p className="faculty-source">The neural hybrid uses a layout-aware candidate list of up to 128 pages; Model A and Model B use up to 16. All predictors are scored on the same held-out transitions.</p>}
    <p className="faculty-source">{snapshot.context_policy?.scope || "This saved demo predates the context-policy comparison. Run a new small demo to measure it."}</p>
    <LatencySummary title="Small demo query time" metric="Median of trial medians" entries={(suite?.strategies || []).map(row => ({ key: row.strategy || "", label: labels[row.strategy || ""] || row.strategy || "Unknown policy", value: row.median_of_run_p50_ms }))} modelKey="model_ranker" scope={`${suite?.queries_per_run ?? "n/a"} queries per trial across ${suite?.seeds?.length ?? "n/a"} workload seeds. Median of trial medians; no injected delay. The hybrid ran after the other policies; the OS cache was not reset.`} />
    <div className="demo-result-footer"><p>The standard hybrid is separate from the final selected neural-Markov hybrid.</p><button className="faculty-link" onClick={onInspect}>Inspect the page-read samples</button></div>
    <details className="runtime-details"><summary>Timing and runtime details</summary><p className="workspace-note">{suite?.queries_per_run ?? "n/a"} queries per trial | {suite?.seeds?.length ?? "n/a"} seeds | no injected delay. These timings are the median of the trial medians, separate from prediction accuracy.</p><div className="faculty-table-wrap"><table className="faculty-table"><thead><tr><th>Policy</th><th>Median query time</th></tr></thead><tbody>{Object.entries(labels).filter(([policy]) => policy !== "none" && policy !== "stock").map(([policy, label]) => <tr key={policy}><th scope="row">{label}</th><td>{number(suite?.strategies?.find(row => row.strategy === policy)?.median_of_run_p50_ms, " ms")}</td></tr>)}</tbody></table></div></details>
  </section>;
}
