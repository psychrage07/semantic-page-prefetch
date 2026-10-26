export type DemoSnapshot = {
  schema_version: number;
  prediction?: { test_executions?: number; policies?: { policy: string; evaluated_next_page_predictions?: number; top1_accuracy_pct?: number | null; top5_accuracy_pct?: number | null; candidate_recall_pct?: number | null }[] };
  runtime_suites?: { strategies?: { strategy?: string; median_of_run_p50_ms?: number | null }[]; queries_per_run?: number; seeds?: number[] }[];
};
const labels: Record<string, string> = { none: "No prefetch", sequential: "Sequential", markov1: "Markov-1", markov2: "Markov-2", model_a: "Model A | history", model_b: "Model B | semantics", hybrid: "Standard hybrid" };
const number = (value: number | null | undefined, suffix: string) => value == null || !Number.isFinite(value) ? "n/a" : `${value.toFixed(2)}${suffix}`;
export default function DemoResults({ snapshot, onInspect }: { snapshot: DemoSnapshot; onInspect: () => void }) {
  const suite = snapshot.runtime_suites?.[0];
  const metric = (policy: string) => snapshot.prediction?.policies?.find(row => row.policy === policy);
  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(snapshot, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = "pagewise-live-demo.json"; link.click(); URL.revokeObjectURL(url);
  }
  return <section id="demo-results" className="faculty-section demo-results" aria-label="Measured demo results"><div className="faculty-section-heading"><div><p className="eyebrow">MEASURED ACCURACY | SMALL LOCAL DATABASE</p><h2>Your demo results</h2></div><button className="local-secondary" onClick={download}>Download results</button></div>
    <p className="faculty-description">These prediction scores come from this run's held-out query executions. This small development model is separate from the final neural hybrid study.</p>
    <div className="faculty-metrics"><article><span>Model B Top-1 accuracy</span><strong>{number(metric("model_b")?.top1_accuracy_pct, "%")}</strong><p>The first guessed page was correct</p></article><article><span>Model B Top-5 accuracy</span><strong>{number(metric("model_b")?.top5_accuracy_pct, "%")}</strong><p>The actual page was in the first five guesses</p></article><article><span>Model B candidate coverage</span><strong>{number(metric("model_b")?.candidate_recall_pct, "%")}</strong><p>The actual next page was among the candidates</p></article></div>
    <div className="faculty-table-wrap"><table className="faculty-table"><thead><tr><th>Predictor</th><th>Top-1 accuracy</th><th>Top-5 accuracy</th></tr></thead><tbody>{Object.entries(labels).filter(([policy]) => policy !== "none").map(([policy, label]) => <tr key={policy} className={policy === "model_b" ? "highlight" : ""}><th scope="row">{label}</th><td>{number(metric(policy)?.top1_accuracy_pct, "%")}</td><td>{number(metric(policy)?.top5_accuracy_pct, "%")}</td></tr>)}</tbody></table></div>
    <p className="faculty-source">{snapshot.prediction?.test_executions ?? "n/a"} held-out queries | {metric("model_b")?.evaluated_next_page_predictions ?? "n/a"} evaluated transitions. Both accuracy columns weight each transition equally. No prefetch makes no predictions.</p>
    <div className="demo-result-footer"><p>The standard hybrid is separate from the final selected neural-Markov hybrid.</p><button className="faculty-link" onClick={onInspect}>Inspect the page-read samples</button></div>
    <details className="runtime-details"><summary>Timing and runtime details</summary><p className="workspace-note">{suite?.queries_per_run ?? "n/a"} queries per trial | {suite?.seeds?.length ?? "n/a"} seeds | no injected delay. These timings are the median of the trial medians, separate from prediction accuracy.</p><div className="faculty-table-wrap"><table className="faculty-table"><thead><tr><th>Policy</th><th>Median query time</th></tr></thead><tbody>{Object.entries(labels).map(([policy, label]) => <tr key={policy}><th scope="row">{label}</th><td>{number(suite?.strategies?.find(row => row.strategy === policy)?.median_of_run_p50_ms, " ms")}</td></tr>)}</tbody></table></div></details>
  </section>;
}
