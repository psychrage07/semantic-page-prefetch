export type DemoSnapshot = {
  schema_version: number;
  prediction?: { test_executions?: number; policies?: { policy: string; top1_accuracy_pct?: number | null }[] };
  runtime_suites?: { strategies?: { strategy?: string; median_of_run_p50_ms?: number | null }[]; queries_per_run?: number; seeds?: number[] }[];
};

const labels: Record<string, string> = { none: "No prefetch", sequential: "Sequential", markov1: "Markov-1", markov2: "Markov-2", model_a: "Model A · history", model_b: "Model B · semantics", hybrid: "Standard hybrid" };
const number = (value: number | null | undefined, suffix: string) => value == null || !Number.isFinite(value) ? "n/a" : `${value.toFixed(2)}${suffix}`;

export default function DemoResults({ snapshot, onInspect }: { snapshot: DemoSnapshot; onInspect: () => void }) {
  const suite = snapshot.runtime_suites?.[0];
  const accuracy = (policy: string) => snapshot.prediction?.policies?.find(row => row.policy === policy)?.top1_accuracy_pct;
  const latency = (policy: string) => suite?.strategies?.find(row => row.strategy === policy)?.median_of_run_p50_ms;
  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(snapshot, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = "pagewise-live-demo.json"; link.click(); URL.revokeObjectURL(url);
  }
  return <section id="demo-results" className="faculty-section demo-results" aria-label="Measured demo results"><div className="faculty-section-heading"><div><p className="eyebrow">MEASURED OUTPUT · SMALL LOCAL DATABASE</p><h2>Your demo results</h2></div><button className="local-secondary" onClick={download}>Download results</button></div>
    <p className="faculty-description">These values were calculated from this run. They verify the development pipeline; they are separate from the final neural hybrid study.</p>
    <div className="faculty-metrics"><article><span>Model B prediction accuracy</span><strong>{number(accuracy("model_b"), "%")}</strong><p>Top-1 across held-out page transitions</p></article><article><span>No prefetch query time</span><strong>{number(latency("none"), "")}<small> ms</small></strong><p>Median of the trial median query times</p></article><article><span>Model B query time</span><strong>{number(latency("model_b"), "")}<small> ms</small></strong><p>Compare with no prefetch; lower is better</p></article></div>
    <div className="faculty-table-wrap"><table className="faculty-table"><thead><tr><th>Policy</th><th>Top-1 accuracy ↑</th><th>Median query time ↓</th></tr></thead><tbody>{Object.entries(labels).map(([policy, label]) => <tr key={policy} className={policy === "model_b" ? "highlight" : ""}><th scope="row">{label}</th><td>{policy === "none" ? "Not a predictor" : number(accuracy(policy), "%")}</td><td>{number(latency(policy), " ms")}</td></tr>)}</tbody></table></div>
    <div className="demo-result-footer"><p>{snapshot.prediction?.test_executions ?? "—"} held-out queries · {suite?.queries_per_run ?? "—"} queries per runtime trial · {suite?.seeds?.length ?? "—"} runtime seeds · no injected delay</p><button className="faculty-link" onClick={onInspect}>Inspect the page-read samples →</button></div>
    <p className="faculty-source">The standard hybrid in this small demo is a separate policy from the final selected neural–Markov hybrid.</p>
  </section>;
}
