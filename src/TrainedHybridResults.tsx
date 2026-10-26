import SemanticEvaluation from "./SemanticEvaluation";
import type { TrainedHybridSnapshot } from "./selectedModel";
export type { TrainedHybridSnapshot } from "./selectedModel";
const fixed = (value: number | null | undefined) => value == null ? "n/a" : value.toFixed(2);

export default function TrainedHybridResults({ result, showEvaluation = true }: { result: TrainedHybridSnapshot; showEvaluation?: boolean }) {
  const model = result.rows.find(row => row.policy === "model_ranker");
  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = "pagewise-trained-hybrid-live.json"; link.click(); URL.revokeObjectURL(url);
  }
  return <section id="demo-results" className="faculty-section demo-results" aria-label="Trained hybrid live results">
    <div className="faculty-section-heading"><div><p className="eyebrow">TRAINED MODEL | MEASURED PREDICTION ACCURACY</p><h2>Selected neural hybrid: live results</h2></div><button className="local-secondary" onClick={download}>Download results</button></div>
    <p className="faculty-description">The trained semantic model predicted next pages on its matching database. Accuracy checks those predictions against the observed next pages. No retraining took place.</p>
    {showEvaluation && <SemanticEvaluation result={result} />}
    <p className="faculty-source">{model?.scored_jobs ?? "n/a"} background model-scoring jobs | {result.queries_per_seed} queries per policy/seed | {result.seeds.length} workload seeds | 8.62 GB database.</p>
    {result.prediction_samples.length > 0 && <div className="hybrid-predictions"><p className="eyebrow">ACTUAL MODEL OUTPUT | CAPPED SAMPLE</p><h3>What did the hybrid predict?</h3>{result.prediction_samples.map((sample, index) => <div key={index}><span>Read history: {sample.history.join(" -> ")}</span><strong>Next candidates: {sample.candidates.map(candidate => `page ${candidate.page} (${candidate.score.toFixed(3)})`).join(" -> ")}</strong></div>)}<p className="faculty-source">Candidate scores rank possible pages. Accuracy is calculated separately from actual outcomes.</p></div>}
    <details className="runtime-details"><summary>Timing and prefetch details</summary><p className="workspace-note">These are separately measured runtime outcomes, not prediction accuracies. Better accuracy has not established a query speedup. No injected delay; diagnostics are enabled and the OS cache is not reset.</p><div className="faculty-table-wrap"><table className="faculty-table"><thead><tr><th>Policy</th><th>Mean query time</th><th>Median query time</th><th>Cache hit rate</th><th>Unused entries</th><th>Inference / scored job</th></tr></thead><tbody>{result.rows.map(row => <tr key={row.policy}><th scope="row">{row.label}</th><td>{fixed(row.mean_ms)} ms</td><td>{fixed(row.median_ms)} ms</td><td>{fixed(row.cache_hit_pct)}%</td><td>{row.unused_pages ?? "n/a"}</td><td>{row.inference_ms_per_job == null ? "n/a" : `${fixed(row.inference_ms_per_job)} ms`}</td></tr>)}</tbody></table></div></details>
    <details className="demo-log"><summary>Model verification and workload provenance</summary><p className="local-note">Frozen model: {result.variant}<br />Checkpoint SHA-256: {result.checkpoint_sha256}<br />Database: {result.database_bytes.toLocaleString()} bytes<br />Workload seeds: {result.seeds.join(", ")}<br />Paired workloads verified: {result.workloads_paired ? "yes" : "no"}</p></details>
  </section>;
}
