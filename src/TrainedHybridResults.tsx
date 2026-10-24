export type TrainedHybridSnapshot = {
  artifact_type: "trained_hybrid_live"; checkpoint_sha256: string; variant: string; database_bytes: number;
  seeds: number[]; queries_per_seed: number; scope: string; workloads_paired: boolean;
  rows: { policy: string; label: string; queries: number; mean_ms: number; median_ms: number; scored_jobs: number; pages_prefetched: number; useful_pages: number }[];
  prediction_samples: { query_id: string; seed: number; history: number[]; candidates: { page: number; score: number }[] }[];
};
const fixed = (value: number | undefined) => value == null ? "n/a" : value.toFixed(2);

export default function TrainedHybridResults({ result }: { result: TrainedHybridSnapshot }) {
  const model = result.rows.find(row => row.policy === "model_ranker");
  const baseline = result.rows.find(row => row.policy === "none");
  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = "pagewise-trained-hybrid-live.json"; link.click(); URL.revokeObjectURL(url);
  }
  return <section id="demo-results" className="faculty-section demo-results" aria-label="Trained hybrid live results">
    <div className="faculty-section-heading"><div><p className="eyebrow">TRAINED MODEL · FRESH LOCAL EXECUTION</p><h2>Selected neural hybrid: live results</h2></div><button className="local-secondary" onClick={download}>Download results</button></div>
    <p className="faculty-description">The existing selected checkpoint was loaded and used for actual inference on its matching database. No training took place. The timing values below were measured in this run.</p>
    <div className="faculty-metrics"><article><span>Selected hybrid inference</span><strong>{model?.scored_jobs ?? "n/a"}</strong><p>Prediction jobs actually scored by the model</p></article><article><span>No prefetch query time</span><strong>{fixed(baseline?.mean_ms)}<small> ms</small></strong><p>Mean across the paired live workloads</p></article><article><span>Selected hybrid query time</span><strong>{fixed(model?.mean_ms)}<small> ms</small></strong><p>Lower is better; a speedup is not assumed</p></article></div>
    <div className="faculty-table-wrap"><table className="faculty-table"><thead><tr><th>Policy</th><th>Mean query time ↓</th><th>Median query time ↓</th><th>Prefetched pages</th><th>Useful pages</th></tr></thead><tbody>{result.rows.map(row => <tr key={row.policy} className={row.policy === "model_ranker" ? "highlight" : ""}><th scope="row">{row.label}</th><td>{fixed(row.mean_ms)} ms</td><td>{fixed(row.median_ms)} ms</td><td>{row.pages_prefetched}</td><td>{row.useful_pages}</td></tr>)}</tbody></table></div>
    <p className="faculty-source">{result.queries_per_seed} queries × 3 policies × {result.seeds.length} new workload seeds · 8.62 GB database · no injected delay · diagnostics enabled equally for all policies.</p>
    <p className="demo-scope">This is a live inference and runtime demonstration. It does not recompute the final study’s held-out accuracy; generated values may overlap training. Diagnostic logging differs from the final runtime study, and the OS cache is not reset.</p>
    {result.prediction_samples.length > 0 && <div className="hybrid-predictions"><p className="eyebrow">ACTUAL MODEL OUTPUT · CAPPED SAMPLE</p><h3>What did the hybrid predict?</h3>{result.prediction_samples.map((sample, index) => <div key={index}><span>Read history: {sample.history.join(" → ")}</span><strong>Next candidates: {sample.candidates.map(candidate => `page ${candidate.page} (${candidate.score.toFixed(3)})`).join(" · ")}</strong></div>)}<p className="faculty-source">Scores rank candidates; they do not guarantee correctness or that a prefetch arrived in time.</p></div>}
    <details className="demo-log"><summary>Model verification and workload provenance</summary><p className="local-note">Frozen model: {result.variant}<br />Checkpoint SHA-256: {result.checkpoint_sha256}<br />Database: {result.database_bytes.toLocaleString()} bytes<br />Workload seeds: {result.seeds.join(", ")}<br />Paired workloads verified: {result.workloads_paired ? "yes" : "no"}</p></details>
  </section>;
}
