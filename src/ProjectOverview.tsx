import { useEffect, useState } from "react";

type Metric = { top1_macro_pct: number };
type Summary = { splits: { params: Record<string, Metric> }; seeds: number[] };

export default function ProjectOverview({ onDemo, onResults, onMethod }: { onDemo: () => void; onResults: () => void; onMethod: () => void }) {
  const [study, setStudy] = useState<Summary | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    fetch("/pagewise-neural-improvement.json", { signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error(); return response.json(); })
      .then((data: Summary) => { if (!data.splits?.params?.selected || !data.splits.params.markov2) throw new Error(); setStudy(data); })
      .catch(() => { if (!controller.signal.aborted) setFailed(true); });
    return () => controller.abort();
  }, []);
  const rows = study?.splits.params;
  const score = (key: string) => rows?.[key]?.top1_macro_pct;
  const fixed = (value: number | undefined) => value == null ? "—" : value.toFixed(2);
  const improvement = rows ? rows.selected.top1_macro_pct - rows.markov2.top1_macro_pct : undefined;
  const models = [["a_bce", "Model A", "Page history"], ["b_bce", "Model B", "History + semantics"], ["markov1", "Markov-1", "Previous page"], ["markov2", "Markov-2", "Previous two pages"], ["selected", "Our neural hybrid", "Markov prior + neural correction"]];

  return <div className="faculty-page">
    <section className="project-hero">
      <div><span className="presentation-tag">PAGEWISE · SQLITE PAGE PREFETCHING</span><h1>Predict the next page.<br /><em>Measure the real cost.</em></h1><p>SQLite reads data in pages. Pagewise predicts which page a query will need next and fetches it early, then checks whether that work makes the query faster.</p><div className="presentation-actions"><button className="faculty-primary" onClick={onDemo}>Run a live demonstration <span aria-hidden="true">↗</span></button><button className="faculty-link" onClick={onMethod}>See how it works →</button></div></div>
      <div className="project-model"><span className="eyebrow">OUR SELECTED APPROACH</span><h2>Neural–Markov hybrid</h2><p>Combine familiar page patterns with database context.</p><div className="model-input"><span>01</span><div><strong>Page history</strong><small>Which pages did the query read?</small></div></div><div className="model-input"><span>02</span><div><strong>Database semantics</strong><small>Which table, index, page type and query plan?</small></div></div><div className="model-output">Markov prior + learned neural correction</div><small className="model-caption">Selected on validation data before final testing.</small></div>
    </section>
    <div className="presentation-flow" aria-label="System flow">{[["1", "Observe", "Trace actual SQLite page reads"], ["2", "Predict", "Rank candidate next pages"], ["3", "Prefetch", "Read early into a bounded cache"], ["4", "Measure", "Compare accuracy and query time"]].map(([step, title, text]) => <div key={step}><span>{step}</span><div><strong>{title}</strong><small>{text}</small></div></div>)}</div>
    <section className="faculty-section"><div className="faculty-section-heading"><div><p className="eyebrow">FINAL STUDY · SAVED MEASUREMENTS</p><h2>Did the predictions improve?</h2></div><button className="faculty-link" onClick={onResults}>Explore the evidence →</button></div><p className="faculty-description">Results below use new query values on the same tested database and query types. Each query contributes equally to Top-1 accuracy.</p>
      {failed ? <p role="status" className="faculty-callout">The study export could not be loaded. No example scores are substituted.</p> : <><div className="faculty-metrics"><article><span>Our neural hybrid</span><strong>{fixed(score("selected"))}<small>%</small></strong><p>Query-average Top-1 accuracy</p></article><article><span>Markov-2 baseline</span><strong>{fixed(score("markov2"))}<small>%</small></strong><p>The strongest native Markov baseline tested</p></article><article><span>Improvement over Markov-2</span><strong>{improvement == null ? "—" : `+${fixed(improvement)}`}<small> pp</small></strong><p>Percentage points, on this evaluation</p></article></div>
      <div className="faculty-bars" aria-label="Measured query-average Top-1 accuracy">{models.map(([key, label, description]) => <div className={`faculty-bar-row ${key === "selected" ? "selected" : ""}`} key={key}><div><strong>{label}</strong><small>{description}</small></div><div className="faculty-bar-track"><span style={{ width: `${score(key) ?? 0}%` }} /></div><b>{fixed(score(key))}%</b></div>)}</div><p className="faculty-source">8.62 GB database · {study?.seeds.length ?? "—"} held-out workload seeds · {study ? "Measured study from 20 October 2026" : "Loading measured results…"}</p></>}
    </section>
    <div className="faculty-findings"><article><span className="finding-label">PREDICTION RESULT</span><h3>Semantic information helped.</h3><p>Model B improved over history-only Model A. The selected hybrid improved further on familiar query types with new values.</p></article><article><span className="finding-label">RUNTIME RESULT</span><h3>A speedup was not established.</h3><p>In the measured runtime tests, prediction and prefetch overhead outweighed the benefit. New query types also remain difficult.</p></article></div>
    <section className="presentation-next"><div><p className="eyebrow">NEXT: SHOW THE SYSTEM RUNNING</p><h2>From database to measured results.</h2><p>Run the selected trained hybrid on its matching database, or build a small development experiment.</p></div><button className="faculty-primary" onClick={onDemo}>Open live demonstration →</button></section>
  </div>;
}
