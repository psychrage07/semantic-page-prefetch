type LatencyEntry = { key: string; label: string; value: number | null | undefined };
type Props = {
  title?: string;
  metric: string;
  entries: LatencyEntry[];
  modelKey: string;
  referenceKey?: string;
  comparable?: boolean;
  scope: string;
};

const valid = (value: number | null | undefined): value is number =>
  value != null && Number.isFinite(value) && value >= 0;
const format = (value: number | null | undefined) => valid(value) ? `${value.toFixed(3)} ms` : "n/a";

export default function LatencySummary({ title = "Measured query time", metric, entries, modelKey, comparable = true, scope }: Props) {
  const model = entries.find(entry => entry.key === modelKey);
  const markov = entries.find(entry => entry.key === "markov2");
  const context = entries.find(entry => entry.key === "context");
  return <div className="latency-summary" aria-label={title}>
    <div className="latency-heading"><div><p className="eyebrow">ACTUAL QUERY TIMING</p><h3>{title}</h3></div><span>{metric}</span></div>
    <div className="latency-metrics">
      <article><span>{model?.label || "Model"}</span><strong>{format(model?.value)}</strong></article>
      <article><span>{markov?.label || "Markov-2"}</span><strong>{format(markov?.value)}</strong></article>
      {context && <article><span>{context.label}</span><strong>{format(context.value)}</strong></article>}
    </div>
    <p className="faculty-source">{comparable ? "Times for the displayed prediction policies; accuracy is measured separately." : "The workload results were not verified as paired, so these times cannot support a direct comparison."} Complete measurements are available in the downloaded results.</p>
    <p className="faculty-source">{scope}</p>
  </div>;
}
