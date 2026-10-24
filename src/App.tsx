import { useEffect, useMemo, useState, type ReactNode } from "react";
import NeuralStudy from "./NeuralStudy";
import LocalLab from "./LocalLab";
import ProjectOverview from "./ProjectOverview";
import HowItWorks from "./HowItWorks";
import SemanticEvaluation from "./SemanticEvaluation";
import TrainedHybridResults from "./TrainedHybridResults";
import { useSelectedRun } from "./selectedModel";
import InteractiveWorkspace from "./InteractiveWorkspace";

type View = "overview" | "traces" | "comparison" | "method" | "neural" | "lab" | "workspace" | "preview" | "protocol";
type WorkloadId = "point" | "join" | "traversal" | "range" | "mixed";
type ModeId = "nvme" | "sata" | "hdd";
type PolicyId = "stock" | "sequential" | "markov" | "modelA" | "modelB" | "oracle";

type Workload = {
  label: string;
  family: string;
  description: string;
  reads: number;
  template: string;
  p50: Record<PolicyId, number>;
  hitRate: Record<PolicyId, number>;
  useful: Record<PolicyId, number>;
  wasted: Record<PolicyId, number>;
  top1: Record<PolicyId, number>;
};

type TraceEvent = {
  id: number;
  time: string;
  queryId: string;
  workload: WorkloadId;
  template: string;
  page: number;
  previousPage: number;
  object: string;
  pageType: "leaf" | "interior" | "overflow";
  depth: number;
  plan: string;
  cache: "hit" | "miss";
  predictedPage: number;
  confidence: number;
  prefetched: boolean;
  useful: boolean;
};

type DashboardPolicyMetric = {
  policy: string;
  evaluated_next_page_predictions?: number | null;
  top1_accuracy_pct?: number | null;
  top5_accuracy_pct?: number | null;
  candidate_recall_pct?: number | null;
};

type DashboardSuite = {
  name: string;
  mode?: string | null;
  queries_per_run?: number | null;
  seeds?: number[] | null;
  strategies?: Record<string, unknown>[];
  paired_query_comparisons_vs_none?: Record<string, unknown>[];
  best_mean_delta_vs_none?: Record<string, unknown> | null;
  query_latency_series?: {
    strategy: string;
    trial?: number | null;
    seed?: number | null;
    points: {
      index: number;
      query_id?: string | null;
      template_id?: string | null;
      workload_family?: string | null;
      plan_class?: string | null;
      elapsed_ms?: number | null;
    }[];
  }[];
};

type DashboardSnapshot = {
  schema_version: number;
  source_run_dir?: string;
  prediction?: {
    test_executions?: number | null;
    top_k?: number | null;
    policies?: DashboardPolicyMetric[];
  };
  multi_seed_training?: {
    training_seeds?: number[] | null;
    fixed_split_seed?: number | null;
    epochs?: number | null;
    aggregate?: Record<string, unknown>[];
  } | null;
  runtime_suites?: DashboardSuite[];
  measured_samples?: {
    suite: string;
    strategy: string;
    trace_events: {
      seq?: number | null;
      query_id?: string | null;
      template_id?: string | null;
      workload_family?: string | null;
      plan_class?: string | null;
      plan_details?: string[] | null;
      page?: number | null;
      last_page?: number | null;
      offset?: number | null;
      amount?: number | null;
      duration_us?: number | null;
      cache_status?: "hit" | "miss" | string | null;
      underlying_bytes?: number | null;
      source?: string | null;
    }[];
    predictions: {
      prediction_id?: number | null;
      query_id?: string | null;
      template_id?: string | null;
      workload_family?: string | null;
      policy?: string | null;
      history?: number[] | null;
      inference_and_prefetch_ms?: number | null;
      queue_wait_ms?: number | null;
      ranked_candidates?: { page?: number; score?: number }[];
      selected?: { page?: number; score?: number; status?: string; bytes?: number }[];
    }[];
  } | null;
  sources?: string[];
};

const POLICY_INFO: { id: PolicyId; label: string; short: string; color: string }[] = [
  { id: "stock", label: "No prefetch", short: "Stock", color: "#9ca69e" },
  { id: "sequential", label: "Next-N", short: "Next-N", color: "#7895a4" },
  { id: "markov", label: "Markov-2", short: "Markov", color: "#cf8a60" },
  { id: "modelA", label: "Model A / history", short: "Model A", color: "#687d9d" },
  { id: "modelB", label: "Model B / semantic", short: "Model B", color: "#52745f" },
  { id: "oracle", label: "Oracle ceiling", short: "Oracle", color: "#b0bd78" },
];

const WORKLOADS: Record<WorkloadId, Workload> = {
  point: {
    label: "Indexed point lookups",
    family: "Point lookup",
    description: "Index seek followed by a table-row fetch across unseen keys.",
    reads: 6_840,
    template: "SELECT * FROM users WHERE user_id = ?",
    p50: { stock: 8.7, sequential: 7.5, markov: 6.9, modelA: 7.1, modelB: 6.8, oracle: 4.4 },
    hitRate: { stock: 31, sequential: 37, markov: 46, modelA: 44, modelB: 48, oracle: 76 },
    useful: { stock: 0, sequential: 27, markov: 52, modelA: 47, modelB: 55, oracle: 100 },
    wasted: { stock: 0, sequential: 38, markov: 25, modelA: 29, modelB: 23, oracle: 0 },
    top1: { stock: 0, sequential: 12.4, markov: 39.7, modelA: 42.8, modelB: 46.3, oracle: 100 },
  },
  join: {
    label: "Indexed nested-loop joins",
    family: "Multi-table join",
    description: "Orders joined to line items and products through secondary indexes.",
    reads: 12_420,
    template: "SELECT ... FROM orders JOIN order_items USING (order_id) ...",
    p50: { stock: 24.6, sequential: 22.9, markov: 18.8, modelA: 19.4, modelB: 17.6, oracle: 11.2 },
    hitRate: { stock: 28, sequential: 33, markov: 51, modelA: 49, modelB: 56, oracle: 82 },
    useful: { stock: 0, sequential: 22, markov: 57, modelA: 53, modelB: 63, oracle: 100 },
    wasted: { stock: 0, sequential: 43, markov: 22, modelA: 26, modelB: 18, oracle: 0 },
    top1: { stock: 0, sequential: 11.6, markov: 43.2, modelA: 45.1, modelB: 50.7, oracle: 100 },
  },
  traversal: {
    label: "Data-dependent traversal",
    family: "Tree traversal",
    description: "Recursive parent-to-child walks with variable fan-out and path length.",
    reads: 18_760,
    template: "WITH RECURSIVE walk AS (...) SELECT * FROM walk",
    p50: { stock: 31.4, sequential: 28.2, markov: 22.8, modelA: 23.6, modelB: 19.5, oracle: 9.8 },
    hitRate: { stock: 19, sequential: 25, markov: 48, modelA: 46, modelB: 57, oracle: 86 },
    useful: { stock: 0, sequential: 19, markov: 58, modelA: 56, modelB: 69, oracle: 100 },
    wasted: { stock: 0, sequential: 49, markov: 24, modelA: 27, modelB: 16, oracle: 0 },
    top1: { stock: 0, sequential: 9.2, markov: 41.8, modelA: 43.4, modelB: 51.6, oracle: 100 },
  },
  range: {
    label: "Indexed range scans",
    family: "Range control",
    description: "Mostly sequential leaf-page reads, included as a readahead control.",
    reads: 9_180,
    template: "SELECT * FROM orders WHERE created_at BETWEEN ? AND ?",
    p50: { stock: 13.5, sequential: 10.9, markov: 10.1, modelA: 10.3, modelB: 10.7, oracle: 8.5 },
    hitRate: { stock: 41, sequential: 62, markov: 65, modelA: 64, modelB: 63, oracle: 78 },
    useful: { stock: 0, sequential: 72, markov: 68, modelA: 65, modelB: 62, oracle: 100 },
    wasted: { stock: 0, sequential: 15, markov: 17, modelA: 18, modelB: 20, oracle: 0 },
    top1: { stock: 0, sequential: 61.2, markov: 54.3, modelA: 51.8, modelB: 51.2, oracle: 100 },
  },
  mixed: {
    label: "Mixed application traffic",
    family: "Mixed workload",
    description: "Interleaved lookups, joins, scans, and recursive account-history reads.",
    reads: 22_540,
    template: "12 parameterized templates, interleaved by execution",
    p50: { stock: 27.8, sequential: 25.3, markov: 20.4, modelA: 20.8, modelB: 19.7, oracle: 12.6 },
    hitRate: { stock: 24, sequential: 29, markov: 49, modelA: 47, modelB: 53, oracle: 84 },
    useful: { stock: 0, sequential: 20, markov: 55, modelA: 52, modelB: 61, oracle: 100 },
    wasted: { stock: 0, sequential: 46, markov: 24, modelA: 27, modelB: 19, oracle: 0 },
    top1: { stock: 0, sequential: 10.8, markov: 42.6, modelA: 44.2, modelB: 49.1, oracle: 100 },
  },
};

const MODES: Record<ModeId, { label: string; delay: number; factor: number; short: string }> = {
  nvme: { label: "NVMe-like / 100 us", delay: 0.1, factor: 0.62, short: "100 us" },
  sata: { label: "SATA-like / 1 ms", delay: 1, factor: 1, short: "1 ms" },
  hdd: { label: "HDD-like / 8 ms", delay: 8, factor: 2.2, short: "8 ms" },
};

const chartPolicies: PolicyId[] = ["stock", "markov", "modelA", "modelB"];
const objectNames = ["users_pk", "orders_created", "order_items_order", "products_pk", "account_edges", "sessions_user"];
const planNames: Record<WorkloadId, string> = {
  point: "SEARCH users USING INTEGER PRIMARY KEY",
  join: "SEARCH order_items USING INDEX order_items_order",
  traversal: "RECURSIVE STEP / SEARCH account_edges",
  range: "SEARCH orders USING INDEX orders_created",
  mixed: "SEARCH sessions USING INDEX sessions_user",
};

const TRACE_EVENTS: TraceEvent[] = Array.from({ length: 96 }, (_, index) => {
  const workload = (["point", "join", "traversal", "range", "mixed"] as WorkloadId[])[index % 5];
  const page = 12 + ((index * 37 + Math.floor(index / 6) * 13) % 489);
  const useful = index % 6 === 0 || index % 11 === 0 || index % 13 === 0;
  const prefetched = index % 4 !== 1;
  const pageType = index % 13 === 0 ? "overflow" : index % 4 === 0 ? "interior" : "leaf";

  return {
    id: index + 1,
    time: `+${(index * 0.37).toFixed(2)} ms`,
    queryId: `Q${String(701 + Math.floor(index / 6)).padStart(4, "0")}`,
    workload,
    template: WORKLOADS[workload].template,
    page,
    previousPage: Math.max(1, page - (index % 7) * 3 - 1),
    object: objectNames[index % objectNames.length],
    pageType,
    depth: pageType === "interior" ? 2 + (index % 3) : pageType === "overflow" ? 4 : 3 + (index % 2),
    plan: planNames[workload],
    cache: index % 3 === 0 ? "miss" : "hit",
    predictedPage: useful ? page : page + (index % 17) + 1,
    confidence: Number((0.56 + ((index * 17) % 42) / 100).toFixed(2)),
    prefetched,
    useful: prefetched && useful,
  };
});

function formatPctValue(value: unknown, digits = 1) {
  return typeof value === "number" ? `${value.toFixed(digits)}%` : "n/a";
}

function formatDeltaMs(value: unknown) {
  if (typeof value !== "number") return "n/a";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(3)} ms`;
}

function numericValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function latencyChange(baseline: number | null, candidate: number | null) {
  return baseline != null && baseline > 0 && candidate != null ? ((baseline - candidate) / baseline) * 100 : null;
}

function policyLabel(policy: string) {
  const labels: Record<string, string> = {
    none: "No prefetch",
    stock: "Stock SQLite",
    sequential: "Next-N",
    markov1: "Markov-1",
    markov2: "Markov-2",
    model_a: "Model A",
    model_b: "Model B",
    oracle: "Oracle",
    oracle_miss: "Oracle miss control",
    null: "Null control",
    hybrid: "Hybrid",
  };
  return labels[policy] ?? policy.replaceAll("_", " ");
}

function measuredPolicy(snapshot: DashboardSnapshot | null, policy: string) {
  return snapshot?.prediction?.policies?.find((item) => item.policy === policy) ?? null;
}

function measuredSuite(snapshot: DashboardSnapshot | null, preferredName = "final-latency-0p2ms") {
  return snapshot?.runtime_suites?.find((suite) => suite.name === preferredName) ?? snapshot?.runtime_suites?.[0] ?? null;
}

function measuredStrategy(suite: DashboardSuite | null, strategy: string) {
  return suite?.strategies?.find((row) => row.strategy === strategy) ?? null;
}

function strategyForPolicy(policyId: PolicyId) {
  const map: Record<PolicyId, string> = {
    stock: "none",
    sequential: "sequential",
    markov: "markov2",
    modelA: "model_a",
    modelB: "model_b",
    oracle: "oracle",
  };
  return map[policyId];
}

function measuredSeriesForPolicy(suite: DashboardSuite | null, policyId: PolicyId) {
  const strategy = strategyForPolicy(policyId);
  return suite?.query_latency_series?.find((line) => line.strategy === strategy) ?? null;
}

function measuredMetricFor(snapshot: DashboardSnapshot | null, policyId: PolicyId) {
  return measuredMetricForStrategy(snapshot, strategyForPolicy(policyId));
}

function measuredMetricForStrategy(snapshot: DashboardSnapshot | null, strategyName: string) {
  if (!snapshot) return null;
  const suite = measuredSuite(snapshot);
  const strategy = measuredStrategy(suite, strategyName);
  const offline = measuredPolicy(snapshot, strategyName);
  return {
    p50: numericValue(strategy?.median_of_run_p50_ms),
    p95: numericValue(strategy?.median_of_run_p95_ms),
    hitRate: numericValue(strategy?.mean_cache_hit_pct),
    useful: numericValue(strategy?.mean_useful_prefetches),
    wasted: numericValue(strategy?.mean_wasted_prefetches_evicted),
    top1: numericValue(offline?.top1_accuracy_pct),
  };
}

function measuredTraceEvents(snapshot: DashboardSnapshot | null): TraceEvent[] {
  const predictionsByQuery = new Map<string, NonNullable<DashboardSnapshot["measured_samples"]>["predictions"][number]>();
  snapshot?.measured_samples?.predictions?.forEach((prediction) => {
    if (prediction.query_id && !predictionsByQuery.has(prediction.query_id)) {
      predictionsByQuery.set(prediction.query_id, prediction);
    }
  });
  const sourceRows = snapshot?.measured_samples?.trace_events ?? [];
  return sourceRows.map((event, index) => {
    const prediction = event.query_id ? predictionsByQuery.get(event.query_id) : undefined;
    const selected = prediction?.selected?.find((item) => item.status === "prefetched") ?? prediction?.selected?.[0];
    const page = event.page ?? 0;
    return {
      id: Number(event.seq ?? index + 1),
      time: typeof event.duration_us === "number" ? `${event.duration_us.toFixed(1)} us` : "+0 us",
      queryId: event.query_id ?? "unknown",
      workload: "mixed",
      template: event.template_id ?? "unknown",
      page,
      previousPage: Math.max(0, page - 1),
      object: event.source ?? "demand",
      pageType: "leaf",
      depth: 0,
      plan: event.plan_details?.[0] ?? event.plan_class ?? "unknown",
      cache: event.cache_status === "hit" ? "hit" : "miss",
      predictedPage: selected?.page ?? page,
      confidence: typeof selected?.score === "number" ? selected.score : 0,
      prefetched: event.source === "prefetch",
      useful: event.source === "prefetch" || selected?.page === page,
    };
  });
}

function metricFor(workloadId: WorkloadId, modeId: ModeId, cacheMb: number, policyId: PolicyId, degree = 4) {
  const workload = WORKLOADS[workloadId];
  const cacheFactor = cacheMb === 128 ? 1.12 : cacheMb === 512 ? 0.92 : 1;
  const factor = MODES[modeId].factor * cacheFactor;
  const degreeEffects: Record<number, { latency: number; useful: number; wasted: number; hitRate: number }> = {
    1: { latency: 0.42, useful: -7, wasted: -6, hitRate: -2 },
    2: { latency: 0.16, useful: -3, wasted: -3, hitRate: -1 },
    4: { latency: 0, useful: 0, wasted: 0, hitRate: 0 },
    8: { latency: 0.58, useful: 3, wasted: 10, hitRate: 1 },
  };
  const change = policyId === "stock" || policyId === "oracle" ? { latency: 0, useful: 0, wasted: 0, hitRate: 0 } : degreeEffects[degree];
  const p50 = workload.p50[policyId] * factor + change.latency * MODES[modeId].factor;
  const p95Factor: Record<PolicyId, number> = {
    stock: 1.78,
    sequential: 1.69,
    markov: 1.58,
    modelA: 1.64,
    modelB: 1.55,
    oracle: 1.4,
  };
  return {
    p50,
    p95: p50 * p95Factor[policyId],
    hitRate: Math.max(0, Math.min(100, workload.hitRate[policyId] + change.hitRate)),
    useful: Math.max(0, Math.min(100, workload.useful[policyId] + change.useful)),
    wasted: Math.max(0, Math.min(100, workload.wasted[policyId] + change.wasted)),
    top1: workload.top1[policyId],
  };
}

function formatMs(value: number | null | undefined) {
  return value != null ? `${value.toFixed(1)} ms` : "n/a";
}

function downloadFile(filename: string, content: string, type = "text/plain") {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.7,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true as const,
  };

  switch (name) {
    case "overview":
      return <svg {...common}><rect x="3.5" y="3.5" width="7" height="7" rx="1" /><rect x="13.5" y="3.5" width="7" height="7" rx="1" /><rect x="3.5" y="13.5" width="7" height="7" rx="1" /><rect x="13.5" y="13.5" width="7" height="7" rx="1" /></svg>;
    case "trace":
      return <svg {...common}><path d="M5 4.5h14M5 9.5h14M5 14.5h8M5 19.5h11" /><circle cx="3.5" cy="4.5" r=".7" fill="currentColor" /><circle cx="3.5" cy="9.5" r=".7" fill="currentColor" /><circle cx="3.5" cy="14.5" r=".7" fill="currentColor" /><circle cx="3.5" cy="19.5" r=".7" fill="currentColor" /></svg>;
    case "compare":
      return <svg {...common}><path d="M4 19.5h16M6.5 16V10M12 16V5M17.5 16v-8" /><circle cx="6.5" cy="8" r="1" /><circle cx="12" cy="3" r="1" /><circle cx="17.5" cy="6" r="1" /></svg>;
    case "method":
      return <svg {...common}><path d="M6 3.5h8l4 4v13H6z" /><path d="M14 3.5v5h4M9 13h6M9 17h6" /></svg>;
    case "download":
      return <svg {...common}><path d="M12 3.5v11M8 10.5l4 4 4-4M5 20.5h14" /></svg>;
    case "play":
      return <svg {...common}><path d="m8 5.5 11 6.5-11 6.5z" fill="currentColor" stroke="none" /></svg>;
    case "arrow":
      return <svg {...common}><path d="M4 12h15M13 6l6 6-6 6" /></svg>;
    case "search":
      return <svg {...common}><circle cx="10.8" cy="10.8" r="6.5" /><path d="m16 16 4.5 4.5" /></svg>;
    case "database":
      return <svg {...common}><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7" /></svg>;
    case "filter":
      return <svg {...common}><path d="M4 5h16M7 12h10M10 19h4" /></svg>;
    case "check":
      return <svg {...common}><path d="m5 12.5 4.3 4.2L19.5 6.8" /></svg>;
    case "close":
      return <svg {...common}><path d="m6 6 12 12M18 6 6 18" /></svg>;
    case "layers":
      return <svg {...common}><path d="m12 3 9 5-9 5-9-5zM3 12l9 5 9-5M3 16l9 5 9-5" /></svg>;
    default:
      return <svg {...common}><circle cx="12" cy="12" r="8" /></svg>;
  }
}

function LogoMark() {
  return (
    <span className="logo-mark" aria-hidden="true">
      <i /><i /><i />
    </span>
  );
}

function DemoNotice({ onOpenMethod, snapshot }: { onOpenMethod: () => void; snapshot: DashboardSnapshot | null }) {
  return (
    <div className="demo-notice">
      <span className="notice-mark" aria-hidden="true" />
      <div className="notice-copy">
        <strong>{snapshot ? "Measured artifact loaded" : "Fixture mode"}</strong>
        <span>{snapshot ? `Backend snapshot from ${snapshot.runtime_suites?.length ?? 0} runtime suite(s) and ${snapshot.prediction?.policies?.length ?? 0} predictor rows. Unavailable measurements are shown as n/a.` : "This preview uses illustrative browser data. Open Local experiments to run real SQLite measurements."}</span>
      </div>
      <button className="text-action" onClick={onOpenMethod}>
        Implementation boundary <Icon name="arrow" size={15} />
      </button>
    </div>
  );
}

function MeasuredBackendPanel({ snapshot, local }: { snapshot: DashboardSnapshot | null; local: boolean }) {
  if (!snapshot) return null;
  const suite = measuredSuite(snapshot);
  const modelB = measuredPolicy(snapshot, "model_b");
  const modelA = measuredPolicy(snapshot, "model_a");
  const markov2 = measuredPolicy(snapshot, "markov2");
  const noneRuntime = measuredStrategy(suite, "none");
  const modelBRuntime = measuredStrategy(suite, "model_b");
  const bestDelta = suite?.best_mean_delta_vs_none ?? null;
  const topPolicies = [modelB, modelA, markov2].filter(Boolean) as DashboardPolicyMetric[];

  return (
    <section className="measured-panel panel" aria-label="Measured backend snapshot">
      <div className="panel-heading">
        <div><p className="eyebrow">MEASURED BACKEND SNAPSHOT</p><h2>Latest experiment artifact</h2></div>
        <span className="measured-badge">JSON LOADED</span>
      </div>
      <div className="measured-grid">
        <div className="measured-card">
          <span>Model B top-1</span>
          <strong>{formatPctValue(modelB?.top1_accuracy_pct)}</strong>
          <small>Top-{snapshot.prediction?.top_k ?? 5}: {formatPctValue(modelB?.top5_accuracy_pct)}</small>
        </div>
        <div className="measured-card">
          <span>Candidate recall</span>
          <strong>{formatPctValue(modelB?.candidate_recall_pct)}</strong>
          <small>{modelB?.evaluated_next_page_predictions?.toLocaleString() ?? "n/a"} held-out predictions</small>
        </div>
        <div className="measured-card">
          <span>{suite?.name ?? "Runtime suite"}</span>
          <strong>{formatDeltaMs(bestDelta?.mean_delta_ms)}</strong>
          <small>{policyLabel(String(bestDelta?.strategy ?? "best"))} vs no prefetch</small>
        </div>
        <div className="measured-card">
          <span>Model B p50</span>
          <strong>{typeof modelBRuntime?.median_of_run_p50_ms === "number" ? `${modelBRuntime.median_of_run_p50_ms.toFixed(2)} ms` : "n/a"}</strong>
          <small>No prefetch: {typeof noneRuntime?.median_of_run_p50_ms === "number" ? `${noneRuntime.median_of_run_p50_ms.toFixed(2)} ms` : "n/a"}</small>
        </div>
      </div>
      <div className="measured-table-wrap">
        <table className="measured-table">
          <thead><tr><th>Predictor</th><th>Top-1</th><th>Top-5</th><th>Recall</th></tr></thead>
          <tbody>
            {topPolicies.map((policy) => (
              <tr key={policy.policy}>
                <td>{policyLabel(policy.policy)}</td>
                <td>{formatPctValue(policy.top1_accuracy_pct)}</td>
                <td>{formatPctValue(policy.top5_accuracy_pct)}</td>
                <td>{formatPctValue(policy.candidate_recall_pct)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="chart-footnote">Loaded from {local ? "a local experiment artifact" : <code>/pagewise-dashboard.json</code>}. Fixture controls remain illustrative; measured values come only from the loaded snapshot.</p>
    </section>
  );
}

function PageIntro({
  eyebrow,
  title,
  description,
  trailing,
}: {
  eyebrow: string;
  title: string;
  description: string;
  trailing?: ReactNode;
}) {
  return (
    <div className="page-intro">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p className="intro-copy">{description}</p>
      </div>
      {trailing && <div className="intro-trailing">{trailing}</div>}
    </div>
  );
}

function LatencyChart({
  workloadId,
  modeId,
  cacheMb,
  degree,
  visible,
  setVisible,
  snapshot,
}: {
  workloadId: WorkloadId;
  modeId: ModeId;
  cacheMb: number;
  degree: number;
  visible: PolicyId[];
  setVisible: (policies: PolicyId[]) => void;
  snapshot: DashboardSnapshot | null;
}) {
  const width = 740;
  const height = 250;
  const margin = { top: 13, right: 16, bottom: 31, left: 44 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const waves = [0.05, -0.08, 0.02, 0.16, -0.11, 0.07, -0.04, 0.2, -0.13, 0.03, 0.12, -0.07, 0.04];
  const suite = measuredSuite(snapshot);
  const sequence = useMemo(() => {
    if (snapshot) {
      return chartPolicies.map((policyId) => {
        const measured = measuredSeriesForPolicy(suite, policyId);
        return {
          policyId,
          values: (measured?.points ?? []).map((point) => point.elapsed_ms).filter((value): value is number => typeof value === "number").slice(0, 60),
        };
      }).filter((line) => line.values.length > 0);
    }
    return chartPolicies.map((policyId, policyIndex) => {
      const baseline = metricFor(workloadId, modeId, cacheMb, policyId, degree).p50;
      const values = waves.map((wave, index) => {
        const detail = Math.sin((index + 1) * (policyIndex + 2)) * 0.045;
        return Math.max(baseline * 0.45, baseline * (1 + wave + detail));
      });
      return { policyId, values };
    });
  }, [workloadId, modeId, cacheMb, degree, suite, snapshot]);
  const maxPoints = Math.max(1, ...sequence.map((line) => line.values.length));
  const maxValue = Math.ceil(Math.max(0, ...sequence.flatMap((line) => line.values)) / 5) * 5 + 5;
  const y = (value: number) => margin.top + plotHeight - (value / maxValue) * plotHeight;
  const x = (index: number) => margin.left + (index / Math.max(1, maxPoints - 1)) * plotWidth;

  function toggle(policyId: PolicyId) {
    setVisible(visible.includes(policyId) ? visible.filter((item) => item !== policyId) : [...visible, policyId]);
  }

  return (
    <section className="chart-panel panel" aria-labelledby="latency-title">
      <div className="panel-heading chart-heading">
        <div>
          <p className="eyebrow">{snapshot ? "MEASURED QUERY LATENCIES" : "ILLUSTRATIVE REPLAY PROFILE"}</p>
          <h2 id="latency-title">Query latency</h2>
        </div>
        <div className="unit-label"><span className="unit-dot" /> {snapshot ? "query elapsed time" : "p50 estimate"} <span>/ ms</span></div>
      </div>
      <div className="chart-plot" role="img" aria-label={snapshot ? "Measured query latency comparison" : "Illustrative query latency comparison"}>
        {snapshot && sequence.length === 0 ? <p className="empty-state">No measured query latency series available in this snapshot.</p> :
        <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
          {[0, 1, 2, 3, 4].map((tick) => {
            const value = (maxValue / 4) * tick;
            const tickY = y(value);
            return (
              <g key={tick}>
                <line x1={margin.left} x2={width - margin.right} y1={tickY} y2={tickY} className="chart-gridline" />
                <text x={margin.left - 9} y={tickY + 3} className="chart-axis-text" textAnchor="end">{value.toFixed(0)}</text>
              </g>
            );
          })}
          {[0, Math.floor((maxPoints - 1) * 0.25), Math.floor((maxPoints - 1) * 0.5), Math.floor((maxPoints - 1) * 0.75), maxPoints - 1].map((tick) => (
            <text key={tick} x={x(tick)} y={height - 7} className="chart-axis-text" textAnchor="middle">
              {suite ? tick + 1 : tick * 10}
            </text>
          ))}
          {sequence.filter((line) => visible.includes(line.policyId)).map((line) => {
            const policy = POLICY_INFO.find((item) => item.id === line.policyId)!;
            const path = line.values.map((value, index) => `${index === 0 ? "M" : "L"}${x(index)},${y(value)}`).join(" ");
            return (
              <g key={line.policyId} className="chart-series">
                <path d={path} fill="none" stroke={policy.color} strokeWidth={line.policyId === "modelB" ? 2.7 : 2.1} strokeLinecap="round" strokeLinejoin="round" />
                {line.values.filter((_, index) => index === 0 || index === Math.floor((line.values.length - 1) / 2) || index === line.values.length - 1).map((value, pointIndex) => {
                  const index = [0, Math.floor((line.values.length - 1) / 2), line.values.length - 1][pointIndex];
                  return <circle key={`${line.policyId}-${index}`} cx={x(index)} cy={y(value)} r="3" fill={policy.color} />;
                })}
              </g>
            );
          })}
        </svg>}
      </div>
      <div className="chart-footer">
        <div className="chart-legend" aria-label="Toggle chart series">
          {sequence.map(({ policyId }) => {
            const policy = POLICY_INFO.find((item) => item.id === policyId)!;
            return (
              <button key={policyId} className={`legend-button ${visible.includes(policyId) ? "is-on" : ""}`} onClick={() => toggle(policyId)} aria-pressed={visible.includes(policyId)}>
                <span className="legend-line" style={{ backgroundColor: policy.color }} />{policy.short}
              </button>
            );
          })}
        </div>
        <span className="axis-caption">query sequence</span>
      </div>
      <p className="chart-footnote">{snapshot ? sequence.length ? `Measured query latencies from ${suite?.name}; capped to first ${maxPoints} queries per strategy. Only available series are shown.` : "This snapshot has no query latency series for the displayed policies." : "Deterministic fixture curve. Replace with query-level SQLite timings for measured results."}</p>
    </section>
  );
}

function ReplayControls({
  workloadId,
  setWorkloadId,
  modeId,
  setModeId,
  cacheMb,
  setCacheMb,
  degree,
  setDegree,
  onRun,
}: {
  workloadId: WorkloadId;
  setWorkloadId: (value: WorkloadId) => void;
  modeId: ModeId;
  setModeId: (value: ModeId) => void;
  cacheMb: number;
  setCacheMb: (value: number) => void;
  degree: number;
  setDegree: (value: number) => void;
  onRun: () => void;
}) {
  return (
    <section className="run-panel panel" aria-labelledby="run-heading">
      <div className="panel-heading run-panel-heading">
        <div>
          <p className="eyebrow">EXPERIMENT CONFIG</p>
          <h2 id="run-heading">Replay settings</h2>
        </div>
        <span className="ready-state"><span /> fixture ready</span>
      </div>
      <label className="field-label" htmlFor="workload">Workload</label>
      <select id="workload" value={workloadId} onChange={(event) => setWorkloadId(event.target.value as WorkloadId)}>
        {Object.entries(WORKLOADS).map(([id, workload]) => <option key={id} value={id}>{workload.label}</option>)}
      </select>
      <label className="field-label" htmlFor="storage-mode">Miss latency profile</label>
      <select id="storage-mode" value={modeId} onChange={(event) => setModeId(event.target.value as ModeId)}>
        {Object.entries(MODES).map(([id, mode]) => <option key={id} value={id}>{mode.label}</option>)}
      </select>
      <div className="field-pair">
        <div>
          <label className="field-label" htmlFor="cache-size">App cache</label>
          <select id="cache-size" value={cacheMb} onChange={(event) => setCacheMb(Number(event.target.value))}>
            {[128, 256, 512].map((value) => <option key={value} value={value}>{value} MB</option>)}
          </select>
        </div>
        <div>
          <label className="field-label" htmlFor="prefetch-degree">Prefetch degree</label>
          <select id="prefetch-degree" value={degree} onChange={(event) => setDegree(Number(event.target.value))}>
            {[1, 2, 4, 8].map((value) => <option key={value} value={value}>{value} pages</option>)}
          </select>
        </div>
      </div>
      <div className="run-setup-note">
        <Icon name="layers" size={16} />
        <span>Same fixture query order, cache size, and LRU policy across every strategy.</span>
      </div>
      <button className="run-button" type="button" onClick={onRun}>
        <Icon name="play" size={17} />
        <span>Run a real local experiment</span>
        <span className="button-arrow"><Icon name="arrow" size={16} /></span>
      </button>
      <p className="run-caption">The controls above preview the fixture. The local experiment workspace runs actual SQLite queries with its own measured settings.</p>
    </section>
  );
}

function MetricStrip({
  workloadId,
  modeId,
  cacheMb,
  degree,
  activePolicy,
  snapshot,
}: {
  workloadId: WorkloadId;
  modeId: ModeId;
  cacheMb: number;
  degree: number;
  activePolicy: PolicyId;
  snapshot: DashboardSnapshot | null;
}) {
  const measuredSelected = measuredMetricFor(snapshot, activePolicy);
  const measuredStock = measuredMetricFor(snapshot, "stock");
  const selected = metricFor(workloadId, modeId, cacheMb, activePolicy, degree);
  const stock = metricFor(workloadId, modeId, cacheMb, "stock", degree);
  const selectedMetrics = snapshot ? measuredSelected : selected;
  const selectedP50 = selectedMetrics?.p50 ?? null;
  const stockP50 = snapshot ? measuredStock?.p50 ?? null : stock.p50;
  const delta = latencyChange(stockP50, selectedP50);
  return (
    <section className="metric-strip" aria-label={snapshot ? "Measured current policy metrics" : "Illustrative current policy metrics"}>
      <div className="metric-item metric-primary">
        <span>{POLICY_INFO.find((policy) => policy.id === activePolicy)?.label} / p50</span>
        <strong>{formatMs(selectedP50)}</strong>
        <small>{delta == null ? "Runtime comparison unavailable" : `${Math.abs(delta).toFixed(1)}% ${delta >= 0 ? "below" : "above"} no prefetch`}</small>
      </div>
      <div className="metric-item">
        <span>p95 latency</span>
        <strong>{formatMs(selectedMetrics?.p95)}</strong>
        <small>{snapshot ? "available suite measurement" : "same fixture run"}</small>
      </div>
      <div className="metric-item">
        <span>App-cache hit rate</span>
        <strong>{formatPctValue(selectedMetrics?.hitRate)}</strong>
        <small>bounded LRU cache</small>
      </div>
      <div className="metric-item">
        <span>Useful prefetches</span>
        <strong>{selectedMetrics?.useful != null ? snapshot ? selectedMetrics.useful.toFixed(0) : `${selectedMetrics.useful}%` : "n/a"}</strong>
        <small>{selectedMetrics?.wasted != null ? snapshot ? `${selectedMetrics.wasted.toFixed(0)} wasted` : `${selectedMetrics.wasted}% marked wasted` : "Waste measurement unavailable"}</small>
      </div>
    </section>
  );
}

function ResultsTable({
  workloadId,
  modeId,
  cacheMb,
  degree,
  activePolicy,
  onSelect,
  snapshot,
  compact = false,
}: {
  workloadId: WorkloadId;
  modeId: ModeId;
  cacheMb: number;
  degree: number;
  activePolicy: PolicyId;
  onSelect: (policy: PolicyId) => void;
  snapshot: DashboardSnapshot | null;
  compact?: boolean;
}) {
  const displayPolicies: PolicyId[] = compact || snapshot ? ["stock", "sequential", "markov", "modelA", "modelB"] : ["stock", "sequential", "markov", "modelA", "modelB", "oracle"];
  const measuredBaseline = measuredMetricFor(snapshot, "stock");
  const baseline = snapshot ? measuredBaseline?.p50 ?? null : metricFor(workloadId, modeId, cacheMb, "stock", degree).p50;
  const rows = displayPolicies.map((policyId) => {
    const policy = POLICY_INFO.find((item) => item.id === policyId)!;
    return { strategy: strategyForPolicy(policyId), policyId: policyId as PolicyId | null, label: policy.label, color: policy.color };
  });
  if (snapshot) {
    for (const row of measuredSuite(snapshot)?.strategies ?? []) {
      if (typeof row.strategy === "string" && !rows.some((item) => item.strategy === row.strategy)) {
        rows.push({ strategy: row.strategy, policyId: null, label: policyLabel(row.strategy), color: "#687466" });
      }
    }
  }

  return (
    <div className="results-table-wrap">
      <table className={`results-table ${compact ? "compact-table" : ""}`}>
        <thead><tr><th>Strategy</th><th>p50</th><th>p95</th><th>Cache hit</th><th>Useful</th><th>Wasted</th>{!compact && <th>Top-1</th>}</tr></thead>
        <tbody>
          {rows.map((row) => {
            const policyId = row.policyId;
            const metrics = snapshot ? measuredMetricForStrategy(snapshot, row.strategy) : policyId ? metricFor(workloadId, modeId, cacheMb, policyId, degree) : null;
            const p50 = metrics?.p50 ?? null;
            const change = latencyChange(baseline, p50);
            return (
              <tr key={row.strategy} className={`${activePolicy === policyId ? "selected-row" : ""} ${policyId === "oracle" ? "oracle-row" : ""}`} onClick={policyId ? () => onSelect(policyId) : undefined} tabIndex={policyId ? 0 : undefined} onKeyDown={policyId ? (event) => { if (event.key === "Enter" || event.key === " ") onSelect(policyId); } : undefined}>
                <td><span className="strategy-name"><i style={{ backgroundColor: row.color }} />{row.label}</span></td>
                <td className="number-cell">{formatMs(p50)}{policyId !== "stock" && change != null && <small className={`delta ${change >= 0 ? "delta-down" : "delta-up"}`}>{change >= 0 ? "-" : "+"}{Math.abs(change).toFixed(1)}%</small>}</td>
                <td className="number-cell">{formatMs(metrics?.p95)}</td>
                <td className="number-cell">{formatPctValue(metrics?.hitRate, snapshot ? 1 : 0)}</td>
                <td className="number-cell">{policyId === "stock" || metrics?.useful == null ? "n/a" : snapshot ? metrics.useful.toFixed(0) : `${metrics.useful}%`}</td>
                <td className="number-cell">{policyId === "stock" || metrics?.wasted == null ? "n/a" : snapshot ? metrics.wasted.toFixed(0) : `${metrics.wasted}%`}</td>
                {!compact && <td className="number-cell">{formatPctValue(metrics?.top1)}</td>}
              </tr>
            );
          })}
        </tbody>
      </table>
      {snapshot && <p className="chart-footnote">n/a means a measurement is unavailable. Offline accuracy is reported independently of runtime availability.</p>}
    </div>
  );
}

function PredictionHeatmap({ workloadId, snapshot }: { workloadId: WorkloadId; snapshot: DashboardSnapshot | null }) {
  const measured = measuredTraceEvents(snapshot);
  const events = (snapshot ? measured : TRACE_EVENTS.filter((event) => event.workload === workloadId)).slice(0, 36);
  if (snapshot && events.length === 0) {
    return <section className="heatmap-panel"><h2>Prediction alignment</h2><p className="empty-state">No measured trace samples available in this snapshot.</p></section>;
  }
  const rows = [
    { label: "Demand", key: "demand" },
    { label: "Model B", key: "model" },
  ] as const;

  return (
    <section className="heatmap-panel" aria-labelledby="heatmap-title">
      <div className="section-title-row">
        <div>
          <p className="eyebrow">PAGE-LEVEL VIEW</p>
          <h2 id="heatmap-title">Prediction alignment</h2>
        </div>
        <span className="subtle-count">{events.length} {measured.length ? "measured reads" : "fixture reads"}</span>
      </div>
      <p className="section-description">{measured.length ? "Each column is an observed VFS read from the exported Model B run sample." : "Each column is a page demand. Model B cells show an illustrative candidate arriving before demand, missing it, or being absent."}</p>
      <div className="heatmap-wrap">
        {rows.map((row) => (
          <div className="heatmap-row" key={row.key}>
            <span className="heatmap-label">{row.label}</span>
            <div className="heatmap-cells" style={{ gridTemplateColumns: `repeat(${events.length}, minmax(0, 1fr))` }}>
              {events.map((event, index) => {
                const type = row.key === "demand" ? "demand-cell" : event.useful ? "match-cell" : event.prefetched ? "waste-cell" : "miss-cell";
                const tooltip = row.key === "demand" ? `Demand page ${event.page}, ${event.object}` : event.useful ? `Correct prefetch for page ${event.page}` : event.prefetched ? `Wasted candidate ${event.predictedPage}; demanded page ${event.page}` : `No timely candidate for page ${event.page}`;
                return <span key={event.id} className={`heat-cell ${type}`} title={`Q${index + 1}: ${tooltip}`} />;
              })}
            </div>
          </div>
        ))}
      </div>
      <div className="heatmap-legend"><span><i className="legend-demand" />demand</span><span><i className="legend-match" />useful</span><span><i className="legend-waste" />wasted</span><span><i className="legend-absent" />not prefetched</span></div>
      <p className="chart-footnote">{measured.length ? "Measured trace sample loaded from the backend dashboard snapshot." : "Generated fixture sequence; colors do not represent observed VFS events."}</p>
    </section>
  );
}

function AblationChart() {
  const values = [
    { label: "History only (Model A)", value: 43.1 },
    { label: "+ object identity", value: 47.5 },
    { label: "+ page type", value: 49.2 },
    { label: "+ B-tree depth", value: 48.6 },
    { label: "+ plan node", value: 51.4 },
    { label: "+ template + position (Model B)", value: 50.8 },
  ];
  return (
    <section className="ablation-panel panel">
      <div className="panel-heading">
        <div><p className="eyebrow">FEATURE ABLATION / TOP-1</p><h2>What context carries signal?</h2></div>
        <span className="fixture-tag">ILLUSTRATIVE</span>
      </div>
      <p className="section-description">A sample shape for the ablation report, not a measured result. Each feature set must be retrained with matched seeds and budgets.</p>
      <div className="ablation-bars">
        {values.map((item, index) => (
          <div className={`ablation-row ${index === 0 || index === values.length - 1 ? "ablation-emphasis" : ""}`} key={item.label}>
            <span className="ablation-label">{item.label}</span>
            <span className="bar-track"><i style={{ width: `${(item.value / 60) * 100}%`, animationDelay: `${index * 60}ms` }} /></span>
            <strong>{item.value.toFixed(1)}%</strong>
          </div>
        ))}
      </div>
      <div className="ablation-axis"><span>0%</span><span>30%</span><span>60%</span></div>
    </section>
  );
}

function TraceExplorer({
  workloadFilter,
  setWorkloadFilter,
  search,
  setSearch,
  onlyWasted,
  setOnlyWasted,
  selectedId,
  setSelectedId,
  snapshot,
}: {
  workloadFilter: WorkloadId | "all";
  setWorkloadFilter: (value: WorkloadId | "all") => void;
  search: string;
  setSearch: (value: string) => void;
  onlyWasted: boolean;
  setOnlyWasted: (value: boolean) => void;
  selectedId: number;
  setSelectedId: (value: number) => void;
  snapshot: DashboardSnapshot | null;
}) {
  const measured = measuredTraceEvents(snapshot);
  const sourceEvents = snapshot ? measured : TRACE_EVENTS;
  const filteredEvents = useMemo(() => {
    const normalized = search.trim().toLowerCase();
    return sourceEvents.filter((event) => {
      const matchesWorkload = measured.length || workloadFilter === "all" || event.workload === workloadFilter;
      const matchesWasted = !onlyWasted || (event.prefetched && !event.useful);
      const searchTarget = `${event.page} ${event.object} ${event.plan} ${event.template} ${event.queryId}`.toLowerCase();
      return matchesWorkload && matchesWasted && (!normalized || searchTarget.includes(normalized));
    });
  }, [workloadFilter, search, onlyWasted, measured.length, sourceEvents]);
  const selected = filteredEvents.find((event) => event.id === selectedId) ?? filteredEvents[0] ?? null;

  if (snapshot && sourceEvents.length === 0) {
    return <><PageIntro eyebrow="TRACE INSPECTOR / MEASURED SNAPSHOT" title="Follow a page through the query." description="The loaded snapshot does not include page-level trace samples." /><section className="panel"><p className="empty-state">No measured trace samples available in this snapshot.</p></section></>;
  }

  return (
    <>
      <PageIntro eyebrow={measured.length ? "TRACE INSPECTOR / MEASURED VFS SAMPLE" : "TRACE INSPECTOR / FIXTURE EVENTS"} title="Follow a page through the query." description={measured.length ? "Inspect sampled xRead events from the exported Model B backend run, with page, query, plan, source, and cache status." : "Inspect query reads, page metadata, candidate predictions, and cache outcomes. The rows below are generated fixtures, not xRead events."} trailing={<span className="intro-count">{sourceEvents.length} <small>sample events</small></span>} />
      <section className="trace-controls panel" aria-label="Trace filters">
        <label className="search-field"><Icon name="search" size={17} /><input aria-label="Search trace rows" placeholder="Search page, object, query plan..." value={search} onChange={(event) => setSearch(event.target.value)} /></label>
        {!measured.length && <label className="filter-select"><Icon name="filter" size={16} /><select aria-label="Filter by workload" value={workloadFilter} onChange={(event) => setWorkloadFilter(event.target.value as WorkloadId | "all")}><option value="all">All workloads</option>{Object.entries(WORKLOADS).map(([id, workload]) => <option key={id} value={id}>{workload.family}</option>)}</select></label>}
        {!measured.length && <label className="wasted-toggle"><input type="checkbox" checked={onlyWasted} onChange={(event) => setOnlyWasted(event.target.checked)} /><span className="toggle-box"><Icon name="check" size={13} /></span>Wasted prefetches only</label>}
        <span className="filtered-count">{filteredEvents.length} / {sourceEvents.length} rows</span>
      </section>
      <div className="trace-layout">
        <section className="trace-table-panel panel" aria-label="Trace events">
          <div className="trace-table-top"><span className="eyebrow">READ EVENTS / PAGE ORDER</span><span>select a row to inspect</span></div>
          <div className="results-table-wrap trace-scroll">
            <table className="trace-table">
              <thead><tr><th>Time</th><th>Query</th><th>Page</th><th>{measured.length ? "Source" : "Object"}</th><th>{measured.length ? "Plan" : "Type"}</th><th>Cache</th><th>{measured.length ? "Prediction" : "Model B"}</th></tr></thead>
              <tbody>
                {filteredEvents.slice(0, 40).map((event) => (
                  <tr key={event.id} className={selected?.id === event.id ? "selected-row" : ""}>
                    <td className="muted-cell">{event.time}</td>
                    <td>{event.queryId}</td>
                    <td><button className="page-link" onClick={() => setSelectedId(event.id)}>{event.page}</button></td>
                    <td className="object-cell">{event.object}</td>
                    <td>{measured.length ? <span className="page-type type-leaf">{event.plan.slice(0, 24)}</span> : <span className={`page-type type-${event.pageType}`}>{event.pageType}</span>}</td>
                    <td><span className={`cache-state cache-${event.cache}`}>{event.cache}</span></td>
                    <td>{measured.length ? <span className="outcome useful-outcome">{event.predictedPage ? `p${event.predictedPage}` : "none"}</span> : event.useful ? <span className="outcome useful-outcome">useful</span> : event.prefetched ? <span className="outcome wasted-outcome">wasted</span> : <span className="outcome absent-outcome">none</span>}</td>
                  </tr>
                ))}
                {filteredEvents.length === 0 && <tr><td colSpan={7} className="empty-state">No fixture rows match these filters.</td></tr>}
              </tbody>
            </table>
          </div>
          {filteredEvents.length > 40 && <p className="table-note">Showing the first 40 filtered rows. Export contains the current browser sample.</p>}
        </section>
        <aside className="trace-detail panel" aria-label="Selected page details">
          {selected ? <>
            <div className="detail-topline"><span className="eyebrow">PAGE INSPECTOR</span><span className="detail-id">#{String(selected.id).padStart(3, "0")}</span></div>
            <div className="page-number-block"><span>DATABASE PAGE</span><strong>{selected.page}</strong><small>{selected.object}</small></div>
            <dl className="detail-list">
              <div><dt>Query execution</dt><dd>{selected.queryId}</dd></div>
              <div><dt>Workload</dt><dd>{WORKLOADS[selected.workload].family}</dd></div>
              <div><dt>Page type / depth</dt><dd>{selected.pageType} / level {selected.depth}</dd></div>
              <div><dt>Previous page</dt><dd>{selected.previousPage}</dd></div>
              <div><dt>VFS cache</dt><dd>{selected.cache}</dd></div>
              <div><dt>Predicted page</dt><dd>{selected.predictedPage}</dd></div>
              <div><dt>Model confidence</dt><dd>{(selected.confidence * 100).toFixed(0)}%</dd></div>
              <div><dt>Prefetch outcome</dt><dd className={selected.useful ? "good-text" : selected.prefetched ? "warning-text" : "muted-cell"}>{selected.useful ? "useful" : selected.prefetched ? "wasted" : "not issued"}</dd></div>
            </dl>
            <div className="plan-snippet"><span>{measured.length ? "EXPLAIN QUERY PLAN / MEASURED SAMPLE" : "EXPLAIN QUERY PLAN / FIXTURE"}</span><code>{selected.plan}</code></div>
          </> : <div className="empty-inspector">Choose a read event to inspect its page metadata.</div>}
        </aside>
      </div>
      <p className="scope-footnote"><Icon name="database" size={15} /> {measured.length ? "Rows are sampled from a wrapped APSW VFS trace; xRead events do not prove physical disk I/O." : "In the research build, these columns are populated from a wrapped APSW VFS and a cached dbstat page map."}</p>
    </>
  );
}

function ComparisonPage({
  workloadId,
  modeId,
  cacheMb,
  degree,
  activePolicy,
  setActivePolicy,
  inferenceCost,
  setInferenceCost,
  snapshot,
}: {
  workloadId: WorkloadId;
  modeId: ModeId;
  cacheMb: number;
  degree: number;
  activePolicy: PolicyId;
  setActivePolicy: (value: PolicyId) => void;
  inferenceCost: number;
  setInferenceCost: (value: number) => void;
  snapshot: DashboardSnapshot | null;
}) {
  const timelyMisses = Math.ceil(inferenceCost / MODES[modeId].delay);
  const suite = measuredSuite(snapshot);
  return (
    <>
      <PageIntro eyebrow={snapshot ? "MODEL EVALUATION / MEASURED SUITE" : "MODEL EVALUATION / CONTROLLED REPLAY"} title="Compare the predictor, then count its cost." description={snapshot ? "Available results come from the backend export. Missing runtime measurements are shown as n/a; offline accuracy may still be available." : "One query order, one bounded LRU cache, and one miss-latency profile for every strategy. Select a row to drive the overview metrics."} trailing={<span className={snapshot ? "measured-badge" : "fixture-tag"}>{snapshot ? "MEASURED RESULTS" : "FIXTURE RESULTS"}</span>} />
  <section className="comparison-context">
        <div><span className="eyebrow">ACTIVE WORKLOAD</span><strong>{snapshot ? suite?.name ?? "backend suite" : WORKLOADS[workloadId].label}</strong><small>{snapshot ? `${suite?.queries_per_run ?? "n/a"} queries per run` : `${WORKLOADS[workloadId].reads.toLocaleString()} fixture read events`}</small></div>
        <div><span className="eyebrow">MISS PROFILE</span><strong>{snapshot ? suite?.mode ?? "measured" : MODES[modeId].label}</strong><small>{snapshot ? `seeds ${(suite?.seeds ?? []).join(", ")}` : `${cacheMb} MB cache / degree ${degree} / shared LRU policy`}</small></div>
        <div className="context-note"><span className="notice-mark" /><span>{snapshot ? "Rows summarize exported backend measurements; injected-delay suites remain separate from real-storage claims." : "Fixture data is illustrative. No row is an SQLite measurement."}</span></div>
      </section>
      <section className="comparison-table-section panel">
        <div className="panel-heading">
          <div><p className="eyebrow">END-TO-END / LATENCY + PREFETCH QUALITY</p><h2>Strategy scorecard</h2></div>
          <span className="table-method-note">Click a strategy to focus it</span>
        </div>
        <ResultsTable workloadId={workloadId} modeId={modeId} cacheMb={cacheMb} degree={degree} activePolicy={activePolicy} onSelect={setActivePolicy} snapshot={snapshot} />
        <div className="confidence-note"><span className="confidence-icon">i</span><span>{snapshot ? "Suite summaries include coarse run-level intervals and paired query deltas; inspect backend JSON for full provenance." : "Confidence intervals, seed variance, and real inference time are intentionally absent until measured runs are imported."}</span></div>
      </section>
      <div className="comparison-lower-grid">
        <AblationChart />
        <section className="break-even-panel panel">
          <div className="panel-heading">
            <div><p className="eyebrow">OVERHEAD CHECK / SIMPLE MODEL</p><h2>When does inference pay?</h2></div>
            <span className="fixture-tag">ILLUSTRATIVE</span>
          </div>
          <p className="section-description">Break-even illustration only: predictions must hide at least this many cache-miss delays to offset the entered inference budget.</p>
          <label className="range-label" htmlFor="inference-cost"><span>Inference budget</span><strong>{inferenceCost.toFixed(2)} ms</strong></label>
          <input id="inference-cost" className="cost-slider" type="range" min="0.1" max="2" step="0.05" value={inferenceCost} onChange={(event) => setInferenceCost(Number(event.target.value))} />
          <div className="break-even-result"><strong>{timelyMisses}</strong><span>timely misses to cover inference at {MODES[modeId].short}</span></div>
          <p className="formula-note">ceil(inference cost / miss delay). Ignores overlap, CPU contention, and extra I/O; replace with ONNX timing and end-to-end runs.</p>
        </section>
      </div>
      <section className="oracle-note"><span className="oracle-symbol">O</span><div><strong>Oracle is a future-aware diagnostic control.</strong><p>It uses future reads to explore prefetching headroom under the tested scheduling and storage conditions. It is not a deployable predictor or a universal upper bound.</p></div><span className="oracle-value">{snapshot ? formatMs(numericValue(measuredStrategy(suite, "oracle_miss")?.median_of_run_p50_ms ?? measuredStrategy(suite, "oracle")?.median_of_run_p50_ms)) : formatMs(metricFor(workloadId, modeId, cacheMb, "oracle", degree).p50)}<small>{snapshot ? "measured p50" : "fixture p50"}</small></span></section>
    </>
  );
}

function MethodPage({
  onExportConfig,
}: {
  onExportConfig: () => void;
}) {
  const steps = [
    { number: "01", title: "Trace at the VFS", detail: "Wrap APSW xRead calls. Record page offset, execution ID, monotonic timestamp, and query-template metadata." },
    { number: "02", title: "Label page context", detail: "Build a page-to-object map from dbstat; attach B-tree type and depth, plan-node type, and query position." },
    { number: "03", title: "Train without leakage", detail: "Split by execution and unseen parameters. Keep templates held out for a harder generalization test." },
    { number: "04", title: "Replay with bounded I/O", detail: "Use the same LRU cache, cold-start sequence, miss delay, and query order for every baseline." },
  ];
  return (
    <>
      <PageIntro eyebrow="METHOD / SCOPE / REPRODUCIBILITY" title="A fair test begins before the model." description="The protocol is designed to isolate semantic context, measure prediction overhead, and keep synthetic latency injection separate from real storage results." trailing={<button className="outline-button" onClick={onExportConfig}><Icon name="download" size={16} />Export run config</button>} />
      <section className="method-flow" aria-label="Experiment workflow">
        {steps.map((step, index) => (
          <article className="method-step" key={step.number}>
            <div className="step-number">{step.number}<span>{index < steps.length - 1 && <i />}</span></div>
            <div><h2>{step.title}</h2><p>{step.detail}</p></div>
          </article>
        ))}
      </section>
      <div className="method-columns">
        <section className="feature-section panel">
          <div className="panel-heading"><div><p className="eyebrow">FEATURE CONTRACT</p><h2>What Model B can know</h2></div><span className="fixture-tag">PLANNED INPUTS</span></div>
          <div className="feature-list">
            {[
              ["Page history", "Last 8 page IDs and page deltas", "Both"],
              ["Object identity", "Stable table or index ID from page map", "Semantic"],
              ["Page structure", "Interior, leaf, or overflow; B-tree depth", "Semantic"],
              ["Plan context", "EXPLAIN QUERY PLAN node type and template", "Semantic"],
              ["Query position", "Reads since execution start; previous page type", "Semantic"],
            ].map(([name, detail, group]) => <div className="feature-row" key={name}><strong>{name}</strong><span>{detail}</span><small>{group}</small></div>)}
          </div>
          <p className="scope-footnote"><Icon name="layers" size={15} /> Candidate set: pages observed after the current page in training, top 32 plus the next sequential page. This limits generalization to unseen transitions.</p>
        </section>
        <section className="protocol-section panel">
          <div className="panel-heading"><div><p className="eyebrow">CONTROL CHECKLIST</p><h2>Keep comparisons honest</h2></div><Icon name="check" size={20} /></div>
          <ul className="protocol-list">
            <li>Split by query execution, never by individual reads.</li>
            <li>Use matched model capacity, training budget, and tuning effort.</li>
            <li>Run multiple seeds and report means with confidence intervals.</li>
            <li>Report real storage separately from explicitly injected delay.</li>
            <li>Measure prediction cost, CPU, useful reads, and wasted I/O.</li>
            <li>Keep the database static for the primary experiment.</li>
          </ul>
          <div className="baseline-line"><span>Baselines</span><strong>No prefetch / Next-N / Markov-1,2 / Model A / Model B / Oracle</strong></div>
        </section>
      </div>
      <section className="boundary-section">
        <div className="boundary-heading"><span className="boundary-mark"><Icon name="database" size={19} /></span><div><p className="eyebrow">CURRENT IMPLEMENTATION BOUNDARY</p><h2>Measured runs and fixture previews are separate.</h2></div></div>
        <div className="boundary-columns">
          <div><span className="boundary-status status-present"><Icon name="check" size={14} />Local experiments</span><p>Generate a database, trace APSW VFS reads, train models, run the bounded prefetch cache, compare policies, and inspect measured output through the local backend.</p></div>
          <div><span className="boundary-status status-pending"><span />Illustrative controls</span><p>The overview workload, latency-profile and cache sliders preview a deterministic fixture. Start a local experiment to set real backend options and obtain new measurements.</p></div>
        </div>
        <p className="boundary-footnote">Only values labelled as measured backend output should be cited as SQLite experiment results. Development runs are separate from the frozen final study.</p>
      </section>
      <section className="next-work-section">
        <div><p className="eyebrow">REPRODUCIBLE LOCAL PATH</p><h2>Run and inspect fresh measurements.</h2><p>Use Local experiments for a complete development pipeline or select a specific backend command. Open its dashboard export to replace the bundled overview snapshot.</p></div>
        <div className="schema-preview"><span>trace_event.v1</span><code>query_id, offset, page_no, object_id,<br />page_type, depth, plan_node, timestamp</code></div>
      </section>
    </>
  );
}

function App() {
  const selectedRun = useSelectedRun();
  const [view, updateView] = useState<View>(() => {
    const requested = new URLSearchParams(window.location.search).get("view");
    return ["overview", "traces", "comparison", "method", "neural", "lab", "workspace", "preview", "protocol"].includes(requested ?? "") ? requested as View : "overview";
  });
  function setView(next: View) {
    updateView(next);
    const url = new URL(window.location.href);
    url.searchParams.set("view", next);
    window.history.replaceState(null, "", url);
    window.scrollTo({ top: 0 });
  }
  const [dashboardSnapshot, setDashboardSnapshot] = useState<DashboardSnapshot | null>(null);
  const [localSnapshot, setLocalSnapshot] = useState(false);
  const [workloadId, setWorkloadId] = useState<WorkloadId>("mixed");
  const [modeId, setModeId] = useState<ModeId>("sata");
  const [cacheMb, setCacheMb] = useState(256);
  const [degree, setDegree] = useState(4);
  const [activePolicy, setActivePolicy] = useState<PolicyId>("modelB");
  const [visibleSeries, setVisibleSeries] = useState<PolicyId[]>(["stock", "markov", "modelA", "modelB"]);
  const [traceSearch, setTraceSearch] = useState("");
  const [traceWorkload, setTraceWorkload] = useState<WorkloadId | "all">("all");
  const [traceOnlyWasted, setTraceOnlyWasted] = useState(false);
  const [selectedTraceId, setSelectedTraceId] = useState(TRACE_EVENTS[0].id);
  const [inferenceCost, setInferenceCost] = useState(0.45);

  useEffect(() => {
    let cancelled = false;
    fetch("/pagewise-dashboard.json", { cache: "no-store" })
      .then((response) => response.ok ? response.json() : null)
      .then((payload: DashboardSnapshot | null) => {
        if (!cancelled && payload?.schema_version === 1) {
          setDashboardSnapshot(payload);
        }
      })
      .catch(() => {
        if (!cancelled) setDashboardSnapshot(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const workload = WORKLOADS[workloadId];

  function exportResults() {
    if (dashboardSnapshot) {
      downloadFile("pagewise-measured-dashboard.json", JSON.stringify(dashboardSnapshot, null, 2), "application/json");
      return;
    }
    const header = "strategy,workload,miss_profile,cache_mb,prefetch_degree,p50_ms,p95_ms,cache_hit_pct,useful_prefetch_pct,wasted_prefetch_pct,top1_pct,source\n";
    const rows = POLICY_INFO.map((policy) => {
      const metric = metricFor(workloadId, modeId, cacheMb, policy.id, degree);
      return [policy.label, workload.label, MODES[modeId].label, cacheMb, degree, metric.p50.toFixed(2), metric.p95.toFixed(2), metric.hitRate, metric.useful, metric.wasted, metric.top1, "illustrative-browser-fixture"].join(",");
    });
    downloadFile("pagewise-fixture-results.csv", header + rows.join("\n"), "text/csv");
  }

  async function exportNeuralStudy() {
    const response = await fetch("/pagewise-neural-improvement.json", { cache: "no-store" });
    if (!response.ok) return;
    downloadFile("pagewise-neural-improvement.json", JSON.stringify(await response.json(), null, 2), "application/json");
  }

  function exportTrace() {
    if (!localSnapshot && selectedRun.result?.evaluation) {
      downloadFile("pagewise-semantic-prediction-samples.json", JSON.stringify(selectedRun.result.evaluation.samples, null, 2), "application/json");
      return;
    }
    const header = "time,query_id,workload,page,previous_page,object,page_type,depth,plan_node,cache,predicted_page,confidence,prefetched,useful\n";
    const rows = (dashboardSnapshot ? measuredTraceEvents(dashboardSnapshot) : TRACE_EVENTS).map((event) => [event.time, event.queryId, event.workload, event.page, event.previousPage, event.object, event.pageType, event.depth, `"${event.plan.replace(/"/g, '""')}"`, event.cache, event.predictedPage, event.confidence, event.prefetched, event.useful].join(","));
    downloadFile(dashboardSnapshot ? "pagewise-measured-trace.csv" : "pagewise-fixture-trace.csv", header + rows.join("\n"), "text/csv");
  }

  function exportConfig() {
    const config = {
      project: "Pagewise SQLite prefetch lab",
      source: "browser-fixture-only; not measured SQLite output",
      workload: workloadId,
      missProfile: { ...MODES[modeId], emulated: true },
      cacheMb,
      policySet: POLICY_INFO.map((policy) => policy.id),
      prefetchDegree: degree,
      split: "by query execution; test unseen parameter values",
      candidatePolicy: "top 32 observed successors plus next sequential page",
      note: "Replace fixture metrics with APSW VFS and ONNX measurements before making performance claims.",
    };
    downloadFile("pagewise-run-config.json", JSON.stringify(config, null, 2), "application/json");
  }

  const pageLabels: Record<View, string> = {
    overview: "Project overview",
    traces: "Page-read inspector",
    comparison: "Detailed comparison",
    method: "How it works",
    neural: "Research results",
    lab: "Live demonstration",
    workspace: "Interactive workspace",
    preview: "Historical dashboard & fixture preview",
    protocol: "Research protocol",
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <LogoMark />
          <div><strong>pagewise</strong><span>SQLITE RESEARCH LAB</span></div>
        </div>
        <div className="nav-caption">PROJECT WALKTHROUGH</div>
        <nav className="primary-nav" aria-label="Primary navigation">
          {([
            ["overview", "Project overview", "overview"],
            ["method", "How it works", "method"],
            ["lab", "Live demonstration", "play"],
            ["workspace", "Interactive workspace", "database"],
            ["neural", "Research results", "compare"],
          ] as [View, string, string][]).map(([id, label, icon]) => (
            <button key={id} className={`nav-link ${view === id ? "nav-active" : ""}`} onClick={() => setView(id)} aria-current={view === id ? "page" : undefined}>
              <Icon name={icon} size={18} /><span>{label}</span>{view === id && <i className="nav-indicator" />}
            </button>
          ))}
        </nav>
        <details className="sidebar-advanced"><summary>Research workspace</summary><nav className="primary-nav" aria-label="Advanced research navigation">{([
          ["traces", "Page-read inspector", "trace"], ["comparison", "Detailed comparison", "compare"], ["preview", "Historical dashboard", "overview"], ["protocol", "Protocol & fixtures", "method"],
        ] as [View, string, string][]).map(([id, label, icon]) => <button key={id} className={`nav-link ${view === id ? "nav-active" : ""}`} onClick={() => setView(id)} aria-current={view === id ? "page" : undefined}><Icon name={icon} size={18} /><span>{label}</span></button>)}</nav></details>
        <div className="sidebar-bottom">
          <div className="sidebar-project">
            <span className="project-icon"><Icon name="database" size={16} /></span>
            <div><small>PAGEWISE</small><strong>Semantics-aware SQLite prefetching</strong></div>
          </div>
          <div className="sidebar-status"><div><strong>LOCAL EDITION</strong><small>Research evidence & live experiments</small></div></div>
          <div className="sidebar-meta">DATABASE SYSTEMS PROJECT</div>
        </div>
      </aside>

      <main className="main-shell">
        <header className="topbar">
          <div className="breadcrumbs"><span>Pagewise</span><i>/</i><strong>{pageLabels[view]}</strong></div>
          <div className="topbar-actions">
            {!["overview", "method", "lab", "workspace", "comparison"].includes(view) && <button className="export-button" onClick={view === "neural" ? exportNeuralStudy : view === "traces" ? exportTrace : view === "protocol" ? exportConfig : exportResults}><Icon name="download" size={15} /><span>{view === "neural" ? "Download saved study" : view === "protocol" ? "Export fixture config" : view === "traces" ? localSnapshot ? "Export read sample" : "Export prediction sample" : dashboardSnapshot ? "Export measured JSON" : "Export fixture results"}</span></button>}
            {view !== "lab" && <button className="header-demo-button" onClick={() => setView("lab")}>Live demo <Icon name="arrow" size={14} /></button>}
          </div>
        </header>

        <div className="content-area" key={view}>
          {["preview", "protocol"].includes(view) && <DemoNotice onOpenMethod={() => setView("protocol")} snapshot={dashboardSnapshot} />}
          {view === "overview" && <ProjectOverview selectedResult={selectedRun.result} onDemo={() => setView("lab")} onResults={() => setView("neural")} onMethod={() => setView("method")} />}
          {view === "method" && <HowItWorks onDemo={() => setView("lab")} />}
          {view === "workspace" && <InteractiveWorkspace />}
          {view === "neural" && <NeuralStudy selectedResult={selectedRun.result} onOpenLab={() => setView("lab")} />}
          {view === "lab" && <LocalLab onLoadSnapshot={(payload) => { setDashboardSnapshot(payload as DashboardSnapshot); setLocalSnapshot(true); setView("traces"); }} />}
          {view === "preview" && <>
            <PageIntro eyebrow="PAGEWISE / SQLITE PAGE-PREFETCH STUDY" title="Can semantics predict the next page?" description="Inspect measured backend exports, then run fresh SQLite experiments in the local workspace. Preview controls below use an illustrative fixture." trailing={<span className="intro-count">01 <small>research question</small></span>} />
            <MeasuredBackendPanel snapshot={dashboardSnapshot} local={localSnapshot} />
            <div className="overview-grid">
              <LatencyChart workloadId={workloadId} modeId={modeId} cacheMb={cacheMb} degree={degree} visible={visibleSeries} setVisible={setVisibleSeries} snapshot={dashboardSnapshot} />
              <ReplayControls workloadId={workloadId} setWorkloadId={setWorkloadId} modeId={modeId} setModeId={setModeId} cacheMb={cacheMb} setCacheMb={setCacheMb} degree={degree} setDegree={setDegree} onRun={() => setView("lab")} />
            </div>
            <div className="run-meta-line"><span className="run-state-dot" />Previewing deterministic fixture data<span className="run-meta-separator">/</span>{workload.family}<span className="run-meta-separator">/</span>{MODES[modeId].short} miss profile</div>
            <MetricStrip workloadId={workloadId} modeId={modeId} cacheMb={cacheMb} degree={degree} activePolicy={activePolicy} snapshot={dashboardSnapshot} />
            <div className="overview-lower-grid">
              <section className="overview-results panel">
                <div className="panel-heading">
                  <div><p className="eyebrow">STRATEGY SNAPSHOT / FIXTURE</p><h2>Same workload, six policies</h2></div>
                  <button className="text-action" onClick={() => setView("comparison")}>Full comparison <Icon name="arrow" size={15} /></button>
                </div>
                <ResultsTable workloadId={workloadId} modeId={modeId} cacheMb={cacheMb} degree={degree} activePolicy={activePolicy} onSelect={setActivePolicy} snapshot={dashboardSnapshot} compact />
                <div className="oracle-inline"><span className="oracle-mark">O</span><span>Oracle ceiling</span><strong>{dashboardSnapshot ? "n/a" : formatMs(metricFor(workloadId, modeId, cacheMb, "oracle", degree).p50)}</strong><small>not a deployable predictor</small></div>
              </section>
              <PredictionHeatmap workloadId={workloadId} snapshot={dashboardSnapshot} />
            </div>
            <div className="overview-endnote"><span>READ THE RESULT CAREFULLY</span><p>Model B is not guaranteed to win. The range-scan control in this fixture favors sequential prediction, and sample numbers are not measured SQLite output.</p><button className="text-action" onClick={() => setView("method")}>Read method & scope <Icon name="arrow" size={15} /></button></div>
          </>}

          {view === "traces" && <div className="faculty-page"><div className="faculty-title"><p className="eyebrow">SELECTED SEMANTIC MODEL</p><h1>Page prediction inspector</h1><p>Inspect the exact history, semantic page metadata, model ranking and observed next page.</p></div><SemanticEvaluation result={selectedRun.result} inspector onDemo={() => setView("lab")} /><details className="faculty-advanced" open={localSnapshot || undefined}><summary>Earlier A/B dashboard read samples</summary><TraceExplorer workloadFilter={traceWorkload} setWorkloadFilter={setTraceWorkload} search={traceSearch} setSearch={setTraceSearch} onlyWasted={traceOnlyWasted} setOnlyWasted={setTraceOnlyWasted} selectedId={selectedTraceId} setSelectedId={setSelectedTraceId} snapshot={dashboardSnapshot} /></details></div>}
          {view === "comparison" && <div className="faculty-page"><div className="faculty-title"><p className="eyebrow">ACTUAL SELECTED MODEL OUTPUT</p><h1>Prediction and runtime comparison</h1><p>{selectedRun.started_at ? `Latest evaluated local run: ${new Date(selectedRun.started_at).toLocaleString()} · ${selectedRun.job_id}` : "Complete a trained-hybrid evaluation to populate this page."}</p></div><SemanticEvaluation result={selectedRun.result} onDemo={() => setView("lab")} />{selectedRun.result && <TrainedHybridResults result={selectedRun.result} showEvaluation={false} />}<details className="faculty-advanced"><summary>Earlier A/B dashboard comparison and fixture calculators</summary><ComparisonPage workloadId={workloadId} modeId={modeId} cacheMb={cacheMb} degree={degree} activePolicy={activePolicy} setActivePolicy={setActivePolicy} inferenceCost={inferenceCost} setInferenceCost={setInferenceCost} snapshot={dashboardSnapshot} /></details></div>}
          {view === "protocol" && <MethodPage onExportConfig={exportConfig} />}
          <footer className="page-footer"><span>PAGEWISE / SQLITE PAGE PREFETCHING</span><span>Measured results describe the tested conditions.</span><button onClick={() => setView("method")}>How it works</button></footer>
        </div>
      </main>
    </div>
  );
}

export default App;
