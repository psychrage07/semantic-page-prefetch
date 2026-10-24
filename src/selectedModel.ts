import { useEffect, useState } from "react";

export type EvaluationMetric = {
  queries: number; transitions: number; top1_macro_pct: number; top1_micro_pct: number;
  top5_micro_pct: number; candidate_recall_pct: number;
  by_family: Record<string, { transitions: number; top1_micro_pct: number }>;
};
export type Candidate = { page: number; score: number; object?: string; page_type?: string; depth?: number };
export type SemanticSample = {
  query_id: string; seed: number; position: number; template_id: string; family: string; plan_class: string;
  history: number[]; target_page: number; target_metadata: { object?: string; page_type?: string; depth?: number };
  top1_correct: boolean; top5_correct: boolean; candidates: Candidate[];
};
export type SemanticEvaluation = {
  checkpoint_sha256: string; semantics_enabled: boolean; source: string; scope: string;
  policies: Record<string, EvaluationMetric>; samples: SemanticSample[];
};
export type TrainedHybridSnapshot = {
  artifact_type: "trained_hybrid_live"; checkpoint_sha256: string; variant: string; database_bytes: number;
  seeds: number[]; queries_per_seed: number; scope: string; workloads_paired: boolean;
  rows: { policy: string; label: string; queries: number; mean_ms: number; median_ms: number; scored_jobs: number;
    pages_prefetched: number; useful_pages: number; cache_hit_pct?: number | null; unused_pages?: number; inference_ms_per_job?: number | null }[];
  prediction_samples: { query_id: string; seed: number; history: number[]; candidates: Candidate[] }[];
  evaluation?: SemanticEvaluation;
};
export type SelectedRun = { available: boolean; job_id?: string; started_at?: string; result?: TrainedHybridSnapshot; reason?: string };

export function useSelectedRun() {
  const [run, setRun] = useState<SelectedRun>({ available: false });
  useEffect(() => {
    let disposed = false;
    const refresh = async () => {
      try {
        const response = await fetch("/api/selected-results", { cache: "no-store" });
        if (!response.ok) return;
        const data = await response.json() as SelectedRun;
        if (!disposed) setRun(data);
      } catch { /* Static research results remain available without the API. */ }
    };
    void refresh(); const timer = window.setInterval(() => void refresh(), 5000);
    return () => { disposed = true; window.clearInterval(timer); };
  }, []);
  return run;
}
