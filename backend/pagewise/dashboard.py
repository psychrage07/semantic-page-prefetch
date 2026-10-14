"""Optional Streamlit viewer for measured Pagewise result files.

No synthetic values are inserted. Upload suite_summary.json, predictor_metrics.json,
run_metadata.json, predictions.jsonl, or runtime_stats.json produced by the CLI.
"""
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pagewise measured results", page_icon="▦", layout="wide")
st.title("Pagewise · SQLite page-prefetch research")
st.caption("Measured-run viewer. Empty states are intentional; this page never invents benchmark results.")

st.sidebar.header("Load experiment artifacts")
suite_upload = st.sidebar.file_uploader("suite_summary.json", type=["json"], key="suite")
predictor_upload = st.sidebar.file_uploader("predictor_metrics.json", type=["json"], key="predictors")
run_upload = st.sidebar.file_uploader("run_metadata.json", type=["json"], key="run")
runtime_upload = st.sidebar.file_uploader("runtime_stats.json", type=["json"], key="runtime")
predictions_upload = st.sidebar.file_uploader("predictions.jsonl", type=["jsonl", "txt"], key="prediction_log")


def parse_json(upload) -> dict[str, Any] | None:
    if upload is None:
        return None
    try:
        return json.loads(upload.getvalue().decode("utf-8-sig"))
    except Exception as exc:
        st.error(f"Could not parse {upload.name}: {exc}")
        return None


suite = parse_json(suite_upload)
predictors = parse_json(predictor_upload)
run_meta = parse_json(run_upload)
runtime = parse_json(runtime_upload)

if suite:
    st.subheader("Repeated-run comparison")
    st.info(suite.get("warning", "Check the run metadata for the exact measurement conditions."))
    st.write(f"**Mode:** {suite.get('mode', 'not specified')} · **Queries per run:** {suite.get('queries_per_run', 'unknown')} · **Seeds:** {suite.get('seeds', [])}")
    rows = suite.get("strategies", [])
    if rows:
        frame = pd.DataFrame(rows)
        st.dataframe(frame, use_container_width=True, hide_index=True)
        latency_column = "median_of_run_p50_ms"
        if latency_column in frame and frame[latency_column].notna().any():
            chart = frame.set_index("strategy")[[latency_column]].dropna()
            st.markdown("#### Median query p50 across runs")
            st.bar_chart(chart)
        p95_column = "median_of_run_p95_ms"
        if p95_column in frame and frame[p95_column].notna().any():
            st.markdown("#### Median query p95 across runs")
            st.bar_chart(frame.set_index("strategy")[[p95_column]].dropna())
    else:
        st.warning("This suite summary does not contain strategy rows.")

if predictors:
    st.subheader("Held-out prediction metrics")
    st.caption(predictors.get("note", "Metrics are prediction results; they do not by themselves establish runtime speedup."))
    policies = predictors.get("policies", {})
    if policies:
        rows = [{"policy": policy, **metrics} for policy, metrics in policies.items()]
        frame = pd.DataFrame(rows)
        st.dataframe(frame, use_container_width=True, hide_index=True)
        for metric in ["top1_accuracy_pct", "top5_accuracy_pct", "candidate_recall_pct"]:
            if metric in frame and frame[metric].notna().any():
                st.markdown(f"#### {metric.replace('_', ' ').title()}")
                st.bar_chart(frame.set_index("policy")[[metric]].dropna())

if run_meta:
    st.subheader("Single run details")
    st.info(run_meta.get("physical_io_note", "Check trace definitions before interpreting read counts."))
    st.json({key: value for key, value in run_meta.items() if key != "results"})
    latency = run_meta.get("query_latency_ms", {})
    if latency:
        cols = st.columns(4)
        for col, label, key in zip(cols, ["p50", "p95", "Total query time", "Query count"], ["median", "p95", "total", None]):
            value = run_meta.get("query_count") if key is None else latency.get(key)
            col.metric(label, f"{value} {'ms' if key is not None else ''}" if value is not None else "—")

if runtime:
    st.subheader("Prefetch and cache accounting")
    st.json(runtime)
    demand = int(runtime.get("demand_requests", 0))
    hits = int(runtime.get("cache_hits", 0))
    if demand:
        st.metric("Custom cache hit rate", f"{hits / demand * 100:.2f}%")

if predictions_upload is not None:
    st.subheader("Prediction log")
    try:
        text = predictions_upload.getvalue().decode("utf-8-sig")
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not records:
            st.warning("The prediction log is empty. This is expected for the no-prefetch strategy.")
        else:
            frame = pd.json_normalize(records, sep=".")
            query_ids = sorted(str(x) for x in frame.get("query_id", pd.Series(dtype=str)).dropna().unique())
            selected_query = st.selectbox("Query ID", ["All"] + query_ids)
            if selected_query != "All" and "query_id" in frame:
                frame = frame[frame["query_id"].astype(str) == selected_query]
            st.dataframe(frame.head(500), use_container_width=True, hide_index=True)
            st.caption(f"Showing at most 500 of {len(frame)} prediction jobs.")
    except Exception as exc:
        st.error(f"Could not parse prediction JSONL: {exc}")

if not any([suite, predictors, run_meta, runtime, predictions_upload]):
    st.warning("No measured artifacts loaded yet. Run the backend CLI, then upload its JSON/JSONL outputs using the sidebar.")
    st.markdown("""
    **Suggested artifacts**

    - `suite_summary.json` for strategy-level latency comparisons
    - `predictor_metrics.json` for offline held-out prediction accuracy
    - `run_metadata.json` for a single measured SQLite run
    - `runtime_stats.json` and `predictions.jsonl` for prefetch/cache behavior

    Keep real-storage runs separate from latency-injected runs. Do not interpret a VFS read as proof of physical disk I/O.
    """)
