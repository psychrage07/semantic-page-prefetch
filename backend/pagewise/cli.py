from __future__ import annotations

import argparse
import json
import sys
import platform
import sqlite3
from pathlib import Path

from .database import generate_database
from .dataset import build_sequences
from .evaluate import evaluate_predictors
from .metadata import build_page_map, dbstat_available, read_page_size
from .models import train_models
from .reporting import build_markdown_report, build_consolidated_report, write_dashboard_snapshot
from .runner import trace_workload
from .v21 import run_headroom_v21, diagnose_sim_cells
from .suite import recover_suite, run_runtime_sensitivity_sweep, run_suite, summarize_suite
from .sweep import train_evaluate_sweep
from .offline import run_offline_study, run_ablation


def _print_json(value) -> None:
    print(json.dumps(value, indent=2, default=str))


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_ablation_specs(value: str) -> dict[str, list[str]]:
    specs: dict[str, list[str]] = {}
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            label, groups = item.split("=", 1)
        elif ":" in item:
            label, groups = item.split(":", 1)
        else:
            groups = item
            label = "no_" + "_".join(_parse_str_list(groups))
        label = label.strip().replace(" ", "_")
        if not label:
            raise ValueError(f"Invalid empty ablation label in {item!r}")
        specs[label] = _parse_str_list(groups)
    return specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pagewise", description="SQLite learned page-prefetch research prototype")
    sub = parser.add_subparsers(dest="command", required=True)

    health = sub.add_parser("check-env", help="Check Python, SQLite, APSW, PyTorch, and dbstat support")

    gen = sub.add_parser("generate-db", help="Generate a reproducible e-commerce SQLite database")
    gen.add_argument("--db", default="data/pagewise-dev.sqlite")
    gen.add_argument("--customers", type=int, default=10_000)
    gen.add_argument("--products", type=int, default=1_000)
    gen.add_argument("--orders", type=int, default=50_000)
    gen.add_argument("--avg-items", type=int, default=3)
    gen.add_argument("--graph-nodes", type=int, default=5_000)
    gen.add_argument("--edges-per-node", type=int, default=3)
    gen.add_argument("--item-payload-bytes", type=int, default=0,
                     help="Deterministic BLOB bytes stored in each order-item row for the large-database branch")
    gen.add_argument("--seed", type=int, default=17)
    gen.add_argument("--overwrite", action="store_true")

    trace = sub.add_parser("trace", help="Trace VFS reads for generated SQLite workload (requires APSW)")
    trace.add_argument("--db", required=True)
    trace.add_argument("--out", default="runs/trace-001")
    trace.add_argument("--queries", type=int, default=500)
    trace.add_argument("--seed", type=int, default=1234)
    trace.add_argument("--cache-kib", type=int, default=4096)
    trace.add_argument("--sequential", action="store_true", help="Cycle templates rather than mixed sampling")
    trace.add_argument("--sparse-page-map", action="store_true",
                       help="Store dbstat metadata only for pages observed in this training trace")


    sweep = sub.add_parser("train-sweep", help="Train/evaluate both MLPs across multiple model seeds with a fixed split")
    sweep.add_argument("--sequences", required=True)
    sweep.add_argument("--page-map", required=True)
    sweep.add_argument("--out", default="models/sweep-001")
    sweep.add_argument("--seeds", default="7,17,27")
    sweep.add_argument("--split-seed", type=int, default=7)
    sweep.add_argument("--epochs", type=int, default=15)
    sweep.add_argument("--candidates", type=int, default=32)
    sweep.add_argument("--hidden", type=int, default=128)
    sweep.add_argument("--semantic-ablations", default="", help="Train/evaluate retrained Model B ablations across model seeds")

    prefetch = sub.add_parser("run-prefetch", help="Run real SQLite queries with the bounded VFS cache and an optional async prefetch policy")
    prefetch.add_argument("--db", required=True)
    prefetch.add_argument("--model-dir", required=True, help="Directory containing candidate_index.json and model_a.pt/model_b.pt")
    prefetch.add_argument("--model-path", default=None, help="Optional explicit model checkpoint for model_a/model_b strategies, e.g. a retrained ablation checkpoint")
    prefetch.add_argument("--strategy", choices=["none", "null", "oracle", "oracle_next", "oracle_miss", "oracle_optimal", "sequential", "markov1", "markov2", "model_a", "model_b", "hybrid"], default="model_b")
    prefetch.add_argument("--out", required=True)
    prefetch.add_argument("--queries", type=int, default=250)
    prefetch.add_argument("--seed", type=int, default=9876)
    prefetch.add_argument("--cache-kib", type=int, default=4096)
    prefetch.add_argument("--cache-pages", type=int, default=256)
    prefetch.add_argument("--degree", type=int, default=2)
    prefetch.add_argument("--threshold", type=float, default=0.0, help="Candidate score threshold; scores are ranking scores, not calibrated probabilities. Default 0.0 selects top-ranked candidates up to degree.")
    prefetch.add_argument("--queue-size", type=int, default=64, help="Maximum queued prediction jobs for async prefetching")
    prefetch.add_argument("--predict-on", choices=["all", "miss", "hit"], default="all", help="Schedule predictions after all observed pages, only custom-cache misses, or only hits")
    prefetch.add_argument("--predict-every", type=int, default=1, help="Schedule at most every Nth eligible observed page")
    prefetch.add_argument("--miss-delay-ms", type=float, default=0.0, help="Optional delay injected at custom-cache demand misses; results must be labelled as latency-injected")
    prefetch.add_argument("--max-score-candidates", type=int, default=0, help="When >0, pre-rank candidates by Markov prior and only score the top N neurally; 0 scores all (default)")
    prefetch.add_argument("--storage-model", choices=["independent", "shared", "k_channel"], default="independent", help="Storage resource model: independent (parallel delay), shared (single serialized channel), or k_channel (v2.1 abstract k concurrent slots; simulated concurrency scenario)")
    prefetch.add_argument("--storage-channels", type=int, default=None, help="Number of service slots k for --storage-model k_channel (evaluated at k=1, 4, 16; requires device-specific calibration before any HDD/NVMe label)")
    prefetch.add_argument("--drop-stale-jobs", action="store_true", help="Drop queued prediction jobs whose query has already finished")
    prefetch.add_argument("--oracle-trace", default=None, help="JSONL demand trace the oracle_next strategy replays to prefetch the true next pages")
    prefetch.add_argument("--oracle-miss-ref", default=None, help="demand_reference.json from a none run; oracle_miss prefetches the next K would-be-miss pages")
    prefetch.add_argument("--no-diagnostics", action="store_true", help="Disable JSONL diagnostic logging (trace.jsonl, predictions.jsonl) for benchmark runs")
    prefetch.add_argument("--prefetch-workers", type=int, default=1, help="Prefetch read threads; >1 lets a parallel (independent) device serve prefetches concurrently")

    build = sub.add_parser("build-sequences", help="Convert VFS JSONL to per-query page sequences")
    build.add_argument("--trace", required=True)
    build.add_argument("--page-map", required=True)
    build.add_argument("--out", required=True)

    train = sub.add_parser("train", help="Train matched history-only and semantic MLP rankers")
    train.add_argument("--sequences", required=True)
    train.add_argument("--page-map", required=True)
    train.add_argument("--out", default="models/run-001")
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--candidates", type=int, default=32)
    train.add_argument("--hidden", type=int, default=128)
    train.add_argument("--semantic-ablations", default="", help="Train additional Model B checkpoints with groups removed, e.g. no_object=object;no_query=query_context")

    suite = sub.add_parser("run-suite", help="Run repeated SQLite experiments across stock, heuristics, and both MLP models")
    suite.add_argument("--db", required=True)
    suite.add_argument("--model-dir", required=True)
    suite.add_argument("--out", required=True)
    suite.add_argument("--queries", type=int, default=250)
    suite.add_argument("--seeds", default="101,202,303", help="Comma-separated seeds; each strategy uses the same seed within a trial")
    suite.add_argument("--strategies", default="stock,none,sequential,markov1,markov2,model_a,model_b,hybrid")
    suite.add_argument("--cache-kib", type=int, default=4096)
    suite.add_argument("--cache-pages", type=int, default=256)
    suite.add_argument("--degree", type=int, default=2)
    suite.add_argument("--threshold", type=float, default=0.0)
    suite.add_argument("--queue-size", type=int, default=64)
    suite.add_argument("--predict-on", choices=["all", "miss", "hit"], default="all")
    suite.add_argument("--predict-every", type=int, default=1)
    suite.add_argument("--miss-delay-ms", type=float, default=0.0, help="Use >0 only for an explicitly reported latency-injected mode")
    suite.add_argument("--max-score-candidates", type=int, default=0, help="Pre-rank then cap neural scoring candidates; 0=score all")
    suite.add_argument("--storage-model", choices=["independent", "shared", "k_channel"], default="independent", help="Storage resource model: independent (parallel delay), shared (single serialized channel), or k_channel (v2.1 abstract k concurrent slots; simulated concurrency scenario)")
    suite.add_argument("--storage-channels", type=int, default=None, help="Service slots k for --storage-model k_channel (v2.1: evaluated at k=1, 4, 16)")
    suite.add_argument("--drop-stale-jobs", action="store_true", help="Drop queued prediction jobs whose query has already finished")
    suite.add_argument("--no-diagnostics", action="store_true", help="Disable JSONL diagnostic logging (trace.jsonl, predictions.jsonl) for benchmark runs")
    suite.add_argument("--prefetch-workers", type=int, default=1, help="Prefetch read threads shared by every strategy in the suite")

    runtime_sweep = sub.add_parser("runtime-sweep", help="Run a grid of runtime scheduler/cache settings using run-suite artifacts")
    runtime_sweep.add_argument("--db", required=True)
    runtime_sweep.add_argument("--model-dir", required=True)
    runtime_sweep.add_argument("--out", required=True)
    runtime_sweep.add_argument("--queries", type=int, default=150)
    runtime_sweep.add_argument("--seeds", default="101,202")
    runtime_sweep.add_argument("--strategies", default="none,model_a,model_b")
    runtime_sweep.add_argument("--baseline", choices=["stock", "none", "sequential", "markov1", "markov2", "model_a", "model_b", "hybrid"], default="none")
    runtime_sweep.add_argument("--cache-kib", type=int, default=4096)
    runtime_sweep.add_argument("--cache-pages", default="128,256", help="Comma-separated custom-cache page capacities")
    runtime_sweep.add_argument("--degrees", default="1,2", help="Comma-separated prefetch degrees")
    runtime_sweep.add_argument("--queue-sizes", default="64", help="Comma-separated async queue sizes")
    runtime_sweep.add_argument("--predict-on", default="all,miss", help="Comma-separated values from all,miss,hit")
    runtime_sweep.add_argument("--predict-every", default="1", help="Comma-separated scheduling strides")
    runtime_sweep.add_argument("--threshold", type=float, default=0.0)
    runtime_sweep.add_argument("--miss-delay-ms", type=float, default=0.0)

    summary_parser = sub.add_parser("summarize-suite", help="Rebuild summary files from suite_runs.json")
    summary_parser.add_argument("--runs", required=True, help="Path to suite_runs.json")
    summary_parser.add_argument("--out", required=True)

    recover_parser = sub.add_parser("recover-suite", help="Rebuild a suite manifest and summary from completed run directories")
    recover_parser.add_argument("--out", required=True, help="Existing run-suite output directory")

    report_parser = sub.add_parser("report", help="Create a concise Markdown table report from a Pagewise JSON artifact")
    report_parser.add_argument("--input", required=True, help="suite_summary.json, training_sweep_report.json, or runtime_sensitivity_report.json")
    report_parser.add_argument("--out", default=None, help="Output Markdown path; defaults to input path with .md suffix")
    report_parser.add_argument("--top-n", type=int, default=12, help="Rows to include for runtime sensitivity rankings")

    research_report = sub.add_parser("research-report", help="Generate a consolidated research report from an experiment directory")
    research_report.add_argument("--run-dir", required=True, help="Root directory of an experiment run")
    research_report.add_argument("--out", default=None, help="Output Markdown path; defaults to run_dir/research_report.md")
    research_report.add_argument("--optimized-suite", default=None, help="Optional directory for Model B optimized suite")

    dashboard_export = sub.add_parser("dashboard-export", help="Export a compact JSON snapshot for the React dashboard")
    dashboard_export.add_argument("--run-dir", required=True, help="Root directory of an experiment run")
    dashboard_export.add_argument("--out", required=True, help="Output JSON path, e.g. ../public/pagewise-dashboard.json")

    report_v21 = sub.add_parser("report-v21", help="Generate the v2.1 consolidated research report from a v2.1 study directory (mechanically, from verified artifacts)")
    report_v21.add_argument("--study-dir", required=True, help="New v2.1 output directory containing headroom_v21.json / offline_study_v21.json / run_log.jsonl")
    report_v21.add_argument("--out", default=None, help="Output Markdown path; defaults to study_dir/research_report_v21.md")

    dashboard_v21 = sub.add_parser("dashboard-export-v21", help="Export the v2.1 dashboard snapshot (new file; the frozen v2 snapshot is untouched)")
    dashboard_v21.add_argument("--study-dir", required=True)
    dashboard_v21.add_argument("--out", required=True, help="Output JSON path, e.g. ../public/pagewise-dashboard-v2.1.json")

    ev = sub.add_parser("evaluate", help="Evaluate trained rankers and simple predictors on held-out executions")
    ev.add_argument("--sequences", required=True, help="test_sequences.jsonl produced by training")
    ev.add_argument("--page-map", required=True)
    ev.add_argument("--candidate-index", required=True)
    ev.add_argument("--model-a", required=True)
    ev.add_argument("--model-b", required=True)
    ev.add_argument("--out", default="results/evaluation-001")
    ev.add_argument("--top-k", type=int, default=5)
    ev.add_argument("--semantic-ablations", default="", help="Semicolon-separated Model B ablations, e.g. no_object=object;no_query=query_context")
    ev.add_argument("--retrained-ablation-dir", default=None, help="Directory containing model_b_retrained_*.pt checkpoints from train --semantic-ablations")

    offline = sub.add_parser("offline-study", help="v2 offline next-page prediction study with parameter/template splits")
    offline.add_argument("--sequences", required=True)
    offline.add_argument("--workload", required=True, help="workload.jsonl with query parameters")
    offline.add_argument("--page-map", required=True)
    offline.add_argument("--out", required=True)
    offline.add_argument("--seeds", default="7,17,27,37,47")
    offline.add_argument("--epochs", type=int, default=20)
    offline.add_argument("--hidden", type=int, default=128)
    offline.add_argument("--candidates", type=int, default=32)
    offline.add_argument("--top-k", type=int, default=8)

    offline_v21 = sub.add_parser("offline-study-v21",
                                  help="v2.1 offline study: Model B preserved + B2/B3 + declared family + Holm")
    offline_v21.add_argument("--sequences", required=True)
    offline_v21.add_argument("--workload", required=True, help="workload.jsonl with query parameters")
    offline_v21.add_argument("--page-map", required=True)
    offline_v21.add_argument("--out", required=True, help="New v2.1 output directory (frozen v2 artifacts are never edited)")
    offline_v21.add_argument("--seeds", default="101,102,103,104,105,106,107,108,109,110", help="Development seeds 101-110 for tuning only")
    offline_v21.add_argument("--epochs", type=int, default=20)
    offline_v21.add_argument("--hidden", type=int, default=128)
    offline_v21.add_argument("--candidates", type=int, default=32)
    offline_v21.add_argument("--top-k", type=int, default=8)
    offline_v21.add_argument("--ledger", default=None, help="run_log.jsonl path; every configuration attempt is appended")

    headroom = sub.add_parser("headroom-v21",
                              help="v2.1 oracle headroom grid: oracle_optimal vs none across k-channel configs")
    headroom.add_argument("--reference", required=True, help="demand_reference.json from a none run")
    headroom.add_argument("--caches", default="32,64,128", help="Comma-separated cache-page capacities")
    headroom.add_argument("--delays", default="0.1,0.2,0.5,1.0", help="Comma-separated injected delays (ms)")
    headroom.add_argument("--channels", default="1,4,16", help="k-channel slots k evaluated (k=1 is the shared model)")
    headroom.add_argument("--storage-model", choices=["shared", "k_channel"], default="k_channel")
    headroom.add_argument("--degree", type=int, default=2)
    headroom.add_argument("--threshold-pct", type=float, default=10.0, help="Reopen threshold: any eligible configuration with >= this headroom reopens the runtime comparison")
    headroom.add_argument("--out", required=True, help="New v2.1 output directory")
    headroom.add_argument("--ledger", default=None)

    simdiag = sub.add_parser("sim-validate-cells",
                             help="v2.1 simulator validation cell diagnosis (15%% target; decomposes error causes)")
    simdiag.add_argument("--calibration", required=True)
    simdiag.add_argument("--runs", required=True, nargs="+", help="Run directories (real runtime runs) used as validation targets")
    simdiag.add_argument("--out", required=True)

    ablate = sub.add_parser("offline-ablation", help="Model B semantic-feature ablation on an offline split")
    ablate.add_argument("--sequences", required=True)
    ablate.add_argument("--workload", required=True)
    ablate.add_argument("--page-map", required=True)
    ablate.add_argument("--out", required=True)
    ablate.add_argument("--split", choices=["params", "templates"], default="params")
    ablate.add_argument("--seeds", default="7,17,27")
    ablate.add_argument("--epochs", type=int, default=20)
    ablate.add_argument("--hidden", type=int, default=128)
    ablate.add_argument("--candidates", type=int, default=32)

    args = parser.parse_args(argv)
    try:
        if args.command == "check-env":
            try:
                import apsw
                apsw_info = {"available": True, "version": apsw.apswversion(), "sqlite_version": apsw.sqlitelibversion()}
            except Exception as exc:
                apsw_info = {"available": False, "error": str(exc)}
            try:
                import torch
                torch_info = {"available": True, "version": torch.__version__, "cuda_available": torch.cuda.is_available()}
            except Exception as exc:
                torch_info = {"available": False, "error": str(exc)}
            dbstat_support = dbstat_available()
            _print_json({"python": sys.version, "platform": platform.platform(), "sqlite_version": sqlite3.sqlite_version,
                         "apsw": apsw_info, "torch": torch_info,
                         "dbstat_support": dbstat_support,
                         "dbstat_available": any(dbstat_support.values())})
        elif args.command == "generate-db":
            _print_json(generate_database(args.db, customers=args.customers, products=args.products, orders=args.orders,
                                          avg_items_per_order=args.avg_items, graph_nodes=args.graph_nodes,
                                          edges_per_node=args.edges_per_node,
                                          item_payload_bytes=args.item_payload_bytes,
                                          seed=args.seed, overwrite=args.overwrite))
        elif args.command == "trace":
            _print_json(trace_workload(args.db, args.out, query_count=args.queries, seed=args.seed,
                                       cache_kib=args.cache_kib, mixed=not args.sequential,
                                       sparse_page_map=args.sparse_page_map))
        elif args.command == "train-sweep":
            seeds = _parse_int_list(args.seeds)
            _print_json(train_evaluate_sweep(args.sequences, args.page_map, args.out, seeds=seeds,
                                             split_seed=args.split_seed, epochs=args.epochs,
                                             max_candidates=args.candidates, hidden=args.hidden,
                                             semantic_ablations=_parse_ablation_specs(args.semantic_ablations)))
        elif args.command == "run-prefetch":
            model_dir = Path(args.model_dir).resolve()
            default_model = model_dir / ("model_b.pt" if args.strategy in {"model_b", "hybrid"} else "model_a.pt" if args.strategy == "model_a" else "")
            model_file = Path(args.model_path).resolve() if args.model_path else default_model
            config = {
                "strategy": args.strategy,
                "candidate_index_path": str(model_dir / "candidate_index.json"),
                "model_path": str(model_file) if args.strategy in {"model_a", "model_b", "hybrid"} else None,
                "semantic": args.strategy in {"model_b", "hybrid"},
                "cache_pages": args.cache_pages,
                "degree": args.degree,
                "confidence_threshold": args.threshold,
                "queue_size": args.queue_size,
                "predict_on": args.predict_on,
                "predict_every": args.predict_every,
                "miss_delay_ms": args.miss_delay_ms,
                "max_score_candidates": args.max_score_candidates,
                "storage_model": args.storage_model,
                "storage_channels": args.storage_channels or (4 if args.storage_model == "k_channel" else None),
                "drop_stale_jobs": args.drop_stale_jobs,
                "diagnostics_enabled": not args.no_diagnostics,
                "oracle_trace_path": str(Path(args.oracle_trace).resolve()) if args.oracle_trace else None,
                "oracle_miss_path": str(Path(args.oracle_miss_ref).resolve()) if args.oracle_miss_ref else None,
                "prefetch_workers": args.prefetch_workers,
            }
            _print_json(trace_workload(args.db, args.out, query_count=args.queries, seed=args.seed,
                                       cache_kib=args.cache_kib, mixed=True, prefetch_config=config))
        elif args.command == "run-suite":
            seeds = _parse_int_list(args.seeds)
            strategies = _parse_str_list(args.strategies)
            _print_json(run_suite(args.db, args.model_dir, args.out, queries=args.queries, seeds=seeds,
                                  strategies=strategies, cache_kib=args.cache_kib, cache_pages=args.cache_pages,
                                  degree=args.degree, threshold=args.threshold, miss_delay_ms=args.miss_delay_ms,
                                  queue_size=args.queue_size, predict_on=args.predict_on,
                                  predict_every=args.predict_every,
                                  max_score_candidates=args.max_score_candidates,
                                  storage_model=args.storage_model,
                                  storage_channels=args.storage_channels or (4 if args.storage_model == "k_channel" else None),
                                  drop_stale_jobs=args.drop_stale_jobs,
                                  diagnostics_enabled=not args.no_diagnostics,
                                  prefetch_workers=args.prefetch_workers))
        elif args.command == "runtime-sweep":
            predict_on_values = _parse_str_list(args.predict_on)
            invalid_predict_on = set(predict_on_values) - {"all", "miss", "hit"}
            if invalid_predict_on:
                raise ValueError(f"Unsupported --predict-on values: {sorted(invalid_predict_on)}")
            _print_json(run_runtime_sensitivity_sweep(
                args.db, args.model_dir, args.out, queries=args.queries,
                seeds=_parse_int_list(args.seeds), strategies=_parse_str_list(args.strategies),
                cache_kib=args.cache_kib, cache_pages_values=_parse_int_list(args.cache_pages),
                degree_values=_parse_int_list(args.degrees), queue_size_values=_parse_int_list(args.queue_sizes),
                predict_on_values=predict_on_values, predict_every_values=_parse_int_list(args.predict_every),
                threshold=args.threshold, miss_delay_ms=args.miss_delay_ms, baseline=args.baseline,
            ))
        elif args.command == "summarize-suite":
            report = json.loads(Path(args.runs).read_text(encoding="utf-8"))
            summary = summarize_suite(report)
            out = Path(args.out).resolve()
            out.mkdir(parents=True, exist_ok=True)
            (out / "suite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            if summary["strategies"]:
                import csv
                with (out / "suite_summary.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(summary["strategies"][0].keys()))
                    writer.writeheader()
                    writer.writerows(summary["strategies"])
            _print_json({"summary_path": str(out / "suite_summary.json"), "summary": summary})
        elif args.command == "recover-suite":
            _print_json(recover_suite(args.out))
        elif args.command == "report":
            _print_json(build_markdown_report(args.input, args.out, top_n=args.top_n))
        elif args.command == "research-report":
            _print_json(build_consolidated_report(args.run_dir, args.out,
                                                   optimized_suite_dir=args.optimized_suite))
        elif args.command == "dashboard-export":
            _print_json(write_dashboard_snapshot(args.run_dir, args.out))
        elif args.command == "report-v21":
            from .report_v21 import build_v21_report
            _print_json(build_v21_report(args.study_dir, args.out))
        elif args.command == "dashboard-export-v21":
            from .report_v21 import build_v21_dashboard
            _print_json(build_v21_dashboard(args.study_dir, args.out))
        elif args.command == "build-sequences":
            seqs = build_sequences(args.trace, args.page_map, args.out)
            _print_json({"sequence_count": len(seqs), "output": str(Path(args.out).resolve()),
                         "total_page_requests": sum(len(s["pages"]) for s in seqs)})
        elif args.command == "train":
            _print_json(train_models(args.sequences, args.page_map, args.out, seed=args.seed,
                                     epochs=args.epochs, max_candidates=args.candidates, hidden=args.hidden,
                                     semantic_ablations=_parse_ablation_specs(args.semantic_ablations)))
        elif args.command == "evaluate":
            _print_json(evaluate_predictors(args.sequences, args.page_map, args.candidate_index,
                                            args.model_a, args.model_b, args.out, top_k=args.top_k,
                                            semantic_ablations=_parse_ablation_specs(args.semantic_ablations),
                                            retrained_ablation_dir=args.retrained_ablation_dir))
        elif args.command == "offline-study-v21":
            from .offline import run_offline_study_v21
            seeds = _parse_int_list(args.seeds)
            report = run_offline_study_v21(args.sequences, args.workload, args.page_map, args.out,
                                           seeds=seeds, epochs=args.epochs, hidden=args.hidden,
                                           max_candidates=args.candidates, top_k=args.top_k)
            if args.ledger:
                from . import runledger
                runledger.record_run(args.ledger,
                                     {"study": "offline-study-v21", "seeds": seeds, "epochs": args.epochs,
                                      "hidden": args.hidden, "candidates": args.candidates, "top_k": args.top_k},
                                     status="completed", outputs=[str(Path(args.out) / "offline_study_v21.json")],
                                     metrics={"family_members": len(report.get("family_reports", {}).get("params", {}).get("members", {}))})
            _print_json(report)
        elif args.command == "headroom-v21":
            delays = _parse_str_list(args.delays)
            report = run_headroom_v21(
                args.reference, args.out,
                caches=_parse_int_list(args.caches),
                delays=[float(v) for v in delays],
                channels=_parse_int_list(args.channels),
                storage_model=args.storage_model, degree=args.degree,
                threshold_pct=args.threshold_pct, ledger_path=args.ledger)
            _print_json(report)
        elif args.command == "sim-validate-cells":
            _print_json(diagnose_sim_cells(args.calibration, args.runs, out_dir=args.out,
                                           ledger_path=args.ledger))
        elif args.command == "offline-study":
            _print_json(run_offline_study(args.sequences, args.workload, args.page_map, args.out,
                                          seeds=_parse_int_list(args.seeds), epochs=args.epochs,
                                          hidden=args.hidden, max_candidates=args.candidates, top_k=args.top_k))
        elif args.command == "offline-ablation":
            _print_json(run_ablation(args.sequences, args.workload, args.page_map, args.out,
                                     seeds=_parse_int_list(args.seeds), split_kind=args.split,
                                     epochs=args.epochs, hidden=args.hidden, max_candidates=args.candidates))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
