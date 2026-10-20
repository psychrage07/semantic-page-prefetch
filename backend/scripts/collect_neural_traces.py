"""Collect independently seeded read-only VFS workloads into new study directories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pagewise.dataset import build_sequences
from pagewise.runner import trace_workload
from pagewise.runledger import record_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seeds", default="102,103,104,105,106,107,108,109,110")
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--static-metadata", type=Path, default=None,
                        help="Reuse the full static dbstat snapshot instead of rescanning the database")
    args = parser.parse_args()
    for seed in [int(x) for x in args.seeds.split(",")]:
        target = args.out / f"trace-seed{seed}"
        config = {"stage": "trace", "db": str(args.db.resolve()), "queries": args.queries,
                  "static_metadata": str(args.static_metadata) if args.static_metadata else None}
        if (target / "sequences.jsonl").exists():
            print(f"EXISTS seed={seed}; preserving existing trace", flush=True)
            continue
        try:
            trace_workload(args.db, target, query_count=args.queries, seed=seed, sparse_page_map=True,
                           static_metadata_path=args.static_metadata)
            seqs = build_sequences(target / "trace.jsonl", target / "page_map.json", target / "sequences.jsonl")
            record_run(args.out / "trace_ledger.jsonl", config, seed=seed, status="completed",
                       outputs=[str(target / "sequences.jsonl")],
                       metrics={"queries": len(seqs), "reads": sum(len(s["pages"]) for s in seqs)})
            print(f"TRACED seed={seed}: {len(seqs)} queries", flush=True)
        except Exception as exc:
            record_run(args.out / "trace_ledger.jsonl", config, seed=seed, status="failed", reason=repr(exc))
            raise


if __name__ == "__main__":
    main()
