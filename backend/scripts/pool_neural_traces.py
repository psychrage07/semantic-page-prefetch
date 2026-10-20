"""Pool development traces while preserving workload-seed and query identities."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--seeds", default="101,102,103,104,105,106")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    page_map, sequences, cases = {}, [], []
    for seed in [int(s) for s in args.seeds.split(",")]:
        root = args.study / f"trace-seed{seed}"
        page_map.update(json.loads((root / "page_map.json").read_text(encoding="utf-8")))
        workload = [json.loads(s) for s in (root / "workload.jsonl").read_text(encoding="utf-8").splitlines() if s]
        by_query = {str(c["query_id"]): c for c in workload}
        for line in (root / "sequences.jsonl").read_text(encoding="utf-8").splitlines():
            seq = json.loads(line)
            qid = str(seq["query_id"])
            seq.update(query_id=f"s{seed}:{qid}", workload_seed=seed, params=by_query[qid]["params"])
            sequences.append(seq)
        for case in workload:
            case.update(query_id=f"s{seed}:{case['query_id']}", workload_seed=seed)
            cases.append(case)
    for name, rows in (("sequences.jsonl", sequences), ("workload.jsonl", cases)):
        (args.out / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (args.out / "page_map.json").write_text(json.dumps(page_map), encoding="utf-8")
    print(json.dumps({"queries": len(sequences), "pages": len(page_map), "seeds": args.seeds}))


if __name__ == "__main__":
    main()
