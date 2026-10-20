"""Combine matched validation-only trials without hiding superseded attempts."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pagewise.runledger import record_run, source_hash


def assemble(sources, out):
    if out.exists():
        raise ValueError("Assembly output exists; preserve it")
    out.mkdir(parents=True)
    report = {"source_sha256": source_hash(), "evidence_class": "validation-only selection",
              "test_partition_evaluated": False, "sources": [], "splits": {}}
    for source in sources:
        source_report = json.loads((source / "screen.json").read_text(encoding="utf-8"))
        if source_report.get("test_partition_evaluated"):
            raise ValueError("Screen has evaluated the test partition")
        report["sources"].append({"path": str(source), "report_sha256": hashlib.sha256((source / "screen.json").read_bytes()).hexdigest()})
        for split, trials in source_report["splits"].items():
            if "selected_model" not in trials:
                raise ValueError(f"Incomplete screen: {source}/{split}")
            target = out / split
            if split not in report["splits"]:
                target.mkdir()
                report["splits"][split] = {k: copy.deepcopy(v) for k, v in trials.items() if k not in {"models", "selected_model"}}
                report["splits"][split].update(models={}, attempts=[])
                for file in ("candidate_index.json", "page_map.json", "partition.json"):
                    shutil.copyfile(source / split / file, target / file)
            else:
                for file in ("candidate_index.json", "partition.json"):
                    if json.loads((source / split / file).read_text()) != json.loads((target / file).read_text()):
                        raise ValueError(f"Unmatched candidate/partition contract: {source}/{split}/{file}")
            result = report["splits"][split]
            for name, row in trials["models"].items():
                config = row["fit"]["config"]
                eligible = row["fit"]["best_epoch"] > 0 or config["prior_weight"] > 0
                result["attempts"].append({"source": str(source), "name": name, "eligible": eligible,
                                           "validation_top1_macro_pct": row["validation"]["top1_macro_pct"],
                                           "best_epoch": row["fit"]["best_epoch"]})
                if not eligible:
                    record_run(out / "run_log.jsonl", {"source": str(source), "split": split, "model": name},
                               status="excluded", reason="Non-residual checkpoint was random/untrained (epoch zero); use the retrained binary-loss control")
                    continue
                previous = result["models"].get(name)
                if previous is None or row["validation"]["top1_macro_pct"] > previous["validation"]["top1_macro_pct"]:
                    result["models"][name] = {**row, "source": str(source)}
                    shutil.copyfile(source / split / f"{name}.pt", target / f"{name}.pt")
    for split, result in report["splits"].items():
        winner = max(result["models"], key=lambda n: result["models"][n]["validation"]["top1_macro_pct"])
        result["selected_model"] = winner
        print(split, winner, result["models"][winner]["validation"]["top1_macro_pct"], flush=True)
    (out / "screen.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.sources, args.out)
