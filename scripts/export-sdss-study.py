"""Export verified, separately trained SDSS results for the read-only dashboard.

Usage: backend/.venv/Scripts/python scripts/export-sdss-study.py --experiment PATH
No training, database copying, or writes to the experiment directory occur.
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

REPO = Path(__file__).resolve().parents[1]


def export(experiment: Path):
    experiment = experiment.resolve()
    def read(name):
        return json.loads((experiment / name).read_text(encoding="utf-8"))

    manifest = read("artifacts.sha256.json")
    def verify(name):
        digest = hashlib.sha256((experiment / name).read_bytes()).hexdigest()
        if manifest.get(name.replace("/", "\\"), manifest.get(name, {})).get("sha256") != digest:
            raise ValueError(f"Experiment artifact hash mismatch: {name}")
        return digest

    sources = {
        "accuracy": "results/accuracy.json", "database": "data/database.json",
        "provenance": "data/provenance.json", "protocol": "protocol.json",
        "training": "models/freeze.json", "runtime": "results/runtime-smoke.json",
        "verification": "results/verification.json", "audit": "results/independent-audit.json",
    }
    hashes = {name: verify(name) for name in sources.values()}
    bundle = {key: read(name) for key, name in sources.items()}
    if not bundle["verification"].get("passed") or not bundle["audit"].get("passed"):
        raise ValueError("Only verified completed results can be exported")
    verify("results/per-query.csv")
    with (experiment / "results/per-query.csv").open(encoding="utf-8", newline="") as f:
        per_query = list(csv.DictReader(f))
    policies = bundle["accuracy"]["policies"]
    for name, metrics in policies.items():
        rows = [r for r in per_query if r["policy"] == name]
        n = sum(int(r["transitions"]) for r in rows)
        if len(rows) != metrics["queries"] or n != metrics["transitions"]:
            raise ValueError(f"Mismatched counts for {name}")
        macro = sum(float(r["top1_pct"]) for r in rows) / len(rows)
        micro = 100 * sum(int(r["top1_hits"]) for r in rows) / n
        top5 = 100 * sum(int(r["top5_hits"]) for r in rows) / n
        if any(abs(actual - metrics[key]) > 1e-9 for actual, key in [
            (macro, "top1_macro_pct"), (micro, "top1_micro_pct"), (top5, "top5_micro_pct")
        ]):
            raise ValueError(f"Mismatched measured scores for {name}")
    verify("results/prediction-samples.jsonl")
    bundle["prediction_samples"] = [json.loads(line) for line in
        (experiment / "results/prediction-samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    bundle.update(schema_version=1, study_id=experiment.name, completed_date="2026-10-27",
                  evidence_class="Saved external-dataset evaluation; trained separately",
                  source_hashes=hashes)
    target = REPO / "public/sdss-results"
    target.mkdir(parents=True, exist_ok=True)
    files = [
        ("results/accuracy.csv", "accuracy.csv", "Accuracy comparison"),
        ("results/per-query.csv", "per-query.csv", "All query scores"),
        ("results/prediction-samples.jsonl", "prediction-samples.jsonl", "Recorded next-page predictions"),
        ("models/freeze.json", "training.json", "Training history and frozen checkpoints"),
        ("results/runtime-smoke.json", "runtime.json", "Complete runtime measurements"),
        ("data/provenance.json", "provenance.json", "Dataset origin and download requests"),
        ("protocol.json", "protocol.json", "Experiment settings"),
        ("results/verification.json", "verification.json", "Verification checks"),
        ("results/independent-audit.json", "audit.json", "Independent score audit"),
        ("RESULTS.md", "RESULTS.md", "Readable experiment report"),
    ]
    bundle["downloads"] = []
    for source, name, label in files:
        verify(source)
        shutil.copyfile(experiment / source, target / name)
        bundle["downloads"].append({"label": label, "url": f"/sdss-results/{name}", "bytes": (target / name).stat().st_size})
    # Archive measured outputs and provenance only. Database and weights remain external.
    archive = target / "measured-outputs.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=6) as z:
        paths = [p for folder in ["traces", "results", "logs"]
                 for p in (experiment / folder).rglob("*") if p.is_file()]
        paths += [experiment / name for name in ["README.md", "RESULTS.md", "protocol.json",
            "data/provenance.json", "data/database.json", "models/freeze.json",
            "models/evaluation-seal.json", "models/freeze-complete.json", "source_snapshot/ORIGIN.json",
            "scripts/experiment.py", "scripts/frozen-experiment.py", "scripts/report.py", "environment.json", "requirements-lock.txt", "references/sources.json"]]
        for p in sorted(paths):
            rel = p.relative_to(experiment).as_posix()
            verify(rel)
            z.write(p, rel)
    bundle["downloads"].append({"label": "All measured outputs, raw traces and logs (ZIP)", "url": "/sdss-results/measured-outputs.zip", "bytes": archive.stat().st_size})
    (REPO / "public/pagewise-sdss-study.json").write_text(json.dumps(bundle, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Exported {len(policies)} predictors, {len(per_query)} query scores, {len(bundle['prediction_samples'])} recorded predictions")
    print(f"Raw-output archive: {archive.stat().st_size:,} bytes; database and weights remain in {experiment}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    export(parser.parse_args().experiment)
