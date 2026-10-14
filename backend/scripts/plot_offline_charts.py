"""Generate the v2 offline charts (accuracy by policy, Model B feature ablation).

Usage: python scripts/plot_offline_charts.py [--run-dir runs/m2-offline]
Writes SVG and PNG charts into <run-dir>/charts/.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POLICY_ORDER = ["markov1", "markov2", "markov_ctx", "model_a", "model_b", "model_b_table"]
NEURAL = {"model_a", "model_b", "model_b_table"}


def plot_accuracy(run_dir: pathlib.Path, top_values: dict[str, float], out_path: pathlib.Path) -> None:
    labels = [p for p in POLICY_ORDER if p in top_values]
    values = [top_values[p] for p in labels]
    colors = ["#d9564f" if p in NEURAL else "#4c78a8" for p in labels]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("mean top-1 accuracy (%)")
    ax.set_title("Offline next-page prediction on held-out executions")
    ax.set_ylim(0, 100)
    ax.axhline(12.5, color="grey", ls="--", lw=0.8,
               label="uniform guess over 8 pages")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}",
                ha="center", va="bottom", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".svg"))
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def plot_ablation(run_dir: pathlib.Path, rows: list[dict], out_path: pathlib.Path) -> None:
    ordered = sorted(rows, key=lambda row: row["mean_top1_accuracy_pct"])
    labels = [row["variant"] for row in ordered]
    values = [row["mean_top1_accuracy_pct"] for row in ordered]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = ["#d9564f" if row["variant"] == "full" else ("#7f7f7f" if row["variant"] == "no_page_metadata" else "#4c78a8")
              for row in ordered]
    bars = ax.barh(labels, values, color=colors)
    full_value = next(row["mean_top1_accuracy_pct"] for row in ordered if row["variant"] == "full")
    ax.axvline(full_value, color="#d9564f", ls="--", lw=1.0)
    for bar, value in zip(bars, values):
        ax.text(value + 0.4, bar.get_y() + bar.get_height() / 2, f"{value:.1f}",
                va="center", fontsize=9)
    ax.set_xlabel("mean top-1 accuracy (%)")
    ax.set_title("Model B semantic-feature ablation (held-out parameters)")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".svg"))
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render v2 offline study charts")
    parser.add_argument("--run-dir", default="runs/m2-offline")
    args = parser.parse_args()
    run_dir = pathlib.Path(args.run_dir)
    charts = run_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    accuracy_path = run_dir / "offline_accuracy.json"
    ablation_path = run_dir / "ablation" / "model_b_ablation.json"

    if accuracy_path.exists():
        report = json.loads(accuracy_path.read_text(encoding="utf-8"))
        for split in ("params", "templates"):
            top_values = report["splits"][split]["mean_top1_accuracy_pct"]
            if any(v is not None for v in top_values.values()):
                plot_accuracy(run_dir, {k: v for k, v in top_values.items() if v is not None},
                              charts / f"accuracy_{split}")
                print("wrote", charts / f"accuracy_{split}.png")
    if ablation_path.exists():
        ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
        plot_ablation(run_dir, ablation["rows"], charts / "model_b_ablation")
        print("wrote", charts / "model_b_ablation.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
