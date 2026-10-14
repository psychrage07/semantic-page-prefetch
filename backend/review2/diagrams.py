"""Generate the three Review-2 (Lab Assessment 9) diagrams for the Pagewise report.

Outputs PNG files into backend/review2/figures/:
  fig1_pipeline.png     - end-to-end pipeline diagram
  fig2_architecture.png - runtime architecture (demand thread / worker pool / device)
  fig3_module_workflows.png - workflow diagrams for the four major modules
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C_INPUT = "#dbeafe"
C_PROC = "#e0e7ff"
C_MODEL = "#fce7f3"
C_STORE = "#dcfce7"
C_OUT = "#fef9c3"
EDGE = "#1e3a5f"


def box(ax, x, y, w, h, text, color, fs=8.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=color, ec=EDGE, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, wrap=True)


def arrow(ax, x1, y1, x2, y2, label=None, fs=7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=11, color=EDGE, lw=1.0))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center",
                fontsize=fs, color=EDGE)


# ---------------------------------------------------------------- fig 1: pipeline
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.axis("off")

stages = [
    ("1. Database\ngenerator", C_STORE,
     "Synthetic e-commerce SQLite DB\ncustomers/products/orders/edges"),
    ("2. VFS trace", C_INPUT,
     "APSW TraceVFS xRead capture\n+ dbstat page map + EXPLAIN plans"),
    ("3. Sequences", C_INPUT,
     "Byte ranges expanded to pages,\ngrouped per query execution"),
    ("4. Train", C_MODEL,
     "Model A (history-only) vs\nModel B (semantic) MLPs"),
    ("5. Evaluate", C_MODEL,
     "Top-1/top-5 accuracy,\ncandidate recall, ablations"),
    ("6. Runtime", C_PROC,
     "Bounded LRU cache + async\nprefetch worker pool"),
    ("7. Suite/analysis", C_OUT,
     "Paired per-query deltas,\nbootstrap CIs, reports"),
]
w, h, gap = 12.0, 7.5, 1.8
x = 1.5
centers = []
for i, (title, color, sub) in enumerate(stages):
    box(ax, x, 14, w, h, title, color, fs=9, weight="bold")
    ax.text(x + w / 2, 11.6, sub, ha="center", va="center", fontsize=6.4, color="#333")
    centers.append(x + w / 2)
    if i < len(stages) - 1:
        arrow(ax, x + w, 17.7, x + w + gap, 17.7)
    x += w + gap

# feedback arrow (runtime -> sequences: observed demand stream feeds tables)
arrow(ax, centers[5], 9.6, centers[2] + 2, 9.6)
ax.text((centers[5] + centers[2]) / 2, 9.9, "demand page stream feeds Markov/tables (online)",
        ha="center", fontsize=6.6, color=EDGE)

# row of experiment controls
box(ax, 1.5, 2.5, 97.5, 5.0, "", "#f8fafc")
ax.text(50, 5.9, "Controlled experiment configuration", ha="center", fontsize=8, weight="bold", color=EDGE)
ax.text(50, 3.9,
        "bounded LRU cache (16-512 pages)  |  miss-delay injection 0.2 / 0.5 / 1 / 2 / 5 / 10 ms  |  "
        "storage model shared / independent  |  prefetch workers 1-4  |  seeds paired per trial",
        ha="center", fontsize=7.2, color="#222")

ax.set_title("Pagewise end-to-end experimental pipeline (Review 2)", fontsize=11, weight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig1_pipeline.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------- fig 2: architecture
fig, ax = plt.subplots(figsize=(10.5, 6.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 58)
ax.axis("off")

# demand side
box(ax, 3, 42, 26, 7, "SQLite workload\n(6 SQL templates)", C_INPUT, 8.5, "bold")
box(ax, 3, 30, 26, 9, "APSW TraceVFS\n(custom VFS)\nxRead interception", C_INPUT, 8)
box(ax, 3, 17, 26, 10, "BoundedPageCache\nbounded LRU, N pages\nhit / miss bookkeeping", C_STORE, 8)
box(ax, 3, 4, 26, 10, "Demand miss:\ndevice.read -> page put\n-> cpu cost", C_PROC, 8)

# worker pool
box(ax, 36, 30, 30, 9, "PredictiveRuntime\nasync prediction queue\n(prefetch job per read)", C_MODEL, 8)
for i, y in enumerate((20, 12, 4)):
    box(ax, 36, y, 30, 6, f"Worker {i+1}\nfetch candidate pages -> device", C_MODEL, 7.5)

# device
box(ax, 72, 22, 25, 16, "SimulatedStorageDevice\n\nshared: one serialized\nchannel (lock)\nindependent: parallel\nsleep per access", C_STORE, 7.5)

# models
box(ax, 72, 42, 25, 7, "Page-predictors\nsequential / markov1 / markov2\nmodel_a / model_b / hybrid\noracle_* (reference only)", C_OUT, 7)

arrow(ax, 16, 42, 16, 39.2)             # workload -> VFS
arrow(ax, 16, 30, 16, 27)               # VFS -> cache
arrow(ax, 16, 17, 16, 14)               # cache -> miss path
arrow(ax, 29, 34.5, 36, 34.5, "prediction jobs", 7)
arrow(ax, 51, 26, 66, 30, "")           # workers -> device
arrow(ax, 51, 15, 66, 27)
arrow(ax, 51, 7, 66, 24)
arrow(ax, 66, 38, 66, 44, "")           # device -> predictors? actually model -> jobs
arrow(ax, 72, 45.5, 66, 36, "candidate pages", 6.5)
arrow(ax, 29, 22, 72, 20, "demand misses queue behind prefetch I/O (shared)", 6.5)
arrow(ax, 84.5, 22, 84.5, 16.5, "prefetch puts", 6.5)
ax.text(84.5, 15.6, "cache", ha="center", fontsize=6.5, color=EDGE)

ax.set_title("Runtime architecture: demand thread vs asynchronous prefetch worker pool",
             fontsize=11, weight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig2_architecture.png", dpi=300)
plt.close(fig)

# --------------------------------------------------- fig 3: module workflow panels
fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6))
panels = [
    ("Tracing module (metadata.py / tracing.py)", [
        "TraceVFS installed on APSW connection",
        "xRead(byte range) intercepted per query",
        "dbstat page map: page -> table/index/type/depth",
        "EXPLAIN QUERY PLAN per template",
        "trace.jsonl + page_map.json + plans.json",
    ]),
    ("Learning module (dataset.py / models.py)", [
        "Per-query page sequences (train/val/test split)",
        "Candidate dictionary from train executions only",
        "Model A: last-8 pages + deltas",
        "Model B: + page metadata, plan class, template, family",
        "MLP scoring per candidate; shared training budget",
    ]),
    ("Runtime module (runtime.py / runner.py)", [
        "Per demand read: enqueue prediction job",
        "Worker: rank candidates (skip cached/pending)",
        "Fetch top-K -> device access (shared/independent)",
        "put() into bounded LRU with origin=prefetch",
        "Demand use tracked with lead time",
    ]),
    ("Evaluation module (suite.py / sweep.py)", [
        "Multi-seed paired trials, rotating strategy order",
        "Per-query timings + p50/p95 aggregation",
        "Paired deltas vs none/stock baseline",
        "Bootstrap 95% CIs + improved-query %",
        "Reports: JSON / CSV / Markdown / dashboard",
    ]),
]
for ax, (title, steps) in zip(axes.ravel(), panels):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(steps) * 1.35 + 0.7)
    ax.axis("off")
    ax.set_title(title, fontsize=9.5, weight="bold")
    y = len(steps) * 1.35
    for i, step in enumerate(steps):
        color = C_PROC if i % 2 == 0 else C_INPUT
        box(ax, 0.4, y - 1.05, 9.2, 1.05, step, color, fs=7.4)
        if i < len(steps) - 1:
            arrow(ax, 5.0, y - 1.1, 5.0, y - 1.55)
        y -= 1.35

fig.suptitle("Workflow diagrams for the four major modules", fontsize=11, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT / "fig3_module_workflows.png", dpi=300)
plt.close(fig)

from matplotlib import pyplot as plt  # noqa: E402  (import placed late to allow use above)

print("figures written to", OUT)
