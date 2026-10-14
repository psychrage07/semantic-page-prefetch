"""Generate the Review-2 (Lab Assessment 9) figures for the Pagewise report.

Outputs PNG files into backend/review2/figures/:
  fig1_pipeline.png          - end-to-end experimental pipeline
  fig2_architecture.png      - runtime architecture (demand thread / worker pool / device)
  fig3_module_workflows.png  - workflow diagrams for the four major modules
  fig4_offline_accuracy.png  - measured held-out next-page accuracy (from the dashboard snapshot)
  fig5_paired_deltas.png     - measured paired latency deltas vs `none` (from the dashboard snapshot)

All canvases are about 8 inches wide so that, printed at the 6.2-inch text width of an A4
page with 1-inch margins, no label falls below roughly 6 pt. Figures 4 and 5 read
``public/pagewise-dashboard.json`` so no number is typed by hand.

Run from anywhere:  python backend/review2/diagrams.py
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DASHBOARD = HERE.parents[1] / "public" / "pagewise-dashboard.json"

C_INPUT = "#dbeafe"   # blue   - tracing / input
C_PROC = "#e0e7ff"    # indigo - runtime / processing
C_MODEL = "#fce7f3"   # pink   - learning
C_STORE = "#dcfce7"   # green  - storage / data
C_OUT = "#fef9c3"     # yellow - outputs / analysis
C_NOTE = "#f8fafc"
EDGE = "#1e3a5f"
TEXT = "#111827"
MUTED = "#374151"

plt.rcParams["font.family"] = "DejaVu Sans"


def box(ax, x, y, w, h, text, color, fs=8.5, weight="normal", align="center", radius=0.6, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={radius}",
                                fc=color, ec=EDGE, lw=lw))
    if text:
        if align == "center":
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                    weight=weight, color=TEXT, linespacing=1.25)
        else:
            ax.text(x + 1.0, y + h / 2, text, ha="left", va="center", fontsize=fs,
                    weight=weight, color=TEXT, linespacing=1.25)


def titled_box(ax, x, y, w, h, title, body, color, fs_title=9, fs_body=7.6, body_offset=2.6):
    """Rounded box with a bold title at the top and the body text directly beneath it."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.6",
                                fc=color, ec=EDGE, lw=0.9))
    ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=fs_title,
            weight="bold", color=TEXT, linespacing=1.15)
    title_lines = title.count("\n") + 1
    ax.text(x + w / 2, y + h - 1.3 - body_offset * title_lines, body, ha="center", va="top",
            fontsize=fs_body, color=MUTED, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, label=None, fs=7, lx=None, ly=None, color=EDGE, lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))
    if label:
        tx = (x1 + x2) / 2 if lx is None else lx
        ty = (y1 + y2) / 2 if ly is None else ly
        ax.text(tx, ty, label, ha="center", va="center", fontsize=fs, color=EDGE,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))


def elbow(ax, points, lw=1.1, color=EDGE):
    """Poly-line through `points` with an arrow head on the last segment."""
    for (x1, y1), (x2, y2) in zip(points[:-2], points[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, solid_capstyle="round")
    (x1, y1), (x2, y2) = points[-2], points[-1]
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


# =============================================================== fig 1: pipeline
fig, ax = plt.subplots(figsize=(8, 6.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 67.5)
ax.axis("off")

stages = [
    ("1. Database\ngenerator", C_STORE,
     "database.py\nsynthetic e-commerce DB\n6 tables, 6 indexes\n4 KiB pages, fixed seed",
     "pagewise-medium.sqlite\n+ .metadata.json"),
    ("2. Workload +\nVFS trace", C_INPUT,
     "workloads/tracing/runner\n6 SQL templates + plans\nAPSW TraceVFS xRead log\ndbstat page map",
     "trace.jsonl, page_map.json\nplans.json, workload.jsonl"),
    ("3. Page\nsequences", C_INPUT,
     "dataset.py\nbyte ranges \u2192 pages\ngrouped per execution\npage metadata attached",
     "sequences.jsonl"),
    ("4. Train\nrankers", C_MODEL,
     "models.py\ncandidate index (train)\nModel A: history MLP\nModel B: semantic MLP",
     "model_a.pt, model_b.pt\ncandidate_index.json"),
    ("5. Offline\nevaluation", C_MODEL,
     "evaluate/sweep/offline\ntop-1 / top-5 accuracy\ncandidate recall\nmulti-seed sweeps\nsemantic ablations",
     "predictor_metrics.json\ntraining_sweep_report.json"),
    ("6. Prefetch\nruntime", C_PROC,
     "runtime.py\nbounded LRU page cache\nasync queue + workers\nsimulated storage",
     "runtime_stats.json\npredictions.jsonl"),
    ("7. Suite +\nanalysis", C_OUT,
     "suite / reporting\npaired per-query deltas\nbootstrap 95% CIs\nreports + dashboard",
     "suite_summary.json\nresearch_report.md"),
]
w, h, gap = 22.0, 17.0, 2.5
row_y = {0: 47.0, 1: 18.0}
positions = []
for i, (title, color, sub, artefact) in enumerate(stages):
    row, col = (0, i) if i < 4 else (1, i - 4)
    x = 1.5 + col * (w + gap)
    y = row_y[row]
    titled_box(ax, x, y, w, h, title, sub, color, fs_title=10, fs_body=7.4, body_offset=2.45)
    ax.text(x + w / 2, y - 1.5, artefact, ha="center", va="top", fontsize=7.2, color=MUTED,
            style="italic", linespacing=1.25)
    positions.append((x, y))
    if col < 3 and i != 6:
        arrow(ax, x + w, y + h / 2, x + w + gap, y + h / 2)

# connector from stage 4 (end of row 1) round to stage 5 (start of row 2)
x4, y4 = positions[3]
x5, y5 = positions[4]
elbow(ax, [(x4 + w, y4 + h / 2), (98.8, y4 + h / 2), (98.8, 41.0), (x5 + w / 2, 41.0), (x5 + w / 2, y5 + h)])

# note box in the free slot of row 2
titled_box(ax, 1.5 + 3 * (w + gap), 18.0, w, h, "Run-time coupling",
           "the observed demand-page\nstream is the live input\nto the predictors; stage\n6 loads stage-4 models,\nstage 7 repeats stage 6\nper seed and strategy",
           C_NOTE, fs_title=10, fs_body=7.4, body_offset=2.45)

# experiment-control strip
box(ax, 1.5, 0.5, 96.5, 12.0, "", C_NOTE)
ax.text(49.75, 10.7, "Controlled experiment configuration (identical for every strategy within a trial)",
        ha="center", va="center", fontsize=9, weight="bold", color=EDGE)
ax.text(49.75, 5.4,
        "bounded LRU cache of 32 pages (final protocol; 32\u2013512 in the headroom grid)   |   prefetch degree K = 2\n"
        "injected miss delay: 0.2 / 1.0 ms (final categories); 0 ms = real operating-system storage path\n"
        "storage model: shared (serialised) or independent   |   1\u20134 prefetch workers\n"
        "seeds 901\u2013903, 100 queries per seed, strategy order rotated per seed",
        ha="center", va="center", fontsize=7.3, color=TEXT, linespacing=1.55)

ax.set_title("Figure 1. Pagewise end-to-end experimental pipeline", fontsize=11, weight="bold", pad=4)
fig.tight_layout()
fig.savefig(OUT / "fig1_pipeline.png", dpi=300)
plt.close(fig)

# ============================================================ fig 2: architecture
fig, ax = plt.subplots(figsize=(8.6, 7.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 79)
ax.axis("off")

LX, LW = 1.5, 29.5     # left column  (demand path)
MX, MW = 37.5, 29.5    # middle column (prefetch path)
RX, RW = 73.0, 25.5    # right column (shared components)
FT, FB = 8.0, 7.2      # title / body font sizes

for cx, label in ((LX + LW / 2, "Demand path (SQLite query thread)"),
                  (MX + MW / 2, "Asynchronous prefetch path (workers)"),
                  (RX + RW / 2, "Shared components")):
    ax.text(cx, 76.3, label, ha="center", va="center", fontsize=8.6, weight="bold", color=EDGE)

# ---- left column: demand path
titled_box(ax, LX, 63, LW, 9, "Workload driver (runner.py)",
           "6 parameterised SQL templates\nper-query context: query_id,\ntemplate, workload family, plan class", C_INPUT, FT, FB, 2.3)
titled_box(ax, LX, 50, LW, 9, "APSW connection + TraceVFS",
           "tracing.py: xRead(offset, amount)\n\u2192 page ids; PRAGMA mmap_size = 0\ntiming, hit / miss, context \u2192 trace", C_INPUT, FT, FB, 2.3)
titled_box(ax, LX, 33, LW, 13, "BoundedPageCache (runtime.py)",
           "thread-safe LRU of N pages (RLock)\nhit \u2192 bytes served from the cache\nmiss \u2192 demand-read path\ncounts useful / wasted prefetches,\nevictions, lead time to demand use", C_STORE, FT, FB, 2.3)
titled_box(ax, LX, 20, LW, 9, "Demand-miss path",
           "device.access('demand', 1 page)\n\u2192 base VFS xRead of the page\n\u2192 cache.put(page, origin='demand')", C_PROC, FT, FB, 2.3)

# ---- storage device spanning the demand and prefetch paths
titled_box(ax, LX, 2, MX + MW - LX, 11, "SimulatedStorageDevice (runtime.py) in front of the read-only SQLite file",
           "shared: one serialised channel \u2014 a demand miss waits behind any in-flight prefetch read\n"
           "independent: per-access sleep, unlimited parallelism (diagnostics only)\n"
           "delay 0 ms: real operating-system storage path\n"
           "injected delay is always applied outside the cache and runtime locks", C_STORE, FT, FB, 2.3)

# ---- middle column: prefetch path
titled_box(ax, MX, 50, MW, 9, "PredictiveRuntime (runtime.py)",
           "bounded FIFO job queue (queue_size)\nduplicates coalesced; drop when full\nactive-query id marks stale jobs", C_MODEL, FT, FB, 2.3)
ax.add_patch(FancyBboxPatch((MX, 16), MW, 30, boxstyle="round,pad=0.0,rounding_size=0.6",
                            fc=C_MODEL, ec=EDGE, lw=0.9))
ax.text(MX + MW / 2, 44.8, "Prefetch worker pool (1\u20134 threads)", ha="center", va="top",
        fontsize=FT, weight="bold", color=TEXT)
steps = [
    "1  dequeue job; measure queue wait;\n    drop if stale (--drop-stale-jobs)",
    "2  rank candidates with the predictor",
    "3  prune cached / in-flight pages;\n    cap the neural scoring candidates",
    "4  read top-K pages via a separate\n    file handle; device.access('prefetch')",
    "5  cache.put(page, origin='prefetch')\n    with per-stage monotonic timestamps",
]
yy = 41.6
for step in steps:
    lines = step.count("\n") + 1
    hh = 2.9 + 1.9 * (lines - 1)
    box(ax, MX + 1.2, yy - hh, MW - 2.4, hh, step, "white", fs=6.9, align="left", radius=0.3, lw=0.6)
    yy -= hh + 0.7

# ---- right column: shared components
titled_box(ax, RX, 49, RW, 10, "Candidate index + page map",
           "candidate_index.json (train split)\npage_map.json; SHA-256 fingerprint\nverified at start-up", C_STORE, FT, 7.0, 2.3)
titled_box(ax, RX, 22, RW, 24, "Page predictors\n(models.py / runtime.py)",
           "sequential: next page id\nmarkov1 / markov2: 1st / 2nd-order\nsuccessor tables\nmodel_a: history MLP, 19 features\n"
           "model_b: semantic MLP, 86 features\nhybrid: Markov-2 for known\ncontexts, Model B otherwise\ncontrols: null, oracle_next,\noracle_miss", C_OUT, FT, 7.0, 2.3)
titled_box(ax, RX, 2, RW, 16, "Diagnostics (per run)",
           "predictions.jsonl (job timeline)\nprefetch_uses.jsonl (lead times)\nruntime_stats.json, trace.jsonl\nquery_results.csv\nrun_metadata.json", C_OUT, FT, 7.0, 2.3)

# ---- arrows: demand path
cxl = LX + LW / 2
arrow(ax, cxl, 63, cxl, 59, "SQL + context", 6.8, lx=cxl - 8.5, ly=61)
arrow(ax, cxl, 50, cxl, 46, "page requests", 6.8, lx=cxl - 8.5, ly=48)
arrow(ax, cxl, 33, cxl, 29, "miss", 6.8, lx=cxl - 4, ly=31)
arrow(ax, cxl, 20, cxl, 13, "demand read (1 page)", 6.8, lx=cxl, ly=16.5)
# ---- arrows: prefetch path
cxm = MX + MW / 2
arrow(ax, LX + LW, 54.5, MX, 54.5, "prediction job per observed page", 6.6, lx=(LX + LW + MX) / 2, ly=61.2)
arrow(ax, cxm, 50, cxm, 46, "FIFO", 6.8, lx=cxm + 4, ly=48)
arrow(ax, MX, 40, LX + LW, 40, "prefetched\npages", 6.6, lx=(LX + LW + MX) / 2, ly=42.3)
arrow(ax, cxm, 16, cxm, 13, "prefetch read (top-K pages)", 6.8, lx=cxm, ly=14.5)
# ---- arrows: shared components
arrow(ax, RX + RW / 2, 49, RX + RW / 2, 46, None)
arrow(ax, RX, 34, MX + MW, 34, "ranked\ncandidates", 6.6, lx=(MX + MW + RX) / 2, ly=36.4)
arrow(ax, MX + MW, 17.5, RX, 17.5, "per-job\ntimestamps", 6.6, lx=(MX + MW + RX) / 2, ly=19.9)

ax.set_title("Figure 2. Runtime architecture: demand thread, asynchronous prefetch worker pool\nand shared storage device",
             fontsize=10.5, weight="bold", pad=4)
fig.tight_layout()
fig.savefig(OUT / "fig2_architecture.png", dpi=300)
plt.close(fig)

# ======================================================= fig 3: module workflows
fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.4))
panels = [
    ("Tracing module\n(database, workloads, metadata, tracing)", C_INPUT, [
        "generate-db: 6 tables, 6 indexes, 4 KiB pages",
        "generate_workload: 6 templates, skewed params",
        "EXPLAIN QUERY PLAN per template \u2192 plan class",
        "dbstat page map: object, path, type, depth",
        "TraceVFS.xRead: pages, timing, hit/miss, context",
        "Outputs: trace, page_map, plans, workload (JSON)",
    ]),
    ("Learning module\n(dataset, models, sweep, offline)", C_MODEL, [
        "build_sequences: pages per query execution",
        "Split by execution 70/15/15 (v2 adds 2 splits)",
        "CandidateIndex from training executions (cap 24)",
        "Features: A = 19 history; B = 19 + 67 semantic",
        "Matched MLP rankers: BCE, AdamW, early stopping",
        "Top-1/top-5, candidate recall, sweeps, ablations",
    ]),
    ("Runtime module\n(runtime, tracing, runner)", C_PROC, [
        "xRead \u2192 cache.get(); miss \u2192 device read, put()",
        "Per observed page: should_predict \u2192 request job",
        "Queue: coalesce, drop when full, flag stale jobs",
        "Worker: rank, prune cached/pending, cap scoring",
        "Read top-K via device; put(origin=prefetch)",
        "Lead time on use; predictions + runtime_stats",
    ]),
    ("Evaluation module\n(suite, reporting, sim, dashboard)", C_OUT, [
        "run-suite: same queries per seed, rotated order",
        "Per-query latency; p50/p95/total; cache counters",
        "Paired \u0394 vs none/stock; 2,000-resample 95% CI",
        "Training sweeps, ablations, v2 offline study",
        "Virtual-time simulator: calibrate, headroom grid",
        "suite_summary.json, research report, dashboard",
    ]),
]
for ax, (title, color, steps) in zip(axes.ravel(), panels):
    n = len(steps)
    pitch, bh = 1.55, 1.05
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n * pitch + 0.4)
    ax.axis("off")
    ax.set_title(title, fontsize=9, weight="bold", color=EDGE, pad=5, linespacing=1.15)
    y_top = n * pitch + 0.25
    for i, step in enumerate(steps):
        y = y_top - i * pitch
        box(ax, 0.05, y - bh, 9.9, bh, f"{i + 1}.  {step}", color if i % 2 == 0 else "white",
            fs=7.9, align="left", radius=0.18, lw=0.8)
        if i < n - 1:
            arrow(ax, 5.0, y - bh, 5.0, y - pitch + 0.02, None, lw=1.0)

fig.suptitle("Figure 3. Workflow diagrams for the four major modules", fontsize=11, weight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.4, w_pad=1.0)
fig.savefig(OUT / "fig3_module_workflows.png", dpi=300)
plt.close(fig)

# ======================================= fig 4 / fig 5: measured results (dashboard)
if DASHBOARD.exists():
    data = json.loads(DASHBOARD.read_text(encoding="utf-8"))

    # ---- fig 4: offline accuracy on the held-out split
    policies = {row["policy"]: row for row in data["prediction"]["policies"]}
    order = ["sequential", "model_a", "model_b", "markov1", "markov2"]
    labels = {"sequential": "Sequential", "model_a": "Model A\n(history MLP)", "model_b": "Model B\n(semantic MLP)",
              "markov1": "Markov-1", "markov2": "Markov-2"}
    top1 = [policies[p]["top1_accuracy_pct"] for p in order]
    top5 = [policies[p]["top5_accuracy_pct"] for p in order]
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    xs = range(len(order))
    bw = 0.38
    b1 = ax.bar([i - bw / 2 for i in xs], top1, bw, label="Top-1 accuracy (%)", color="#3b82f6")
    b2 = ax.bar([i + bw / 2 for i in xs], top5, bw, label="Top-5 accuracy (%)", color="#a5b4fc")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}",
                    ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([labels[p] for p in order], fontsize=9.5)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel("accuracy on held-out executions (%)", fontsize=9.5)
    ax.set_ylim(0, 88)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    n_pred = policies["model_b"]["evaluated_next_page_predictions"]
    n_exec = data["prediction"]["test_executions"]
    ax.set_title(f"Figure 4. Held-out next-page prediction accuracy\n({n_exec} test executions, {n_pred} predictions)",
                 fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_offline_accuracy.png", dpi=300)
    plt.close(fig)

    # ---- fig 5: paired latency deltas vs none for the three final categories
    suites = {s["name"]: s for s in data["runtime_suites"]}
    cats = [("final-latency-0p2ms", "Category A \u2014 0.2 ms injected miss delay (latency-injected)"),
            ("final-latency-1p0ms", "Category A \u2014 1.0 ms injected miss delay (latency-injected)"),
            ("final-real-storage", "Category B \u2014 real storage, no injected delay")]
    strat_order = ["stock", "sequential", "markov1", "markov2", "model_a", "model_b", "hybrid"]
    nice = {"stock": "stock SQLite", "sequential": "sequential", "markov1": "markov1", "markov2": "markov2",
            "model_a": "model_a", "model_b": "model_b", "hybrid": "hybrid"}
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2))
    for ax, (name, title) in zip(axes, cats):
        rows = {r["strategy"]: r for r in suites[name]["paired_query_comparisons_vs_none"]}
        lo_all = min(r["delta_ci95_low_ms"] for r in rows.values())
        hi_all = max(r["delta_ci95_high_ms"] for r in rows.values())
        span = max(hi_all, 0) - min(lo_all, 0)
        x_min, x_max = min(lo_all, 0) - 0.22 * span, max(hi_all, 0) + 0.22 * span
        ax.set_xlim(x_min, x_max)
        pad = 0.015 * (x_max - x_min)
        ys = list(range(len(strat_order)))[::-1]
        for y, strat in zip(ys, strat_order):
            r = rows[strat]
            mean = r["mean_delta_ms"]
            lo, hi = r["delta_ci95_low_ms"], r["delta_ci95_high_ms"]
            significant = hi < 0 or lo > 0
            color = ("#16a34a" if mean < 0 else "#dc2626") if significant else "#9ca3af"
            ax.barh(y, mean, color=color, height=0.62)
            ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="none", ecolor="black", capsize=3, lw=1)
            if mean >= 0:
                ax.text(hi + pad, y, f"{mean:+.2f}", va="center", ha="left", fontsize=7.8, color="#111827")
            else:
                ax.text(lo - pad, y, f"\u2212{abs(mean):.2f}", va="center", ha="right", fontsize=7.8, color="#111827")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels([nice[s] for s in strat_order], fontsize=9)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_xlabel("paired mean latency delta vs none (ms); negative = faster", fontsize=8.5)
        ax.set_title(title, fontsize=9.5, weight="bold")
        ax.grid(axis="x", alpha=0.3)
    handles = [Patch(color="#16a34a", label="faster than none (CI excludes 0)"),
               Patch(color="#dc2626", label="slower than none (CI excludes 0)"),
               Patch(color="#9ca3af", label="inconclusive (CI includes 0)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.4, columnspacing=1.2)
    fig.suptitle("Figure 5. Paired per-query latency delta versus the custom-cache \u201cnone\u201d baseline\n"
                 "(seeds 901\u2013903, 300 matched queries; bar = mean, whisker = bootstrap 95% CI)",
                 fontsize=10, weight="bold")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95), h_pad=1.2)
    fig.savefig(OUT / "fig5_paired_deltas.png", dpi=300)
    plt.close(fig)
else:
    print("dashboard snapshot not found; skipping figures 4 and 5")

print("figures written to", OUT)
