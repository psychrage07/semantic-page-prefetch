# Pagewise faculty walkthrough

Open `/?view=workspace` for the visible SQL editor, database explorer and live terminal. Choose an example query, click **Run query**, then inspect the output tabs. **Compare models** runs the default methods on the same query. Optional parameters are under **More options**; the terminal and workspace remain visible.

Start from the repository root:

```powershell
npm run build
npm run preview:local
```

Keep this terminal running. Open `http://localhost:4173/?view=overview`.

## 1. Project overview

Say: “SQLite reads database pages. We predict the next page and fetch it early. We measure both prediction quality and the complete query time.”

Explain that the selected predictor combines a Markov prior with a semantic neural correction. The saved final study on new query values shows 47.74% query-average Top-1 for our hybrid and 44.83% for Markov-2: a 2.92 percentage-point improvement. These numbers describe the tested database and workloads.

## 2. How it works

Walk through Observe → Predict → Prefetch → Measure. Explain that accurate predictions can still be too expensive or too late to make a query faster.

## 3. Live demonstration

The default **Test trained hybrid** mode runs the actual selected model. Choose 20 queries per policy/seed and click **Run trained hybrid**. It verifies the existing frozen checkpoint, predictor source and matching 8.62 GB database, then executes the same fresh workload for no prefetch, Markov-2 and the selected neural hybrid. Two new workload seeds give 120 query executions. The nine stages show verification, six actual policy runs, prediction evaluation and the result export. No weights are trained or changed.

Results show fresh mean/median query times, scored prediction jobs, prefetched/useful page counts and actual candidate samples. A separate offline stage runs the selected semantic model on every eligible transition from the fresh baseline traces, compares predictions with the observed next pages and reports Top-1/Top-5 accuracy and candidate recall. The overview, research results, comparison page and prediction inspector share these fresh computed outputs. This is fresh workload evaluation, not a new held-out accuracy claim. Parameters may overlap training; diagnostics are enabled and the OS cache is not reset. The saved 47.74% score stays in Research results.

The separate **Build a small demo** mode creates its own database, trains Model A and Model B and evaluates the standard policies. Choose 100 traced queries and click **Start live demo** for that seven-stage training pipeline. Its standard hybrid is separate from the selected neural hybrid. The input changes the tracing stage; runtime uses 30 queries per policy/seed, eight policies and two seeds.

A previously completed experiment is labelled **Previous local run**, with its timestamp. Do not assume the neural model wins. **Download results** saves the measured export. Local frozen research artifacts are required for the trained-model test; the small demo remains available on a fresh clone.

## 4. Research results

Use **New query values** and **New query types** to explain the generalization boundary. Unfamiliar query types remain difficult; the selected template-split policy is a Markov fallback without a learned neural increment.

Show the runtime table: no measured speedup was established. The requested delays are artificial and the channel counts are abstract service slots. These are not physical disk benchmarks.

Use **Full research evidence** only for questions about Top-5, candidate recall, statistical uncertainty or detailed runtime counters. Individual commands remain under **Advanced research tools** in the live demo. Historical results and fixture previews are under **Research workspace** in the sidebar.

## Interactive SQL and terminal walkthrough

Open `/?view=workspace` for the database explorer, SQL editor, live page reads, next-page predictions, prefetch monitor, SQL benchmark and real Python terminal output. Build a fresh small demo, select a table, load a query template, run it and inspect the tabs. Download actual SQLite/CSV/JSON/JSONL artifacts under Files. The explicit research database option uses the matching frozen neural checkpoint and saves new outputs separately. See [WORKSPACE_GUIDE.md](WORKSPACE_GUIDE.md) for the complete walkthrough and measurement scope.

The main result cards now report prediction accuracy. In the workspace, Compare models calculates Top-1/Top-5 from that run's actual page trace; its scoring/sampling scope is printed below the table. Timing and prefetch-cost details are expandable. Better prediction accuracy is the demonstrated improvement; runtime measurements have not established a query speedup.

## SDSS dataset results

Open `/?view=sdss` or choose **SDSS dataset results** in the sidebar. Explain: "We also trained a separate copy of our hybrid on real astronomy data from SDSS DR7, a dataset used by SeLeP. On this bounded subset and our generated SQLite workload, the hybrid reached 35.32% query-average Top-1 against Markov-2 at 27.32%. The original project database and model were kept separate."

The page keeps all seven predictors visible. Query-average Top-1 gives each query equal weight; transition Top-1 and Top-5 give every next-page transition equal weight. Candidate coverage is how often the next page was proposed at all, not prediction accuracy. Scroll to see all six query types. Expand training details for validation-selected epochs and actual loss histories, or prediction samples to compare the model's proposed pages against the observed next page. Runtime checks and uncertainty are expandable. Downloads include measured CSV/JSON, all query scores, actual raw traces and real logs.

This page shows saved measurements, not a running experiment. Its SQLite queries and physical page layout differ from SeLeP's original PostgreSQL experiment; do not present these scores as SeLeP's scores or combine them with the 8.62 GB study. The 50,000-record subset is non-random, and objects/pages can overlap between parameter-held-out splits. The original live demonstration and interactive terminal remain available as before.

## Explaining query timing

On Project Overview and Research results, the separate selective experiment compares 8.80 ms for the combined policy with 21.51 ms for the earlier neural hybrid and 9.32 ms for no prefetch in the no-delay condition. The combined policy skipped all prefetches. Its 95% interval against no prefetch includes zero, so do not call it a demonstrated prefetch speedup. The context-filtered policy took 10.26 ms and had 15 useful prefetched pages out of 213, but its interval also did not establish a gain over no prefetch. Research results includes the controlled-delay view and full comparison.

The live trained-model, small-demo, SDSS and SQL workspace timing cards each use their own run's real measurements. Point out the displayed no-prefetch baseline and the run size. The small demo reports the median of trial medians; the other cards report means. A lower number from a few interactive SQL repetitions is an observation from that query, not proof of a general speedup.
