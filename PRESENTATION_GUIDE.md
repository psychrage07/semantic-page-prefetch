# Pagewise faculty walkthrough

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

The default **Test trained hybrid** mode runs the actual selected model. Choose 20 queries per policy/seed and click **Run trained hybrid**. It verifies the existing frozen checkpoint, predictor source and matching 8.62 GB database, then executes the same fresh workload for no prefetch, Markov-2 and the selected neural hybrid. Two new workload seeds give 120 query executions. The eight stages show verification, six actual policy runs and the result export. No weights are trained or changed.

Results show fresh mean/median query times, scored prediction jobs, prefetched/useful page counts and actual candidate samples. This is live inference/runtime evidence, not a new held-out accuracy claim. Parameters may overlap training; diagnostics are enabled and the OS cache is not reset. The saved 47.74% score stays in Research results.

The separate **Build a small demo** mode creates its own database, trains Model A and Model B and evaluates the standard policies. Choose 100 traced queries and click **Start live demo** for that seven-stage training pipeline. Its standard hybrid is separate from the selected neural hybrid. The input changes the tracing stage; runtime uses 30 queries per policy/seed, eight policies and two seeds.

A previously completed experiment is labelled **Previous local run**, with its timestamp. Do not assume the neural model wins. **Download results** saves the measured export. Local frozen research artifacts are required for the trained-model test; the small demo remains available on a fresh clone.

## 4. Research results

Use **New query values** and **New query types** to explain the generalization boundary. Unfamiliar query types remain difficult; the selected template-split policy is a Markov fallback without a learned neural increment.

Show the runtime table: no measured speedup was established. The requested delays are artificial and the channel counts are abstract service slots. These are not physical disk benchmarks.

Use **Full research evidence** only for questions about Top-5, candidate recall, statistical uncertainty or detailed runtime counters. Individual commands remain under **Advanced research tools** in the live demo. Historical results and fixture previews are under **Research workspace** in the sidebar.
