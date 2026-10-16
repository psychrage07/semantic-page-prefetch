# Pagewise final summary

Pagewise's final result is a negative runtime finding with a positive offline semantic
signal. Semantic context improves the matched neural predictor, but the simple Markov
baselines are more accurate, and even a future-aware oracle cannot reduce latency under
the tested shared-device conditions.

## Result

The final evaluation was frozen before using seeds 911-915. The source snapshot is in
`FINAL_SOURCE_MANIFEST_V2.json`, and all experimental choices are in
`FINAL_CONFIG_V2.json`.

| Hypothesis | Result |
|---|---|
| H1: Model B beats Model A offline | Supported: +12.4206 points, CI [10.2872, 14.6252] |
| H2: Model B beats simple Markov predictors | Not supported: Markov-2 leads by 17.6943 points |
| H3: Model B reduces total runtime | Not supported by the no-headroom branch |
| H4: Model B beats simpler runtime baselines | Not supported by the no-headroom branch |
| H5: H1 and H3 generalize to held-out templates | Not supported because H3 failed |

On held-out templates, the offline part of H5 did pass: Model B exceeded Model A by
0.6187 points, CI [0.4864, 0.7693]. This did not change H5 because H5 also requires the
runtime claim.

The five-seed runtime confirmation compared `none`, `null`, and `oracle_miss` over 500
paired queries. Oracle prefetching was 59.5928 ms/query slower than no prefetching,
CI [42.7457, 78.3429], and improved only 1.4% of queries. The neural runtime family was
therefore not opened, as required by the preregistered `NO_HEADROOM` branch.

## Reproduction

From `backend/`, use the project virtual environment and the commands encoded in
`FINAL_CONFIG_V2.json`. The final machine-readable inputs are:

- `runs/large-branch-20261015/final-offline/offline_accuracy.json`
- `runs/large-branch-20261015/final-runtime-no-headroom/suite_summary.json`
- `runs/large-branch-20261015/headroom-grid/headroom_grid.json`

Generate the final hypothesis mapping with:

```powershell
.\.venv\Scripts\python.exe scripts\hypothesis_report.py `
  --offline runs\large-branch-20261015\final-offline\offline_accuracy.json `
  --headroom runs\large-branch-20261015\headroom-grid\headroom_grid.json `
  --oracle-suite runs\large-branch-20261015\final-runtime-no-headroom\suite_summary.json `
  --out runs\large-branch-20261015\final-report
```

The final claim is scoped to the tested read-only SQLite workload, cache budget,
latency-injected shared-device model, and Windows host. VFS reads may have been served by
the operating-system cache.
