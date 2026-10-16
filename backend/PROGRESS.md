# Pagewise v2 progress

Last reconciled: 2026-10-15. The experimental roadmap is complete through the final
evaluation and report export. All reported final results come from the current checkout.

## Final status

| Milestone | Status | Evidence |
|---|---|---|
| Reproducible baseline | DONE | Fresh smoke pipeline under `runs/smoke-20261015-101548` |
| Simulator validation | DONE | 8 of 10 validation cells within 15%; the two misses are retained as limitations |
| Headroom decision | DONE | Large-database shared-device grid and runtime confirmation: `NO_HEADROOM` |
| Runtime and cheaper-policy work | DONE | Protocol branch stopped neural runtime tuning after the future-aware oracle lost |
| Final evaluation | DONE | Frozen config/source plus seeds 911-915 for offline and runtime studies |
| Report and dashboard | DONE | Mechanical H1-H5 report, consolidated report, and measured dashboard snapshot exported |
| Final verification | DONE | 63 backend tests pass, frontend production build passes, JSON and source manifest validate |

## Frozen experiment

- Base revision: `5b884f374f52bfc865391114987bbba7dc1a0ca1` plus the changes identified by
  `FINAL_SOURCE_MANIFEST_V2.json` (`48` files; aggregate SHA-256
  `95052c3c8e594aab080908f230f451df67955621edae01b6806a8577b79075bf`).
- Configuration: `FINAL_CONFIG_V2.json`.
- Database: 8,615,698,432 bytes, 2,103,442 pages, 2.32 million orders, and
  8,030,380 order items.
- Trace: 300 queries, 58,466 page requests, and a sparse 15,620-page metadata map.
- Selected development model: hidden size 128, four epochs, 32 candidates, top-k 8.
  Model B mean top-1 was 76.2555%, compared with 76.1565% for hidden size 64.

## Final results

Offline results use seeds 911-915 and paired bootstrap intervals over held-out query
executions.

| Hypothesis | Outcome | Main evidence |
|---|---|---|
| H1 | Supported | Model B minus Model A: +12.4206 points, 95% CI [10.2872, 14.6252] |
| H2 | Not supported | Model B minus Markov-2: -17.6943 points, 95% CI [-20.8634, -14.6458] |
| H3 | Not supported | Oracle minus none: +59.5928 ms/query, 95% CI [42.7457, 78.3429] |
| H4 | Not supported | No neural runtime comparison was opened after the oracle failed the headroom gate |
| H5 | Not supported | Offline template H1 passed, but the required runtime H3 did not |

The final runtime confirmation contains five matched seeds and 500 paired queries per
comparison. Median total query time was 11,229.8127 ms for `none`, 12,020.6914 ms for
`null`, and 16,650.5128 ms for `oracle_miss`. The future-aware oracle improved 1.4% of
queries. These are latency-injected shared-device results, not physical-device claims.

## Artifacts

- Offline: `runs/large-branch-20261015/final-offline/offline_accuracy.json`
- Runtime: `runs/large-branch-20261015/final-runtime-no-headroom/suite_summary.json`
- Hypotheses: `runs/large-branch-20261015/final-report/final_hypotheses.json`
- Human-readable report: `runs/large-branch-20261015/final-report/research_report.md`
- Dashboard snapshot: `../public/pagewise-dashboard.json`

## Boundaries

- VFS reads can be served by the operating-system cache and are not proof of physical I/O.
- Injected delays are measured host timer delays, not NVMe, SATA, or HDD measurements.
- The oracle is a future-aware experimental control, not a universal mathematical upper
  bound for every possible prefetch implementation.
- Simulator validation missed the 15% target for 2 of 10 declared cells.
