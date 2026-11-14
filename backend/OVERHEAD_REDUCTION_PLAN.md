# Pagewise: turning better predictions into useful prefetching

The goal is to retain prediction quality while reducing the cost of making and using predictions. A better prediction score does not by itself establish faster query execution.

## What the held-out study measured

The separate selective-prefetch study used the synthetic 8.62 GB database and five held-out workload seeds, with 150 queries per policy and condition. On the common reference trace, the selected neural hybrid achieved 45.98% query-average Top-1 versus 43.21% for Markov-2; transition-level candidate coverage was 79.77% versus 70.64%. These metrics use different weighting. See `SELECTIVE_PREFETCH_RESULTS.md` for the complete methods, query-family breakdowns and uncertainty.

| Policy | No added delay, mean query time | 0.2 ms requested artificial delay, mean query time |
|---|---:|---:|
| No prefetch | 9.321 ms | 94.755 ms |
| Markov-2 | 15.929 ms | 114.328 ms |
| Selected neural hybrid | 21.510 ms | 112.198 ms |
| Context-filtered experiment | 10.259 ms | 96.960 ms |

The neural hybrid was slower than no prefetch in both measured conditions. In the artificial-delay condition its observed mean was lower than Markov-2; that observation alone does not establish a reliable speedup. Deadline, adaptive and combined each prefetched zero pages and do not demonstrate successful prefetching. Neural fallback was disabled in the context experiment.

No added delay means no *artificial* service delay, not zero execution or I/O time. The delayed condition uses an artificial shared single-channel service, not a benchmark of a physical slow disk. OS cache was not reset.

## Work sequence

1. **Profile the existing policy.** On development seeds, measure CPU time, candidate generation, feature construction, neural inference, queue wait, reads, useful pages and demand latency. Break these down by query family and whether the current Markov-2 context is known. Preserve the frozen checkpoint and baseline results.
2. **Reduce prediction cost.** Evaluate bounded candidate counts, less frequent prediction, reusable features, efficient scoring and using neural ranking only on selected uncertain contexts. Implement each change as an isolated opt-in policy, keeping the unchanged neural hybrid, Markov-2 and no-prefetch controls. Verify SQL results and prediction behavior before comparing performance.
3. **Check whether prefetches arrive in time.** Measure useful pages, unused pages, in-flight misses and lead time. Reject or revise settings that merely skip all requests; label any zero-prefetch policy as a control, not a successful prefetcher. Recalibrate thresholds using development data only.
4. **Compare on development workloads.** Run the same queries and seeds across policies, vary execution order, and record both no-delay and artificial-delay conditions. Report query-average and transition-average accuracy separately. Select configurations using only development results, not held-out results.
5. **Freeze and evaluate.** Record the chosen configuration and checkpoint hashes before a fresh held-out test. Compare no prefetch, Markov-2, the original neural hybrid and the new policy on matched queries. Report query latency, uncertainty, CPU, pages prefetched, useful pages and accuracy, including unfavorable results. Preserve all raw files and identify OS cache and device conditions.

Only describe a policy as faster when the matched latency evidence supports that specific comparison under the tested conditions. A smaller point estimate, a high candidate-recall score, or skipping all speculative reads is not sufficient.
