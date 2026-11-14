# Selective Pagewise prefetching: measured results

This is a new study on the existing synthetic 8.62 GB SQLite snapshot with fresh workload seeds. It is separate from the original frozen study. The earlier queue-optimization branch, H1 audit and related artifacts were unavailable, so the figures quoted in the request were not verified or reused as evidence.

## Protocol

Development seeds: [1101, 1102]; held-out seeds: [4401, 4402, 4403, 4404, 4405]; 30 queries per seed/policy/scenario. All policies use matched query parameters, 512 Pagewise cache pages, a 4096 KiB SQLite pager, one worker, diagnostic logging and randomized policy order. SQLite's pager is released between queries; the OS cache is not reset.

Two conditions are shown: zero injected delay and a shared, single-channel model with 0.2 ms requested service delay. The delay is artificial and operating-system scheduling may make actual waits longer. It is not a physical-device benchmark. Demand priority is best-effort: a speculative operation already in progress and actual OS I/O cannot be preempted.

The selected frozen checkpoint SHA256 is `23bafd4cce8cc40ddc5dacef7fb4990972cfba985b9d5bf779082b7dd31bf973`; the policy was frozen as `cc3ebc9665a1161fa9e3302af9b10b6527277779118bdfb8e9ea247d37250eb1` before held-out execution. The original predictor source and saved research data remain untouched. New policies are an opt-in extension outside the hashed frozen predictor directories.

The no-prefetch, native Markov-2 and existing neural–Markov ranker are baselines. The context policy uses Markov-2 on known contexts and an evidence-gated fallback. Deadline and adaptive are individual additions to that context policy; combined enables both. This design keeps exact Markov-2 behavior or skips prefetching when cost/deadline rules reject the job.

## Development calibration

Exact Markov-2 contexts occurred in 84.58% of 16,539 development transitions. This is a new result, not a reconstruction of the missing H1 audit. Neural fallback was enabled for: no families. No confidence threshold was added because candidate scores were not calibrated against usefulness.

A family required at least 40 fallback transitions, 10 queries, both development seeds, and a positive paired 95% query-bootstrap lower bound against Markov-1. Recursive traversal remains disabled by default. These two-seed intervals are exploratory; indexed lookup remains especially uncertain.

| Family | Fallback transitions | Queries | Neural − Markov-1 macro Top-1 (pp) | 95% interval | Neural enabled? |
|---|---:|---:|---:|---:|---|
| indexed_join | 674 | 24 | 1.17 | [0.00, 2.65] | no |
| indexed_lookup | 3 | 3 | 0.00 | [0.00, 0.00] | no |
| point_lookup | 35 | 14 | 0.00 | [0.00, 0.00] | no |
| range_scan | 1581 | 13 | 0.02 | [-0.34, 0.37] | no |
| recursive_traversal | 257 | 6 | 0.36 | [0.00, 1.63] | no |

Deadlines are the median observed interval between no-prefetch VFS reads on zero-delay development traces. This is a conservative estimate of time until the next read, not a guaranteed future demand deadline. The cost controller uses development-estimated prediction accuracy, usefulness, miss cost, inference time, queue wait, read time and current queue/cache pressure; Markov-1 cost uses the Markov-2 measurement as a proxy. Unavailable estimates cause skipping. It does not use future accesses.

An early development attempt was retained but excluded because instrumentation changed while it was running. All reported development ablations were rerun with the same fixed code before the policy freeze. No held-out outcome was used to tune the policy.

## Held-out runtime

Every row contains 150 matched query executions. Delta is policy minus no-prefetch mean query latency; negative favors the policy. The 95% paired interval resamples workload seeds and queries within seeds. Intervals are descriptive with only five workload seeds; this is not a claim of physical I/O improvement.

### zero

| Policy | Mean ms | p95 ms | Delta vs none ms | 95% CI ms | Peak RSS max MB | CPU total s | Prefetched | Useful | Wasted |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| none | 9.321 | 30.476 | baseline | n/a | 303.3 | 3.92 | 0 | 0 | 0 |
| markov2 | 15.929 | 31.699 | 6.608 | [1.012, 13.824] | 303.1 | 6.30 | 131 | 2 | 129 |
| model_ranker | 21.510 | 39.754 | 12.189 | [4.307, 22.908] | 382.5 | 29.50 | 775 | 0 | 775 |
| context | 10.259 | 14.060 | 0.939 | [-2.005, 2.734] | 383.1 | 13.33 | 213 | 15 | 198 |
| deadline | 8.422 | 10.207 | -0.899 | [-3.966, 0.956] | 382.9 | 12.56 | 0 | 0 | 0 |
| adaptive | 11.292 | 41.114 | 1.971 | [-2.985, 6.817] | 382.8 | 12.88 | 0 | 0 | 0 |
| combined | 8.804 | 9.883 | -0.517 | [-3.692, 1.853] | 382.8 | 12.92 | 0 | 0 | 0 |

Peak RSS is the largest per-process peak working set across five runs. CPU is total process CPU across those five runs, excluding dependency imports. Wasted means unused evicted plus unused resident prefetched insertions at run end.

| New policy | Verdict vs none | Seeds with lower mean | Delta vs Markov-2 ms (95% CI) | Delta vs existing neural hybrid ms (95% CI) |
|---|---|---:|---|---|
| context | interval includes zero; no supported change | 1/5 | -5.670 [-12.225, -0.806] | -11.250 [-21.317, -3.841] |
| deadline | interval includes zero; no supported change | 1/5 | -7.508 [-14.226, -2.510] | -13.088 [-23.181, -5.583] |
| adaptive | interval includes zero; no supported change | 1/5 | -4.637 [-13.392, 3.345] | -10.218 [-22.092, -0.067] |
| combined | interval includes zero; no supported change | 1/5 | -7.125 [-13.459, -2.392] | -12.706 [-22.407, -5.484] |

| Added component | Delta vs context ms | 95% CI ms | Interpretation |
|---|---:|---|---|
| deadline | -1.838 | [-2.192, -1.486] | lower latency supported by this interval |
| adaptive | 1.033 | [-2.069, 5.052] | interval includes zero; no supported change |
| combined | -1.456 | [-1.953, -0.653] | lower latency supported by this interval |

Completion-before-demand is observable as useful prefetched insertions: useful / successfully inserted pages. Unused pages remain in the denominator. In-flight demand misses are separately counted; this is not an exact timing rate over every candidate.

| New policy | Queue wait ms/job | Inference ms/job | Timely useful / inserted | Demand misses while in flight | Deadline drops | Query-ended drops | Queue-full drops | Coalesced jobs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| context | 3.2089 | 0.0032 | 7.04% | 6 | 0 | 256 | 0 | 0 |
| deadline | n/a | n/a | n/a | 0 | 668 | 0 | 0 | 0 |
| adaptive | n/a | n/a | n/a | 0 | 0 | 0 | 0 | 0 |
| combined | n/a | n/a | n/a | 0 | 0 | 0 | 0 | 0 |

### Query-family runtime: combined policy

| Family | Queries | Mean ms | p95 ms | Delta vs none ms | 95% CI ms | Seed clusters |
|---|---:|---:|---:|---:|---|---:|
| indexed_join | 68 | 3.862 | 7.760 | -1.578 | [-6.376, 0.469] | 5 |
| indexed_lookup | 12 | 68.878 | 249.931 | 6.206 | [0.330, 17.403] | 5 |
| point_lookup | 36 | 1.434 | 2.607 | -0.098 | [-0.898, 0.308] | 5 |
| range_scan | 19 | 6.801 | 10.388 | -2.146 | [-9.694, 1.362] | 5 |
| recursive_traversal | 15 | 3.372 | 6.821 | -0.030 | [-1.504, 1.096] | 5 |

### Offline prediction accuracy

Scored on all next-page transitions from the shared no-prefetch reference. Selective decisions use the frozen development policy with zero queue/cache pressure; deadlines use estimated remaining work, and skips count as errors. This is an idealized offline approximation, not the behavior of every runtime job. The existing neural baseline is scored without its runtime confidence/stride filter. Candidate recall is target inclusion in the available full candidate set.

| Policy | Queries | Transitions | Query-average Top-1 | Transition Top-1 | Transition Top-5 | Candidate recall | Skips |
|---|---:|---:|---:|---:|---:|---:|---:|
| markov2 | 150 | 24859 | 43.21% | 68.11% | 68.94% | 70.64% | 0 |
| model_ranker | 150 | 24859 | 45.98% | 68.59% | 69.18% | 79.77% | 0 |
| context | 150 | 24859 | 43.21% | 68.11% | 68.94% | 70.64% | 3359 |
| deadline | 150 | 24859 | 0.00% | 0.00% | 0.00% | 0.00% | 24859 |
| adaptive | 150 | 24859 | 0.00% | 0.00% | 0.00% | 0.00% | 24859 |
| combined | 150 | 24859 | 0.00% | 0.00% | 0.00% | 0.00% | 24859 |

### Query-family prediction accuracy: all policies

Intervals below resample query accuracies within workload seeds and then resample seeds. Family intervals and paired family comparisons were added in a second analysis pass; the frozen policy and raw outcomes were unchanged.

| Policy | Family | Queries | Transitions | Query-average Top-1 | 95% CI | Transition Top-1 | Transition Top-5 | Candidate recall |
|---|---|---:|---:|---:|---|---:|---:|---:|
| markov2 | indexed_join | 68 | 6001 | 49.95% | [35.926, 66.937] | 46.31% | 47.58% | 49.83% |
| markov2 | indexed_lookup | 12 | 13935 | 41.24% | [18.380, 66.002] | 94.68% | 94.83% | 94.88% |
| markov2 | point_lookup | 36 | 1096 | 69.64% | [49.528, 85.690] | 83.12% | 87.32% | 90.15% |
| markov2 | range_scan | 19 | 2913 | 1.06% | [0.814, 1.315] | 0.93% | 2.16% | 9.20% |
| markov2 | recursive_traversal | 15 | 914 | 4.15% | [1.771, 7.034] | 2.30% | 5.36% | 10.07% |
| model_ranker | indexed_join | 68 | 6001 | 51.13% | [36.977, 68.100] | 46.98% | 48.56% | 61.49% |
| model_ranker | indexed_lookup | 12 | 13935 | 47.24% | [25.954, 70.187] | 94.73% | 94.85% | 99.93% |
| model_ranker | point_lookup | 36 | 1096 | 74.61% | [55.941, 89.755] | 86.31% | 87.32% | 95.99% |
| model_ranker | range_scan | 19 | 2913 | 1.77% | [1.292, 2.329] | 1.48% | 2.06% | 32.89% |
| model_ranker | recursive_traversal | 15 | 914 | 8.84% | [3.843, 15.270] | 4.60% | 5.47% | 22.54% |
| context | indexed_join | 68 | 6001 | 49.95% | [35.926, 66.937] | 46.31% | 47.58% | 49.83% |
| context | indexed_lookup | 12 | 13935 | 41.24% | [18.380, 66.002] | 94.68% | 94.83% | 94.88% |
| context | point_lookup | 36 | 1096 | 69.64% | [49.528, 85.690] | 83.12% | 87.32% | 90.15% |
| context | range_scan | 19 | 2913 | 1.06% | [0.814, 1.315] | 0.93% | 2.16% | 9.20% |
| context | recursive_traversal | 15 | 914 | 4.15% | [1.771, 7.034] | 2.30% | 5.36% | 10.07% |
| deadline | indexed_join | 68 | 6001 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| deadline | indexed_lookup | 12 | 13935 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| deadline | point_lookup | 36 | 1096 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| deadline | range_scan | 19 | 2913 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| deadline | recursive_traversal | 15 | 914 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| adaptive | indexed_join | 68 | 6001 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| adaptive | indexed_lookup | 12 | 13935 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| adaptive | point_lookup | 36 | 1096 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| adaptive | range_scan | 19 | 2913 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| adaptive | recursive_traversal | 15 | 914 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| combined | indexed_join | 68 | 6001 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| combined | indexed_lookup | 12 | 13935 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| combined | point_lookup | 36 | 1096 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| combined | range_scan | 19 | 2913 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| combined | recursive_traversal | 15 | 914 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |

### Query-family latency: all policies

| Policy | Family | Queries | Mean ms | p95 ms | Delta vs none ms | 95% CI ms | Seed clusters |
|---|---|---:|---:|---:|---:|---|---:|
| none | indexed_join | 68 | 5.440 | 28.332 | baseline | n/a | 5 |
| none | indexed_lookup | 12 | 62.672 | 222.401 | baseline | n/a | 5 |
| none | point_lookup | 36 | 1.532 | 2.751 | baseline | n/a | 5 |
| none | range_scan | 19 | 8.947 | 30.976 | baseline | n/a | 5 |
| none | recursive_traversal | 15 | 3.402 | 7.160 | baseline | n/a | 5 |
| markov2 | indexed_join | 68 | 6.828 | 13.827 | 1.389 | [-3.850, 5.061] | 5 |
| markov2 | indexed_lookup | 12 | 122.156 | 440.124 | 59.484 | [8.840, 116.789] | 5 |
| markov2 | point_lookup | 36 | 2.200 | 3.813 | 0.668 | [-0.189, 1.288] | 5 |
| markov2 | range_scan | 19 | 14.425 | 31.386 | 5.478 | [-4.651, 13.600] | 5 |
| markov2 | recursive_traversal | 15 | 7.061 | 17.941 | 3.659 | [0.071, 10.088] | 5 |
| model_ranker | indexed_join | 68 | 11.298 | 27.790 | 5.858 | [0.274, 12.388] | 5 |
| model_ranker | indexed_lookup | 12 | 150.910 | 540.187 | 88.238 | [14.187, 172.903] | 5 |
| model_ranker | point_lookup | 36 | 2.809 | 6.387 | 1.277 | [0.337, 2.070] | 5 |
| model_ranker | range_scan | 19 | 20.940 | 43.581 | 11.992 | [-1.102, 27.515] | 5 |
| model_ranker | recursive_traversal | 15 | 9.887 | 25.269 | 6.485 | [0.814, 13.638] | 5 |
| context | indexed_join | 68 | 5.854 | 12.983 | 0.414 | [-3.982, 2.662] | 5 |
| context | indexed_lookup | 12 | 66.718 | 235.479 | 4.047 | [0.760, 8.627] | 5 |
| context | point_lookup | 36 | 2.244 | 4.854 | 0.712 | [-0.138, 1.339] | 5 |
| context | range_scan | 19 | 9.726 | 13.207 | 0.779 | [-6.271, 4.037] | 5 |
| context | recursive_traversal | 15 | 4.978 | 9.452 | 1.576 | [-0.199, 3.213] | 5 |
| deadline | indexed_join | 68 | 3.731 | 8.095 | -1.708 | [-6.598, 0.434] | 5 |
| deadline | indexed_lookup | 12 | 66.335 | 236.553 | 3.663 | [-0.121, 9.202] | 5 |
| deadline | point_lookup | 36 | 1.264 | 2.340 | -0.268 | [-1.065, 0.105] | 5 |
| deadline | range_scan | 19 | 6.482 | 10.394 | -2.465 | [-9.933, 0.670] | 5 |
| deadline | recursive_traversal | 15 | 2.988 | 6.523 | -0.414 | [-1.737, 0.441] | 5 |
| adaptive | indexed_join | 68 | 6.892 | 22.627 | 1.453 | [-5.771, 7.290] | 5 |
| adaptive | indexed_lookup | 12 | 65.018 | 231.710 | 2.346 | [-1.095, 7.233] | 5 |
| adaptive | point_lookup | 36 | 1.495 | 2.641 | -0.037 | [-0.897, 0.434] | 5 |
| adaptive | range_scan | 19 | 15.555 | 49.168 | 6.608 | [-6.720, 20.829] | 5 |
| adaptive | recursive_traversal | 15 | 6.369 | 18.075 | 2.967 | [-0.998, 10.098] | 5 |
| combined | indexed_join | 68 | 3.862 | 7.760 | -1.578 | [-6.376, 0.469] | 5 |
| combined | indexed_lookup | 12 | 68.878 | 249.931 | 6.206 | [0.330, 17.403] | 5 |
| combined | point_lookup | 36 | 1.434 | 2.607 | -0.098 | [-0.898, 0.308] | 5 |
| combined | range_scan | 19 | 6.801 | 10.388 | -2.146 | [-9.694, 1.362] | 5 |
| combined | recursive_traversal | 15 | 3.372 | 6.821 | -0.030 | [-1.504, 1.096] | 5 |

### Actual scored runtime jobs

| Policy | Aligned scored jobs | Queries | Query-average Top-1 | Transition Top-1 | Transition Top-5 | Ambiguous legacy histories excluded |
|---|---:|---:|---:|---:|---:|---:|
| none | 0 | 0 | n/a | n/a | n/a | 0 |
| markov2 | 3400 | 44 | 36.42 | 96.24 | 96.38 | 0 |
| model_ranker | 1192 | 47 | 33.45 | 78.19 | 78.44 | 0 |
| context | 422 | 106 | 39.62 | 42.65 | 46.21 | 0 |
| deadline | 0 | 0 | n/a | n/a | n/a | 0 |
| adaptive | 0 | 0 | n/a | n/a | n/a | 0 |
| combined | 0 | 0 | n/a | n/a | n/a | 0 |

These are conditional on actual recorded scoring jobs that can be aligned to the same policy's observed demand stream; different policies score different jobs. Legacy job histories that match multiple positions are excluded. Logs truncate candidate lists, so their target-in-logged-candidates measure is not full candidate recall. Full counters, model actions and per-family values are in the JSON evidence.

### controlled-delay

| Policy | Mean ms | p95 ms | Delta vs none ms | 95% CI ms | Peak RSS max MB | CPU total s | Prefetched | Useful | Wasted |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| none | 94.755 | 122.698 | baseline | n/a | 302.9 | 5.48 | 0 | 0 | 0 |
| markov2 | 114.328 | 139.156 | 19.573 | [7.000, 38.589] | 303.0 | 8.33 | 4370 | 136 | 4234 |
| model_ranker | 112.198 | 149.201 | 17.444 | [6.592, 33.393] | 382.9 | 34.42 | 266 | 21 | 245 |
| context | 96.960 | 126.025 | 2.205 | [1.603, 2.957] | 383.6 | 14.73 | 235 | 54 | 181 |
| deadline | 94.950 | 122.378 | 0.195 | [-0.243, 0.846] | 382.4 | 13.83 | 0 | 0 | 0 |
| adaptive | 95.454 | 122.623 | 0.699 | [0.123, 1.479] | 382.6 | 14.11 | 0 | 0 | 0 |
| combined | 95.496 | 122.024 | 0.741 | [0.173, 1.464] | 383.4 | 14.20 | 0 | 0 | 0 |

Peak RSS is the largest per-process peak working set across five runs. CPU is total process CPU across those five runs, excluding dependency imports. Wasted means unused evicted plus unused resident prefetched insertions at run end.

| New policy | Verdict vs none | Seeds with lower mean | Delta vs Markov-2 ms (95% CI) | Delta vs existing neural hybrid ms (95% CI) |
|---|---|---:|---|---|
| context | higher latency supported by this interval | 0/5 | -17.368 [-35.681, -5.257] | -15.239 [-30.676, -4.882] |
| deadline | interval includes zero; no supported change | 3/5 | -19.378 [-37.994, -6.974] | -17.249 [-32.796, -6.601] |
| adaptive | higher latency supported by this interval | 1/5 | -18.874 [-37.237, -6.617] | -16.745 [-32.185, -6.273] |
| combined | higher latency supported by this interval | 0/5 | -18.832 [-37.199, -6.511] | -16.702 [-32.345, -6.155] |

| Added component | Delta vs context ms | 95% CI ms | Interpretation |
|---|---:|---|---|
| deadline | -2.010 | [-2.633, -1.563] | lower latency supported by this interval |
| adaptive | -1.506 | [-1.956, -1.084] | lower latency supported by this interval |
| combined | -1.464 | [-1.977, -0.974] | lower latency supported by this interval |

Completion-before-demand is observable as useful prefetched insertions: useful / successfully inserted pages. Unused pages remain in the denominator. In-flight demand misses are separately counted; this is not an exact timing rate over every candidate.

| New policy | Queue wait ms/job | Inference ms/job | Timely useful / inserted | Demand misses while in flight | Deadline drops | Query-ended drops | Queue-full drops | Coalesced jobs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| context | 0.0931 | 0.0055 | 22.98% | 138 | 0 | 7 | 0 | 0 |
| deadline | n/a | n/a | n/a | 0 | 668 | 0 | 0 | 0 |
| adaptive | n/a | n/a | n/a | 0 | 0 | 0 | 0 | 0 |
| combined | n/a | n/a | n/a | 0 | 0 | 0 | 0 | 0 |

### Query-family runtime: combined policy

| Family | Queries | Mean ms | p95 ms | Delta vs none ms | 95% CI ms | Seed clusters |
|---|---:|---:|---:|---:|---|---:|
| indexed_join | 68 | 44.535 | 109.235 | 0.580 | [0.109, 1.138] | 5 |
| indexed_lookup | 12 | 734.738 | 2594.015 | 3.422 | [-1.282, 9.874] | 5 |
| point_lookup | 36 | 10.782 | 29.726 | -0.081 | [-0.822, 0.302] | 5 |
| range_scan | 19 | 81.751 | 125.701 | 0.989 | [-0.056, 1.956] | 5 |
| recursive_traversal | 15 | 35.847 | 78.991 | 0.989 | [-0.174, 3.242] | 5 |

### Offline prediction accuracy

Scored on all next-page transitions from the shared no-prefetch reference. Selective decisions use the frozen development policy with zero queue/cache pressure; deadlines use estimated remaining work, and skips count as errors. This is an idealized offline approximation, not the behavior of every runtime job. The existing neural baseline is scored without its runtime confidence/stride filter. Candidate recall is target inclusion in the available full candidate set.

| Policy | Queries | Transitions | Query-average Top-1 | Transition Top-1 | Transition Top-5 | Candidate recall | Skips |
|---|---:|---:|---:|---:|---:|---:|---:|
| markov2 | 150 | 24859 | 43.21% | 68.11% | 68.94% | 70.64% | 0 |
| model_ranker | 150 | 24859 | 45.98% | 68.59% | 69.18% | 79.77% | 0 |
| context | 150 | 24859 | 43.21% | 68.11% | 68.94% | 70.64% | 3359 |
| deadline | 150 | 24859 | 0.00% | 0.00% | 0.00% | 0.00% | 24859 |
| adaptive | 150 | 24859 | 0.00% | 0.00% | 0.00% | 0.00% | 24859 |
| combined | 150 | 24859 | 0.00% | 0.00% | 0.00% | 0.00% | 24859 |

### Query-family prediction accuracy: all policies

Intervals below resample query accuracies within workload seeds and then resample seeds. Family intervals and paired family comparisons were added in a second analysis pass; the frozen policy and raw outcomes were unchanged.

| Policy | Family | Queries | Transitions | Query-average Top-1 | 95% CI | Transition Top-1 | Transition Top-5 | Candidate recall |
|---|---|---:|---:|---:|---|---:|---:|---:|
| markov2 | indexed_join | 68 | 6001 | 49.95% | [35.926, 66.937] | 46.31% | 47.58% | 49.83% |
| markov2 | indexed_lookup | 12 | 13935 | 41.24% | [18.380, 66.002] | 94.68% | 94.83% | 94.88% |
| markov2 | point_lookup | 36 | 1096 | 69.64% | [49.528, 85.690] | 83.12% | 87.32% | 90.15% |
| markov2 | range_scan | 19 | 2913 | 1.06% | [0.814, 1.315] | 0.93% | 2.16% | 9.20% |
| markov2 | recursive_traversal | 15 | 914 | 4.15% | [1.771, 7.034] | 2.30% | 5.36% | 10.07% |
| model_ranker | indexed_join | 68 | 6001 | 51.13% | [36.977, 68.100] | 46.98% | 48.56% | 61.49% |
| model_ranker | indexed_lookup | 12 | 13935 | 47.24% | [25.954, 70.187] | 94.73% | 94.85% | 99.93% |
| model_ranker | point_lookup | 36 | 1096 | 74.61% | [55.941, 89.755] | 86.31% | 87.32% | 95.99% |
| model_ranker | range_scan | 19 | 2913 | 1.77% | [1.292, 2.329] | 1.48% | 2.06% | 32.89% |
| model_ranker | recursive_traversal | 15 | 914 | 8.84% | [3.843, 15.270] | 4.60% | 5.47% | 22.54% |
| context | indexed_join | 68 | 6001 | 49.95% | [35.926, 66.937] | 46.31% | 47.58% | 49.83% |
| context | indexed_lookup | 12 | 13935 | 41.24% | [18.380, 66.002] | 94.68% | 94.83% | 94.88% |
| context | point_lookup | 36 | 1096 | 69.64% | [49.528, 85.690] | 83.12% | 87.32% | 90.15% |
| context | range_scan | 19 | 2913 | 1.06% | [0.814, 1.315] | 0.93% | 2.16% | 9.20% |
| context | recursive_traversal | 15 | 914 | 4.15% | [1.771, 7.034] | 2.30% | 5.36% | 10.07% |
| deadline | indexed_join | 68 | 6001 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| deadline | indexed_lookup | 12 | 13935 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| deadline | point_lookup | 36 | 1096 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| deadline | range_scan | 19 | 2913 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| deadline | recursive_traversal | 15 | 914 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| adaptive | indexed_join | 68 | 6001 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| adaptive | indexed_lookup | 12 | 13935 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| adaptive | point_lookup | 36 | 1096 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| adaptive | range_scan | 19 | 2913 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| adaptive | recursive_traversal | 15 | 914 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| combined | indexed_join | 68 | 6001 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| combined | indexed_lookup | 12 | 13935 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| combined | point_lookup | 36 | 1096 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| combined | range_scan | 19 | 2913 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |
| combined | recursive_traversal | 15 | 914 | 0.00% | [0.000, 0.000] | 0.00% | 0.00% | 0.00% |

### Query-family latency: all policies

| Policy | Family | Queries | Mean ms | p95 ms | Delta vs none ms | 95% CI ms | Seed clusters |
|---|---|---:|---:|---:|---:|---|---:|
| none | indexed_join | 68 | 43.956 | 106.358 | baseline | n/a | 5 |
| none | indexed_lookup | 12 | 731.316 | 2583.494 | baseline | n/a | 5 |
| none | point_lookup | 36 | 10.863 | 29.351 | baseline | n/a | 5 |
| none | range_scan | 19 | 80.762 | 125.821 | baseline | n/a | 5 |
| none | recursive_traversal | 15 | 34.858 | 80.045 | baseline | n/a | 5 |
| markov2 | indexed_join | 68 | 51.988 | 128.141 | 8.032 | [6.071, 10.410] | 5 |
| markov2 | indexed_lookup | 12 | 897.154 | 3194.430 | 165.839 | [17.820, 331.234] | 5 |
| markov2 | point_lookup | 36 | 12.680 | 36.701 | 1.817 | [0.502, 3.689] | 5 |
| markov2 | range_scan | 19 | 93.849 | 144.170 | 13.087 | [10.767, 15.914] | 5 |
| markov2 | recursive_traversal | 15 | 40.569 | 91.136 | 5.711 | [3.514, 8.033] | 5 |
| model_ranker | indexed_join | 68 | 50.747 | 130.415 | 6.791 | [5.093, 8.749] | 5 |
| model_ranker | indexed_lookup | 12 | 873.923 | 3100.461 | 142.608 | [19.467, 282.985] | 5 |
| model_ranker | point_lookup | 36 | 12.374 | 35.226 | 1.511 | [0.530, 2.514] | 5 |
| model_ranker | range_scan | 19 | 96.136 | 155.567 | 15.374 | [11.950, 19.898] | 5 |
| model_ranker | recursive_traversal | 15 | 41.321 | 93.490 | 6.463 | [3.976, 9.496] | 5 |
| context | indexed_join | 68 | 46.084 | 112.673 | 2.128 | [1.579, 2.840] | 5 |
| context | indexed_lookup | 12 | 736.457 | 2594.821 | 5.141 | [0.491, 11.155] | 5 |
| context | point_lookup | 36 | 11.850 | 34.131 | 0.988 | [0.080, 2.029] | 5 |
| context | range_scan | 19 | 83.814 | 128.466 | 3.052 | [2.095, 4.075] | 5 |
| context | recursive_traversal | 15 | 36.914 | 80.767 | 2.056 | [1.341, 3.086] | 5 |
| deadline | indexed_join | 68 | 44.067 | 106.847 | 0.112 | [-0.165, 0.365] | 5 |
| deadline | indexed_lookup | 12 | 732.701 | 2587.507 | 1.386 | [-4.029, 8.235] | 5 |
| deadline | point_lookup | 36 | 10.668 | 29.580 | -0.195 | [-0.918, 0.172] | 5 |
| deadline | range_scan | 19 | 81.365 | 125.919 | 0.603 | [-0.207, 1.539] | 5 |
| deadline | recursive_traversal | 15 | 34.899 | 79.041 | 0.040 | [-0.392, 0.444] | 5 |
| adaptive | indexed_join | 68 | 44.279 | 105.649 | 0.323 | [0.009, 0.648] | 5 |
| adaptive | indexed_lookup | 12 | 734.603 | 2591.074 | 3.287 | [-0.212, 10.358] | 5 |
| adaptive | point_lookup | 36 | 10.911 | 30.430 | 0.048 | [-0.791, 0.555] | 5 |
| adaptive | range_scan | 19 | 82.174 | 124.944 | 1.412 | [0.239, 2.465] | 5 |
| adaptive | recursive_traversal | 15 | 35.851 | 79.420 | 0.993 | [0.221, 2.560] | 5 |
| combined | indexed_join | 68 | 44.535 | 109.235 | 0.580 | [0.109, 1.138] | 5 |
| combined | indexed_lookup | 12 | 734.738 | 2594.015 | 3.422 | [-1.282, 9.874] | 5 |
| combined | point_lookup | 36 | 10.782 | 29.726 | -0.081 | [-0.822, 0.302] | 5 |
| combined | range_scan | 19 | 81.751 | 125.701 | 0.989 | [-0.056, 1.956] | 5 |
| combined | recursive_traversal | 15 | 35.847 | 78.991 | 0.989 | [-0.174, 3.242] | 5 |

### Actual scored runtime jobs

| Policy | Aligned scored jobs | Queries | Query-average Top-1 | Transition Top-1 | Transition Top-5 | Ambiguous legacy histories excluded |
|---|---:|---:|---:|---:|---:|---:|
| none | 0 | 0 | n/a | n/a | n/a | 0 |
| markov2 | 4795 | 130 | 39.06 | 80.19 | 80.60 | 0 |
| model_ranker | 5622 | 131 | 36.86 | 68.48 | 69.12 | 0 |
| context | 659 | 126 | 40.05 | 48.56 | 51.29 | 0 |
| deadline | 0 | 0 | n/a | n/a | n/a | 0 |
| adaptive | 0 | 0 | n/a | n/a | n/a | 0 |
| combined | 0 | 0 | n/a | n/a | n/a | 0 |

These are conditional on actual recorded scoring jobs that can be aligned to the same policy's observed demand stream; different policies score different jobs. Legacy job histories that match multiple positions are excluded. Logs truncate candidate lists, so their target-in-logged-candidates measure is not full candidate recall. Full counters, model actions and per-family values are in the JSON evidence.

## Hypothesis verdicts

- **Context / neural fallback:** development evidence did not authorize neural fallback in any family. The held-out context policy scored no neural jobs. On the offline common-reference comparison, its query-average Top-1 was 43.21%, versus 45.98% for the existing hybrid; this run did not preserve the neural prediction advantage.
- **Deadline:** the calibrated next-read windows rejected all deadline-policy held-out scoring requests before enqueue. It prefetched zero pages in both conditions; any measured latency difference from context primarily accompanies avoiding work, rather than a demonstrated timely prefetch gain.
- **Adaptive net benefit:** the development-calibrated estimate rejected all eligible held-out actions in both conditions. It made zero prefetched insertions; this run does not establish that the controller can find profitable prefetch opportunities.
- **Combined:** it made zero prefetched insertions. In the zero-delay condition its paired delta versus none was -0.517 ms with CI [-3.692, 1.853]; with requested delay it was 0.741 ms with CI [0.173, 1.464]. The primary success criterion was not achieved.
- **Resource costs:** all selective variants still load the frozen neural checkpoint in the adapter even when neural fallback is disabled. In zero-delay runs the combined policy's largest process peak was 382.8 MB versus 303.3 MB for none; total CPU was 12.92 s versus 3.92 s. Individual scheduling CPU time was not isolated.
- **Next independent cycle:** consider lazily loading the neural model only when fallback is validated, and calibrating readiness on a less restrictive, observable demand horizon. Freeze any revised choices on development data and use new held-out seeds; do not tune to these results.

## Interpretation and limitations

Treat the paired intervals above as the verdict on the declared latency hypothesis for each condition. A negative mean alone is not a success if its 95% interval includes zero. Context selection may trade the neural ranker's offline accuracy for lower inference cost; deadline and adaptive controls can skip many or all predictions. No confidence threshold or policy was tuned on held-out results.

The candidate-based offline comparison does not simulate queue timing, per-query admission caps or runtime cache state. The actual job analysis is conditional and can contain missing/ambiguous alignments. OS caching, small family sample sizes, five seeds, one host, an artificial storage delay, multiple unadjusted policy comparisons, and possible overlap of parameter values with prior training limit generalization. The controller's readiness/benefit values are estimates, not measured causal savings. The opt-in adapter reuses the existing neural request path, which can parse current page layout even for a queued Markov action; that work is not separately timed.

Outputs and raw per-run traces live under ignored `backend/runs/selective-prefetch-20261027/`. The aggregate machine-readable results are `backend/selective-prefetch-results.json`. The frontend's SDSS and prior research exports remain separate.
