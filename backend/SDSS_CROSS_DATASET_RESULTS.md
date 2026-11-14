# Cross-dataset verification on the SeLeP-linked SDSS subset

The idle-neural execution changes were tested on the existing real SDSS DR7 subset. The run also compared newly trained SDSS idle weights against frozen P4 neural weights from the small synthetic database and the 8.62 GB synthetic database.

The results support cross-dataset CPU-efficiency improvements. They do not establish consistent additional latency gains or reliable transfer of the neural model's first-choice ranking.

## Dataset and isolation

The read-only SQLite snapshot contains 49,999 photometric objects and 50,000 spectra: 15,790,080 bytes, or 3,855 pages. These are real public astronomy measurements. The subset is ordered by spectrum ID and is not a random sample. Local SQL queries use six adapted SDSS templates in shuffled order; they are not SeLeP's original user query logs or PostgreSQL benchmark.

Original source experiment:

```text
D:\shortly-source\OS Project\paper-dataset-experiments\selep-sdss-dr7-20261027
```

All new training, weights, traces and evaluation outputs are separate:

```text
D:\shortly-source\OS Project\paper-dataset-experiments\selep-sdss-idle-20261114
    training-001/
    evaluation-001/
    report-001/
```

The original SDSS artifact manifest, dataset checksum, main dashboard snapshots and original research checkpoint passed before/after checks. The existing frontend's saved SDSS results were not replaced.

## What was verified

| Check | Method |
|---|---|
| Same approach on another dataset | Train separate SDSS query-level P4 weights with the existing B2 semantic features and width-64 MLP |
| Same neural weights on another dataset | Freeze each synthetic-database P4 state dictionary and execute it on SDSS features |
| Target-side adaptation | Fit candidate/Markov statistics on SDSS training queries; use SDSS metadata and page normalization |
| Execution optimization | Compare inherited native waiting with passive waiting and bounded feature caches |
| Prediction equivalence | Check cached versus original SDSS rankings/scores on 396 validation/exclusion cases |
| Correctness and preservation | Match SQL row-stream hashes, audit worker/admission events and verify original artifact hashes |

The transfer arms use **no SDSS neural fitting**, but they do use SDSS training-derived statistics. This is one-way frozen-neural transfer with target-side adaptation, not zero-shot transfer. Reverse transfer was not tested. Runtime tensor hashes match the frozen source tensors, and each source model performed 620 actual runtime neural-scoring calls across the experiment.

## Evaluation

The run retraced 600 training and 200 validation queries. Separate semantic and no-semantics idle models each ran six epochs, selected epoch one, and retained nonzero learned residual weights. Validation early-miss recall at budget 16 equaled the zero-residual Markov reference; that metric did not show a neural gain.

Fresh test combinations exclude all 1,100 original training/validation/test signatures and follow the original held-out SHA256 parameter partitions. There are **144 matched trials and 8,640 runtime SQL executions** across 12 policies.

The main conditions each use five seeds, 60 queries per policy/seed, a 32-page cache and 50 ms imposed idle gaps. One has no artificial delay; the other adds 0.2 ms requested shared-storage service. Additional tests vary the gaps and run with zero gaps and a 128-page cache. Each policy runs in a fresh process with randomized order. A full database SHA256 read before every trial makes this a warm-OS-cache evaluation, not a cold physical-disk benchmark.

## Mean query time

These are means across the five run means. Do not compare them directly with the original 8.62 GB study's timings: database size, query mix and workload length differ.

| Policy | No added delay | 0.2 ms artificial service |
|---|---:|---:|
| Same-cache no prefetch | 3.086 ms | 25.869 ms |
| Within-query Markov-2 | 4.915 ms | 37.055 ms |
| Original SDSS within-query neural hybrid | 5.974 ms | 35.253 ms |
| Original SDSS neural, passive waiting | 6.046 ms | 35.281 ms |
| Idle Markov, full page sets | 3.019 ms | 24.861 ms |
| Idle Markov, matched early-miss targets | 3.002 ms | 25.408 ms |
| Newly trained SDSS idle neural, before optimization | 3.142 ms | 25.130 ms |
| **Optimized SDSS idle semantic neural** | **3.084 ms** | **25.204 ms** |
| Selective SDSS idle neural | 3.022 ms | 25.173 ms |
| Retrained SDSS idle neural without semantics | 3.142 ms | 25.104 ms |
| Frozen small-database neural weights + SDSS statistics | 3.038 ms | 25.146 ms |
| Frozen 8.62 GB neural weights + SDSS statistics | 3.002 ms | 25.164 ms |

The optimized SDSS idle mean was 1.83% lower without added delay and 0.29% higher with artificial service than the unoptimized idle model. Both paired five-seed bootstrap intervals include zero. There is no consistent additional latency improvement from the execution optimization.

## CPU cost

CPU time covers a complete 60-query trial, including process threads; it is not mean elapsed time per query.

| SDSS comparison | Before | After | Reduction |
|---|---:|---:|---:|
| Idle neural, no added delay | 2,396.88 ms | 200.00 ms | 91.66% |
| Idle neural, artificial service | 2,721.88 ms | 365.63 ms | 86.57% |
| Original within-query neural, no added delay: waiting change only | 4,212.50 ms | 1,275.00 ms | 69.73% |
| Original within-query neural, artificial service: waiting change only | 4,500.00 ms | 1,490.63 ms | 66.88% |

The selective model used 19 neural calls versus 287 for the full optimized idle model across each five-seed main condition. Its latency was not consistently better. The no-delay full and selective idle models both supplied 540 useful insertions out of 1,292 prefetched pages.

## What transferred predictively

The diagnostic below uses 295 query boundaries on the same no-prefetch no-delay traces. **First-miss Top-1** checks whether the first proposal equals the next query's first demand miss. **Macro early-miss recall@8** measures coverage of its first 16 distinct misses with eight proposals. These differ from the older within-query next-page accuracy metrics.

| Predictor | First-miss Top-1 | Macro early-miss recall@8 |
|---|---:|---:|
| SDSS idle Markov, matched early-miss targets | 32.54% | 16.47% |
| SDSS-trained optimized semantic idle neural | 32.54% | 16.54% |
| SDSS-trained idle neural without semantics | 32.54% | 16.47% |
| Frozen small-database neural + SDSS statistics | 3.39% | 16.54% |
| Frozen 8.62 GB neural + SDSS statistics | 3.39% | 16.54% |

Frozen source weights executed correctly, but their first-choice ordering was much weaker on SDSS. Their top-eight coverage remained similar because the target-derived candidate sets and Markov statistics were shared. Similar runtime or set coverage therefore does not demonstrate successful transfer of the learned neural ranking. The SDSS-trained neural model also did not improve this first-choice metric over the matched Markov control.

These results support keeping dataset-specific training and reporting target-side adaptation explicitly. They do not justify describing the model as a dataset-independent or zero-shot predictor. The shuffled generated query order is another boundary: this test does not measure transfer to genuine exploratory user sessions.

## Evidence and reproduction

**168 backend tests passed.** Every runtime policy returned identical SQL row streams. Workers shut down cleanly; all useful idle insertions completed before their demand query arrived. Original source and target weights stayed unchanged during evaluation.

Local evidence under `report-001/` includes `RESULTS.md`, `all_trials.csv/json`, `aggregate.csv/json`, `paired_queries.csv`, `paired_seed_comparisons.json`, `query_level_quality.json`, `transfer-verification.json` and `audit.json`. Raw VFS traces, workloads and runtime diagnostics are in `evaluation-001/`; separate fitted weights and validation history are in `training-001/`.

Run from the main project's `backend/` folder, using new output directories:

```powershell
.\.venv\Scripts\python.exe -u sdss_idle_experiment.py prepare --out '..\..\paper-dataset-experiments\sdss-idle-new\training'
.\.venv\Scripts\python.exe -u sdss_idle_experiment.py run --plan '..\..\paper-dataset-experiments\sdss-idle-new\training\plan.json' --out '..\..\paper-dataset-experiments\sdss-idle-new\evaluation' --queries 60
.\.venv\Scripts\python.exe report_sdss_idle.py --plan '..\..\paper-dataset-experiments\sdss-idle-new\training\plan.json' --evaluation '..\..\paper-dataset-experiments\sdss-idle-new\evaluation' --out '..\..\paper-dataset-experiments\sdss-idle-new\report'
```

Reproduction requires the existing downloaded SDSS snapshot, its archived workloads/checkpoint, and the two source P4 checkpoints. No new download is required. The experiment refuses to overwrite output directories. Source files are `sdss_idle_workload.py`, `sdss_idle_experiment.py` and `report_sdss_idle.py`; tests are in `tests/test_sdss_idle.py`.
