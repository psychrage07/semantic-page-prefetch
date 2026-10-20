# Neural improvement cycle — 2026-10-20

The primary objective is lower next-page prediction error, followed by lower
end-to-end query latency when measured runtime costs permit it. Higher Top-1
and Top-5 accuracy is better. Loss alone is not the success criterion.

## Preserved evidence

The v2 final artifacts, original research dashboard, protocol and frozen
manifest remain historical evidence. The new work uses
`runs/neural-improvement-20261020/` on branch `work/neural-improvement`.
Historical missing manifest artifacts cannot be certified as present merely
because the manifest file itself is unchanged.

## Completed foundation

- Use training normalization consistently at prediction time.
- Resolve B3 SQL parameters from the same query context in every stage and
  distinguish missing parameters from an available parameter set.
- Correct paired-effect aggregation, centered bootstrap p-values,
  reproducible bootstrap seeds and monotone Holm adjusted p-values.
- Distinguish model initialization seeds from workload replication units.
- Repair the simulator diagnosis CLI and use the actual measured strategy.
- Keep the action-family oracle simulator-only; its headroom is a lower
  bound over that family, not an upper bound for every prefetch policy.
- Recompute would-be misses for every cache cell and mark absent/failed
  concurrent-storage cells inconclusive.
- Validate exact feature parity, checkpoint replay and the actual VFS worker.

## Data and selection contract

Database: existing static 8,615,698,432-byte SQLite snapshot. Development
collection: 600 queries at seed 101 and 300 each at seeds 102–110, totaling
3,300 independently generated workload queries.

The promoted screen trains on workload seeds 101–104 (1,500 queries before
split filtering), validates on 105–106 (600 before filtering), and reserves
107–110 for development runtime checks. Parameters are grouped into six
stable hash buckets; templates are held out as whole templates. Query IDs
retain their workload-seed identity when pooling.

Screening samples at most 16 transitions per execution, identically across
models. Each execution has equal aggregate training weight. Confirmation
evaluates **all** eligible transitions without sampling. Macro Top-1 averages
query accuracies; micro Top-1/Top-5 weight transitions. Candidate misses remain
errors and conditional ranking accuracy is identified separately.

## Controlled search

Candidate experiments compare legacy, Markov-2-first, balanced and B-tree
proposals with bounded sets. The promoted comparison uses the same selected
generator for every neural model and reports native Markov controls separately.
B-tree proposals parse every child pointer of the current page, independent
of whether those children appear in a future trace. A complete compact dbstat
snapshot supplies static metadata for all 2,103,442 pages in about 8.4 MB of
arrays; feature availability does not reveal future trace membership.

Model experiments compare original history/semantic inputs, B2/B3, binary
loss over all candidates, the original recipe of 16 sampled negatives,
listwise and pairwise ranking, semantic ablation, structured parameters,
miss-only labels, neural corrections to Markov log scores, width 64/128,
prior weight 1/0.25, and learning rate 0.001/0.003. Validation macro Top-1
selects the model and epoch. An untrained random network is ineligible;
an epoch-zero residual network is explicitly a Markov fallback.

Two exploratory structural runs were stopped and recorded as excluded when
trace-filtered metadata was identified as a possible availability cue. Their
scores are not evidence. Failed and superseded attempts remain in ledgers.

## Freeze and confirmation

`scripts/assemble_neural_screen.py` combines matched validation-only trials.
`scripts/evaluate_neural.py freeze` writes checkpoint hashes, calibration,
partition definitions, a comparison family and a source archive before tests.
Binary-loss controls receive their best validation-selected negative recipe.

Fresh workload seeds 921–925 are confirmatory. Seeds 911–915 were used in v2
and can only support a labeled amended comparison. No model, feature,
threshold or candidate rule may be adjusted from confirmatory outcomes and
then described as independently confirmed on the same seeds.

Paired accuracy differences are clustered by independently generated workload
seed and corrected across the frozen family with Holm. Report supported,
unsupported and inconclusive comparisons, including failures to generalize.

## Runtime and final artifacts

`scripts/benchmark_neural_runtime.py` selects among three declared neural
scheduling policies using development seeds 107–108. Confirmation compares
no prefetch, native Markov-2 and the frozen neural policy at k=1/4/16 with
0.2 ms requested artificial service delay, plus k=1 without injected delay.
Prediction, feature construction, queueing, wasted reads, contention and
worker drain are measured. Runtime B-tree proposals use the current cached
page bytes and never reread it outside the modeled device path.

The simulator is separately calibrated/validated. A 60-cell cache/delay/channel
grid measures action-family headroom under explicit simulation assumptions.
Neither VFS reads nor an artificial delay establishes physical disk I/O.

Final deliverables: frozen checkpoints, machine-readable offline/runtime
results, paired uncertainty, error breakdown, plots, a generated report,
reproduction commands and a dashboard-compatible result export. Report
prediction gains and latency gains as separate outcomes.

## Completed results and limits

This focused improvement cycle is complete. On fresh unseen parameters,
macro Top-1 was 16.31% for the semantic binary-loss control, 21.63% for
structured semantic ranking, 46.32% for B2 and 47.74% for the selected
neural–Markov hybrid. Native Markov-1 and Markov-2 achieved 41.59% and
44.83%. The hybrid gained 2.02 percentage points over its semantic ablation.
Those parameter-test gains survived the declared Holm correction.

B3-based hybrids were screened on validation data; the best reached 46.89%,
versus the selected hybrid's 46.96%. B3 was not promoted to final confirmation.
Its validation result must not be substituted into the fresh-test comparison.

Unseen templates remain a failure to generalize: the selected checkpoint is
an untrained residual Markov fallback, at 2.90% macro Top-1. Candidate recall
was 20.68% micro on that split. On held-out parameters, range, aggregate and
graph templates still suffered large candidate-coverage failures.

All 60 runtime confirmation runs completed without prefetch read errors or
unclean worker shutdown. The neural policy was slower than no prefetching in
all four scenarios: 15.62 vs 5.27 ms without injected delay; 121.57 vs 102.65 ms
with serialized delayed service; 113.07 vs 103.18 ms at k=4; and 113.09 vs
104.03 ms at k=16. This cycle improved prediction, not query latency.

All 11 zero-delay simulator validation cells met the 15% error target,
including held-out cache configurations. The raw CPU fit was not physically
valid; an explicitly recorded nonnegative refit preserved all measurements.
The 60-cell exploratory grid reopened 40 restricted-family concurrent cells.
That is modeled headroom, not a measured neural speedup or physical-device
claim. Exact neural simulation and concurrent-device calibration remain
future work.

The next cycle should improve candidate coverage for unseen templates and
measure scheduling/feature cost before expanding neural architectures. GRUs,
Transformers and tree rankers remain optional experiments, not completed work
or evidence that this search found the best possible model. Any new tuning
requires new untouched confirmation seeds.

## Deliverables and checks

- [Full measured report](NEURAL_IMPROVEMENT_RESULTS.md), including error
  breakdowns, uncertainty, runtime telemetry and reproduction commands.
- `public/pagewise-neural-improvement.json` and the dashboard's separate
  **Neural improvement** tab, with split switching and measured JSON export.
- Standalone figures in `backend/figures/`; raw traces, frozen weights and
  detailed append-only ledgers under the ignored study directory.
- 106 backend tests passed. Frontend production build and TypeScript check
  passed. Chrome checks passed for measured values, both splits, JSON download,
  mobile overflow, existing navigation and absence of browser exceptions.
- Frozen predictor source and all 14 checkpoint hashes match. Confirmation,
  amended and runtime ledger chains verified (10, 10 and 60 rows respectively).
