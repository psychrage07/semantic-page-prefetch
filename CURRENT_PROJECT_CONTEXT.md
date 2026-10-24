# Pagewise: current context and complete frontend explanation

Verified against the local source and artifacts on 24 October 2026 (Asia/Kolkata).

**Update later on 24 October:** the default Live demonstration mode now executes the actual trained selected neural hybrid. The earlier small A/B pipeline remains under **Build a small demo**. The following sections on its seven stages and 100-query example describe that small mode, not the selected-model test.

**Latest integration update:** the trained-hybrid pipeline now has nine stages, including fresh offline prediction evaluation after the six timed policy trials. `backend/live_evaluation.py` loads the actual semantic checkpoint and evaluates every eligible transition from the new no-prefetch VFS sequences, including candidate misses. It reports query-average Top-1, transition Top-1/Top-5, candidate recall and family breakdowns. Capped examples align the exact history to its observed next page and include table/index, page type, depth and plan context. Fresh values can overlap training and do not replace held-out results. `/api/selected-results` publishes the latest completed evaluated run; the overview, research results, detailed comparison and prediction inspector share it. Older runtime-only runs are excluded from this feed. Historical A/B views remain under explicit advanced controls. The frozen predictor source and weights are unchanged.

**Interactive workspace update:** `/?view=workspace` adds a real SQLite explorer, read-only SQL execution, live VFS events, actual runtime prediction jobs, prefetch monitoring, SQL benchmarks, artifact downloads and captured Python terminal output. It uses `backend/workspace.py` around the existing APSW TraceVFS and PredictiveRuntime. New small builds use the existing seven-stage pipeline; the research option verifies matching frozen assets. New runs are isolated under ignored local-lab directories. See [WORKSPACE_GUIDE.md](WORKSPACE_GUIDE.md).

## Trained-hybrid live mode

**Test trained hybrid → Run trained hybrid** loads the existing frozen `hybrid_semantics` checkpoint, with no weight fitting. It verifies its hash against both `freeze.json` and the frozen runtime policy, verifies the predictor source hash and the database/metadata snapshot identity. The model runs on its matching 8.62 GB database.

The user chooses 10/20/30 queries per policy/seed. Two newly generated workload seeds are used for no prefetch, Markov-2 and `model_ranker` (the selected neural hybrid). Each policy receives matching query workloads, checked by their workload-file fingerprints. Policy ordering varies by workload seed. Results come from six actual policy executions, with fresh means/medians, scored jobs, prefetched/useful pages and capped actual ranking samples. The checkpoint hash is checked again after execution. The nine job stages include verification, six policy runs, fresh prediction evaluation and result preparation.

The live result is saved as `hybrid-live.json`. It does not recompute the saved 47.74% held-out accuracy. Generated parameter values can overlap training; diagnostic logging is enabled for every policy and the OS cache is not reset. This is a live inference/runtime demonstration, not a new confirmatory study. No speedup is assumed. Missing research artifacts disable this mode with an explicit reason. A fresh clone can still use the small A/B training mode.

New integration files: `backend/live_hybrid.py`, the trained preset in `backend/local_lab.py`, and `src/TrainedHybridResults.tsx`. These are outside the frozen predictor source directories; the frozen weights and historical study artifacts remain unchanged. Backend tests now total 114 passing tests. These changes and the faculty redesign are included in this repository revision.

## 1. Current project context

- Project: **Pagewise — semantics-aware SQLite page prefetching**.
- Repository: https://github.com/psychrage07/semantic-page-prefetch
- Local checkout: `D:\shortly-source\OS Project\semantic-page-prefetch`.
- Branch: `main`.
- Previous integration baseline: `17acb5efb2e3adc2b06ea97aa92ded4b63ffc828`, **Connect dashboard to local Pagewise experiments**.
- This revision includes the faculty-facing frontend redesign and trained-hybrid live execution.
- Current local production preview: `http://localhost:4173/?view=overview`. Python API: `127.0.0.1:8765`.
- The current project has database generation, real SQLite read tracing, page metadata, sequence preparation, model training, held-out evaluation, an asynchronous bounded prefetch cache, policy comparisons, study scripts, reports and a React frontend.
- The main faculty navigation is **Project overview → How it works → Live demonstration → Research results**.
- Individual CLI commands and research scripts remain under **Advanced research tools**. Older dashboards and fixtures remain under the collapsed **Research workspace**.
- Validation: 112 backend tests, TypeScript check and production build passed. Browser verification completed both the small seven-stage pipeline and the trained eight-stage pipeline, displayed results, checked primary navigation and checked mobile overflow. The trained browser run executed 120 queries and 409 neural scoring jobs. There were no captured JavaScript exceptions in that verification.
- The Review 2 document was kept separate. A Review 3 IEEE document has not been prepared.

### What the research established

The final parameter-split selected predictor is a **neural–Markov hybrid**, named `hybrid_semantics`. It learns a semantic neural correction to a training-derived Markov prior. It reached 47.743% query-average Top-1, compared with native Markov-2 at 44.826%, on the new-study held-out-parameter evaluation. The difference is 2.917 percentage points.

Semantics helped prediction, but the selected neural policy was slower than no prefetching in every measured runtime scenario. Unseen SQL query types remain difficult. These findings describe the tested conditions, not every database or device.

The template-split selected checkpoint is different: `hybrid_no_semantics` selected at epoch zero. It is effectively a Markov fallback with no learned neural increment. B3 was screened in development but was not selected for final confirmation.

## 2. The frontend, backend and data files

```text
Browser / React dashboard
    ├─ reads saved final-study JSON → explains the research findings
    └─ sends a local API request → Python job runner
                                  → Pagewise CLI / research script
                                  → SQLite / PyTorch / prefetch runtime
                                  → measured files on disk
                                  → dashboard JSON export
                                  → browser results table
```

The frontend does not train a model in JavaScript. It sends the request, polls status every two seconds and renders the measured outputs. Python runs on the same computer. The browser and backend communicate through the local Vite proxy; no cloud experiment service is involved.

| Evidence source | Used by | Meaning |
|---|---|---|
| `public/pagewise-neural-improvement.json` | Project overview, Research results | Saved final new-study measurements |
| `backend/runs/local-lab/<run-id>/dashboard.json` | Live demo result cards/table | Newly computed small development run |
| `public/pagewise-dashboard.json` | Advanced historical dashboard, default advanced comparison | An older exported experiment; not the selected final hybrid study |
| Constants and formulas in `src/App.tsx` | Legacy fixture controls, illustrative ablation and break-even panels | Interface illustrations, not newly measured SQLite results |

The raw large database, traces and frozen checkpoints are local ignored files. A fresh clone can create small demo data and models, but reproducing the exact final study requires its large database and frozen inputs. Archive-dependent commands can report missing inputs even though the API itself works.

## 3. Page: Project overview

This is the faculty starting screen: `/?view=overview`.

### What it shows

- **Predict the next page. Measure the real cost.** This states the two separate objectives: correct page prediction and lower query time.
- **Our selected approach: Neural–Markov hybrid.** It combines repeated page-transition patterns with a learned neural correction using database context.
- **Page history:** recent page IDs and their changes.
- **Database semantics:** table/index identity, page type, B-tree depth, query-plan class and query context available to the model.
- **Observe → Predict → Prefetch → Measure:** the implemented system flow.
- **Our neural hybrid: 47.74%.** The saved parameter-split query-average Top-1 result.
- **Markov-2: 44.83%.** The matching native baseline in that same evaluation.
- **Improvement: +2.92 pp.** `47.743238 − 44.826422`, rounded. `pp` means percentage points; it is not a relative 2.92% improvement.
- The bars show the saved query-average Top-1 values for Model A, Model B, Markov-1, Markov-2 and the selected hybrid.
- The two finding panels explain the prediction gain and the absence of measured runtime speedup.

This screen loads saved research results; opening it does not run another evaluation. The buttons open the live demonstration, the explanation or the research evidence. If the export is unavailable, example scores are not substituted.

## 4. Page: How it works

This screen explains four implementation steps:

1. **Observe the database.** Generated SQL queries run against a static database. An APSW wrapper around SQLite's VFS records reads reaching the file interface. `dbstat` supplies page metadata, and `EXPLAIN QUERY PLAN` supplies query-plan context.
2. **Predict the next page.** Models receive the recent page history and rank a bounded set of candidate next pages. Candidate and transition statistics are derived from training executions.
3. **Fetch useful pages early.** A background worker issues predicted reads. A bounded LRU cache stores pages below SQLite's own pager.
4. **Test the complete system.** Offline prediction accuracy and actual query latency are measured separately. More correct predictions do not automatically offset inference, scheduling or extra-read costs.

It also explains database page, Top-1 accuracy and query latency, and explicitly distinguishes the saved final study from the small live demo.

## 5. Page: Live demonstration

### Main controls

- **Local service ready:** the browser can read the backend job API. It is a connectivity indicator, not proof that an experiment has passed.
- **Queries to trace:** 100, 150 or 300. This changes only the trace-collection stage. It does not change the fixed 30 runtime queries per policy and seed.
- **Start live demo:** sends a real job request. One job is allowed at a time.
- **Seven-stage display:** reflects the actual backend command currently executing, not a timer-based animation.
- **Stop this run:** asks the local backend to terminate the active process tree.
- **Previous local run / timestamp:** a saved earlier run, not a claim that the experiment has just executed. The screen resumes an active job or the most recent completed seven-stage demo.
- **Execution details:** optional raw backend output and error details.
- **View measured results:** scrolls to the automatically loaded result cards and policy table.
- **Download results:** downloads the measured JSON export for the displayed demo.
- **Inspect the page-read samples:** opens the advanced inspector with this demo's snapshot.

### Exactly what the seven stages do

| Stage | Actual work | Principal artifacts |
|---|---|---|
| Create database | Generates 500 customers, 150 products, 3,000 orders, order items and a 600-node graph with two outgoing edges per node | `pagewise-demo.sqlite`, metadata JSON |
| Observe page reads | Runs the chosen number of generated SQL queries with workload seed 1234 and a 128 KiB SQLite cache setting | `trace/trace.jsonl`, `query_results.csv`, `workload.jsonl`, `page_map.json`, `plans.json` |
| Prepare examples | Expands the observed read ranges into per-query page sequences | `trace/sequences.jsonl` |
| Train models | Builds the training-only candidate index and trains development Model A and Model B | `models/model_a.pt`, `model_b.pt`, `candidate_index.json`, split files, `training_report.json` |
| Test predictions | Predicts the next observed page on held-out test executions; compares neural and simple policies | `evaluation/predictor_metrics.json`, CSV and prediction logs |
| Measure query time | Executes 30 queries for each of eight runtime configurations at seeds 101 and 202 | `suite-local/suite_runs.json`, `suite_summary.json`, per-policy timings and cache counters |
| Prepare results | Summarizes existing evaluation and runtime files into a compact measured export | `dashboard.json` |

The generation seed is 23. The training/split seed is 7. Fixed seeds improve repeatability: a fresh run recomputes the pipeline in a new directory, but it does not necessarily generate a different random workload. Runtime timing can still vary with CPU load, caches and scheduling.

The generated schema includes customers, categories, products, orders, order items and graph edges. The workload includes indexed customer-order lookups, multi-table joins, product-sales lookups, time-range scans, recursive graph walks and customer-profile queries.

### Training in the small demo

With 100 eligible sequences, the verified run split 70/15/15 by complete query execution: 70 training, 15 validation and 15 test. Individual reads from one execution are not randomly scattered across all three sets.

The demo uses at most 16 candidates, five training epochs and a hidden width of 32. The MLP has hidden layers, ReLU activations and dropout. Binary cross-entropy trains it to give the actual next-page candidate a higher score than negative candidates. The checkpoint with the best validation loss is retained. Only the test split supplies the displayed offline test accuracy.

- **Model A:** last eight page IDs, their deltas, and candidate-page relationship features.
- **Model B:** the same basic features plus page metadata and query context.
- **Standard hybrid in the demo:** use the Markov-2 successor ranking when an exact two-page history is known; otherwise use Model B.
- **Final selected neural hybrid:** a separate residual ranking model discussed in the final-study section below. It is not retrained by this one-click demo.

### Eight runtime configurations versus seven displayed rows

The suite executes stock SQLite, no prefetch, sequential, Markov-1, Markov-2, Model A, Model B and the standard hybrid. That is `30 × 8 × 2 = 480` query executions. The friendly result table displays seven cache-policy rows; stock SQLite is retained in the raw suite artifacts.

`stock` uses SQLite without Pagewise's custom cache. `none` retains the custom cache but issues no prefetch predictions. Comparing against `none` helps isolate prediction/prefetch cost within the same cache layering.

The runtime uses 128 custom-cache pages, degree 2, no injected delay and paired workload seeds. With 4,096-byte pages, 128 custom-cache pages hold about 512 KiB. This is distinct from the separate 128 KiB SQLite pager-cache setting.

### How the verified demo numbers were calculated

Verification run: `backend/runs/local-lab/345434fbc4e3/`, executed on 24 October 2026.

The database had 147 pages of 4,096 bytes: 602,112 bytes. It contained 7,805 order-item rows and 1,200 graph edges.

The 15 held-out test queries contained 209 eligible next-page transitions. The first page of a query has no previous history, so it is not a next-page prediction target in this evaluation.

| Predictor | Correct first predictions / transitions | Displayed Top-1 |
|---|---:|---:|
| Sequential | 58 / 209 | 27.75% |
| Markov-1 | 136 / 209 | 65.07% |
| Markov-2 | 157 / 209 | 75.12% |
| Model A | 1 / 209 | 0.48% |
| Model B | 24 / 209 | 11.48% |
| Standard hybrid | 138 / 209 | 66.03% |

For example, `24 ÷ 209 × 100 = 11.483%`, displayed as 11.48%. These are transition-weighted accuracies. No-prefetch is not a predictor, so it is not assigned a fabricated zero accuracy.

For runtime, each query is timed around SQL execution and iteration through its results. Each 30-query trial has its own median. The friendly table then displays the median of the two trial medians:

- No prefetch: trial medians 0.4584 and 0.58235 ms → 0.520375 ms → **0.52 ms**.
- Model B: trial medians 1.3299 and 1.2474 ms → 1.28865 ms → **1.29 ms**.

This is not the pooled mean of all queries. It also does not include database generation or training time. Worker drain/shutdown and total run time are recorded separately. The result shows that Model B was slower than no prefetching in this demo despite having some correct predictions.

## 6. Page: Research results

This page loads the saved final study from `public/pagewise-neural-improvement.json`. It does not invoke the live demo.

### New query values

The SQL template types are familiar, but parameter partitions and confirmatory workload instances are held out. The fixed database is 8,615,698,432 bytes, approximately 8.62 GB. Fresh confirmatory workload seeds are 921–925. There are 292 held-out queries and 22,429 eligible transitions in this split.

| Predictor | Query-average Top-1 | Transition Top-1 | Transition Top-5 |
|---|---:|---:|---:|
| Model A, binary loss | 11.893% | 7.062% | 15.319% |
| Model B, binary loss | 16.309% | 10.807% | 17.067% |
| Model B, structured listwise | 21.631% | 20.126% | 28.142% |
| B2, binary loss | 46.319% | 34.076% | 35.534% |
| Markov-1 | 41.587% | 31.392% | 35.044% |
| Markov-2 | 44.826% | 33.876% | 35.338% |
| Selected neural hybrid | 47.743% | 34.973% | 35.976% |

The final hybrid score is a learned neural score plus a weighted log Markov prior. The Markov probabilities come from training transitions, with second-order-to-first-order backoff. Semantic features let the network adjust this prior. The selected parameter configuration uses B2 features and listwise ranking; validation selected its epoch before confirmation.

**B2** adds Markov statistics to semantic inputs. **B3** adds structured query parameters. **Structured Model B** adds structural features to the semantic ranker; it is not automatically synonymous with B3. **Binary loss** scores candidate correctness independently. **Listwise loss** trains candidates as a ranked group. **Ablation** removes information, such as semantic features, to test its contribution.

Development trained and selected models using separate workload seeds. Confirmatory scores evaluate all eligible transitions rather than the sampled transitions used in screening. Checkpoints, candidate rules and the comparison family were frozen before the fresh confirmation data was collected. Do not compare the screened B3 validation score as if it were a final test score.

### New query types

Whole SQL templates are held out. This split contains 593 queries and 58,827 transitions. The selected policy achieved about 2.899% query-average Top-1. Candidate recall was about 20.681%, compared with about 58.380% on the parameter split.

The selected template checkpoint is an epoch-zero Markov fallback, not evidence of a learned neural generalization improvement. The low scores and missing candidate coverage limit the scope of the project claims.

### Actual runtime table

The runtime compares the frozen selected neural policy, Markov-2 and no prefetching. Each policy/scenario includes 60 queries at each of five seeds, hence 300 executions. Four scenarios × three policies × five seeds = 60 runs and 3,600 query executions.

| Scenario | No prefetch mean | Markov-2 mean | Selected neural mean |
|---|---:|---:|---:|
| 1 channel, no injected delay | 5.271 ms | 10.596 ms | 15.622 ms |
| 1 channel, requested 0.2 ms delay | 102.652 ms | 187.777 ms | 121.567 ms |
| 4 channels, requested 0.2 ms delay | 103.183 ms | 109.923 ms | 113.067 ms |
| 16 channels, requested 0.2 ms delay | 104.032 ms | 110.601 ms | 113.092 ms |

The neural runtime policy used 512 cache pages, one prefetch worker, degree 1, threshold 0.7 and scheduling on every fourth eligible miss. These runtime values are means, whereas the small demo table uses median-of-trial-medians. They are not directly interchangeable.

Channels are abstract service slots. The injected-delay label is requested artificial service time; the host's calibration measured approximately 0.5234 ms for the requested 0.2 ms sleep. Neither an injected delay nor a VFS read establishes physical NVMe/HDD performance.

### Full research evidence

The collapsed panel retains:

- Macro and micro accuracies, Top-5, candidate recall and all matched controls.
- Development-only B3 screen results, separate from confirmation.
- Paired model differences, bootstrap intervals, Holm-adjusted p-values and favorable-support decisions.
- Query-time comparisons, prefetched-page counts, useful rates, scoring time, queue wait and stale-job drops.
- Simulator exploration and evidence limitations.

**95% confidence interval:** a bootstrap uncertainty interval. The final analysis resamples workload seeds and paired queries, preserving the clustered evaluation structure.

**Holm adjustment:** controls multiple-testing error across the frozen comparison family. **Favorable support** also depends on direction: higher accuracy is favorable, lower latency is favorable. A statistically supported slowdown is not a supported speedup.

**Simulator:** explores virtual-time assumptions. Forty of sixty restricted-family concurrent configurations reopened exploratory oracle headroom. Eleven zero-delay validation cells were within the 15% error target, but exact neural replay and concurrent-device timing remain unvalidated. Simulation does not overturn the actual measured slowdowns. Historical `NO_HEADROOM` applies to its tested shared-device setup.

## 7. Advanced research tools inside Live demonstration

The collapsed workspace exposes 22 CLI commands and 14 Python research-script entry points. The form follows the backend's command definitions rather than maintaining a separate hardcoded option list.

| Group | Functions |
|---|---|
| Setup and data | Check environment; generate database; trace reads; build sequences |
| Models and accuracy | Train A/B; multi-seed training; evaluate; parameter/template offline studies; ablations |
| Runtime | Individual prefetch run; paired policy suite; scheduler/cache sensitivity sweeps; oracle headroom; simulator-cell diagnosis |
| Reports and exports | Rebuild/recover suites; Markdown reports; consolidated research reports; dashboard exports |
| Research scripts | Neural trace collection/pooling; ranker improvement screens; model freeze/evaluate; selected-model runtime benchmarks; simulator studies; figures and integrity checks |

Required options carry an asterisk. Blank optional options retain backend defaults. Paths refer to local project data/results folders. Commands needing models, traces or calibration files require those prerequisites. Run status displays backend success/failure, not a fabricated success animation.

The raw artifact browser previews JSON, JSONL, CSV, Markdown, text and logs. Browser previews have a size limit. A model checkpoint is a binary `.pt` file; a SQLite database is binary; neither is meant to render as a JSON preview.

## 8. Research workspace: Page-read inspector

This page opens a capped trace sample from the currently loaded historical or demo snapshot. It supports searching page/query/plan text, selecting a page row and exporting the browser sample. It does not rerun SQL when a row is clicked.

- **Query:** one query execution identifier.
- **Page:** the database page number associated with the observed read range.
- **Source:** demand/preload origin information retained by the export.
- **Plan:** query-plan text or its classified form.
- **Cache hit/miss:** whether Pagewise's custom cache supplied the request.
- **Prediction:** a sampled selected candidate associated with that query.
- **Microseconds (`us`):** millionths of a second. The current measured table's Time field represents the exported read duration, not a reliable global event timestamp.
- **Leaf page:** contains B-tree records. **Interior page:** contains child references. **Overflow page:** stores payload that does not fit in a cell. **Depth:** level within the B-tree. **Object:** owning table or index when actually available from metadata.
- **Query plan:** SQLite's chosen access strategy, such as an indexed search or scan.

### Important current inspector limitation

The legacy display adapter is incomplete. In measured mode it currently supplies defaults/approximations for some side-panel fields: previous page is inferred as `page − 1`, page type is set to `leaf`, depth to zero and workload to mixed. It associates the first sampled prediction by query, not by an exact event-to-prediction identifier. Its inferred useful/outcome fields and displayed confidence percentage are therefore not authoritative per-event causal measurements.

Use the raw trace, page map, prediction logs and runtime counters for exact claims. This display limitation does not change the backend-computed accuracy or latency tables. The friendly main walkthrough does not rely on those approximations.

If the loaded historical snapshot has no read samples, the inspector correctly shows an empty state. Loading a live demo gives it the available demo samples. Exporting the trace exports the capped browser sample, not necessarily the complete raw trace.

## 9. Research workspace: Detailed comparison

The strategy scorecard uses the loaded dashboard snapshot for available p50/p95 latency, custom-cache hit rate, useful/wasted counts and offline Top-1 accuracy. A local demo snapshot can be loaded into this workspace; otherwise the advanced view initially uses the older committed snapshot.

- **p50 / median:** middle of sorted query times. In suite scorecards, the value is the median across trial-level medians.
- **p95:** 95th percentile, showing slower-tail behavior. Suite scorecards use the median across trial-level p95 values.
- **Cache hit rate:** custom-cache hits divided by custom-cache demand requests, multiplied by 100. Hits can arise from demand-loaded pages as well as prefetches.
- **Useful:** in this legacy measured scorecard, an average count of useful prefetches per trial, not a percentage.
- **Wasted:** in this scorecard, an average count of unused prefetches evicted, not the complete total-waste percentage.
- **Top-1:** offline predictor accuracy from the export, independent of whether the corresponding runtime policy was measured.
- **Latency change:** `(baseline p50 − policy p50) / baseline p50 × 100`; a lower policy time is favorable.
- **n/a:** unavailable or inapplicable measurement; it is not zero.

The lower **ablation preview** is explicitly illustrative: its constant feature-removal values are not this project's measured ablation results. The **break-even calculator** uses `ceil(inference cost / assumed miss delay)`. It ignores overlap, extra I/O, cache pollution and contention, so it cannot establish a real speedup. The old hardware-like fixture profile names are illustrative labels, not calibrated hardware claims. Implementation is PyTorch; the legacy calculator's reference to ONNX is not evidence of an ONNX deployment.

The **oracle** is a diagnostic with future information. Its offline Top-1 can be 100% by construction; that does not make its runtime infinitely fast or establish a universal upper bound for all scheduling policies.

## 10. Research workspace: Historical dashboard

This preserves the earlier dashboard. It can show saved metrics, measured latency series and capped alignment samples when the loaded export has them. It also retains a legacy workload/profile/cache/degree preview.

The preview formulas scale predefined fixture values and do not execute the backend. Their cache choices are labelled MB, whereas real backend settings use explicit SQLite KiB and custom-cache page counts. Do not equate those controls with the live demo's settings.

Historical export numbers, final new-study numbers and live-demo numbers come from different artifacts and evaluation procedures. A high number in an older snapshot must not replace the final confirmed score or be averaged with it. The Run button opens the real live-demo workspace rather than playing a fake progress timer.

The **Prediction alignment** colors summarize the capped display sample. They are not a full timing-failure or useful-prefetch opportunity analysis; the adapter limitation above applies to inferred associations.

## 11. Research workspace: Protocol and fixtures

This preserves the detailed research protocol and feature explanation. It explains matched controls, grouped query splitting, multiple seeds, real versus injected delay and static read-only database assumptions. Its **Export fixture config** button exports the legacy preview configuration; it is not an export of the final selected study's frozen policy. Use the tracked study configuration/declaration and research report for the final policy.

## 12. Shared terminology

| Term | Meaning in Pagewise |
|---|---|
| SQLite | The embedded database engine whose reads are being studied |
| Page | A fixed-size database block; generated databases use 4,096 bytes |
| Prefetch | Read a predicted page before demand reaches it |
| Candidate | One proposed next-page ID the ranker may choose |
| Candidate recall | Percentage of targets present in the candidate set; a missing target cannot be ranked correctly |
| Ranker | A model scoring and ordering candidates |
| MLP | Multilayer perceptron, the feed-forward neural network used here |
| Semantics | Database/query meaning represented by metadata and query context |
| Markov-1 | Next-page statistics conditioned on the previous page |
| Markov-2 | Next-page statistics conditioned on the previous two pages, with backoff |
| Prior | Training-derived baseline preference for a candidate |
| Residual/correction | A learned adjustment added to the prior score |
| Seed | A fixed initializer controlling repeatable data/workload/model randomness; model seeds and workload replication units are different concepts |
| Epoch | One training pass through the training examples |
| Checkpoint | Saved model weights and required configuration |
| Train / validation / test | Fit weights / choose model settings / evaluate held-out performance |
| Template | SQL structure; parameters are its concrete bound values |
| VFS / APSW | SQLite file-interface abstraction / the Python SQLite wrapper used to instrument it |
| xRead | A read request that reaches the wrapped file interface |
| dbstat | SQLite metadata exposing page/object/B-tree information |
| LRU | Least recently used eviction: remove an old untouched cached page when space is needed |
| Asynchronous worker | A background thread handling predictions and prefetched reads |
| Degree | Maximum number of selected pages issued per prediction job |
| Threshold | Candidate-score cutoff for issuing a prefetch; not a guarantee of that percentage correctness |
| Queue / queue wait | Pending background jobs / time before a worker starts one |
| Stale job | Prediction work whose query has already ended |
| Coalesced job | Redundant work merged or avoided |
| Cache pollution | Speculative pages displacing more useful cached content |
| In-flight miss | Demand arrives while the desired prefetch is still unfinished |
| Lead time | Time between a prefetch becoming available and subsequent demand |
| Useful prefetch rate | Useful prefetched pages / total prefetched pages × 100 |
| Total wasted prefetches | Unused prefetches evicted plus unused prefetched pages still resident at the end |
| Demand coverage | Useful prefetched pages / all custom-cache demand requests; this is not a miss-opportunity denominator |
| Inference/scoring time | CPU time building candidates/features and ranking; exact included components depend on the named counter |
| Latency | Time taken; milliseconds are thousandths of a second |
| Bootstrap | Resampling procedure used to estimate uncertainty |
| Holm | Multiple-comparison correction used by the final study |
| Frozen | Selection/configuration fixed before confirmation; it does not mean the local files are automatically available in a fresh clone |
| SHA-256 | File/content fingerprint used to check evidence integrity |
| Fixture | Predetermined illustrative data, not a measurement from a new experiment |
| Artifact | A file produced by a run, such as a trace, metric table or model checkpoint |

### Macro, micro and Top-5

For query q with n_q eligible transitions and c_q correct first predictions:

```text
Macro Top-1 = mean across queries of (100 × c_q / n_q)
Micro Top-1 = 100 × sum(c_q) / sum(n_q)
Micro Top-5 = 100 × transitions with target among first five / all transitions
```

Macro gives every query equal weight. Micro gives every transition equal weight. A long scan therefore has more influence on micro accuracy. Top-5 must be at least Top-1 when both use the same weighting. The selected model's 47.743% macro Top-1 and 35.976% micro Top-5 use different weightings; its comparable micro Top-1 is 34.973%.

Candidate misses remain errors in the reported final accuracies. Conditional accuracy among only available candidates is a separate diagnostic, not the main score.

## 13. Why correct predictions can still be slower

The foreground query must pay for observation, scheduling and possible contention. Background work pays for candidate generation, feature extraction, model scoring and page reads. Predictions may be wrong, arrive late, or evict useful pages. The OS cache may already serve demand reads quickly, leaving little latency to hide.

Consequently, a higher cache hit rate or a correct next-page prediction is not proof of a speedup. Only measured end-to-end query times under a stated setup establish that outcome. The current measured research result is prediction improvement without demonstrated runtime acceleration.

## 14. Evidence and code locations

- `src/ProjectOverview.tsx`, `HowItWorks.tsx`: faculty explanation and saved-study summary.
- `src/LocalLab.tsx`, `DemoResults.tsx`: guided demo, job polling and automatic result rendering.
- `src/AdvancedLab.tsx`: individual commands, logs and artifact browser.
- `src/NeuralStudy.tsx`, `StudyDetails.tsx`: final-study summary and detailed evidence.
- `src/App.tsx`: navigation, loaded dashboard snapshot and legacy advanced pages.
- `backend/local_lab.py`: loopback API, command schema, seven-stage preset and job subprocesses.
- `backend/pagewise/models.py`, `evaluate.py`: development model training and offline metrics.
- `backend/pagewise/ranking.py`: final neural residual ranker.
- `backend/pagewise/runner.py`, `runtime.py`, `suite.py`: real query execution, bounded cache/worker and runtime summaries.
- `backend/NEURAL_IMPROVEMENT_RESULTS.md`: final new-study report.
- `backend/NEURAL_IMPROVEMENT_PLAN.md`: split/selection/freeze/confirmation protocol.
- `PRESENTATION_GUIDE.md`: concise faculty walkthrough.
- `backend/runs/local-lab/345434fbc4e3/`: the verified 100-query demo example described here; ignored local output.

## 15. Presentation wording

“Pagewise investigates whether SQLite metadata helps predict the next database page. We implemented the complete local pipeline, from real read tracing through training and runtime evaluation. The selected neural–Markov hybrid improved query-average Top-1 to 47.74% against Markov-2 at 44.83% on the held-out-parameter study. It did not make the measured queries faster, and unfamiliar query types remain challenging. This live demo independently recomputes the small development pipeline; the research page shows the saved final evaluation.”

## Interactive workspace verification (24 October 2026)

125 backend tests passed, including real generated-database SQL execution, read-only enforcement, matching model choices, row caps, artifact output and Windows snapshot publication retry. TypeScript/build passed. Browser walkthrough built small demo `ebb597a40d18` (602,112 bytes, 147 pages), browsed six tables and 25-row previews, ran SQL job `75ddfa8ae897` (162 VFS events and 46 scored Model B jobs), inspected all output tabs, downloaded artifacts and completed three-policy benchmark `703b8db14c77` (nine executions). Invalid SQL was surfaced, mobile tabs had no page overflow and no JavaScript exceptions were captured. Research SQL benchmark `a5d7bcd702aa` executed six queries with matching frozen selected checkpoint and verified unchanged database. Live check `41750d4d9f71` recorded five distinct snapshots while SQL was running. Runtime integer-overflow and cancellation paths were also verified. QA reports/screenshots are under ignored `backend/runs/workspace-browser-qa-20261024/`. The prior runtime table remains removed from the guided result component. These integration changes are included in this repository revision.

## Workspace simplification correction

The SQL/database/live-terminal workspace is directly visible at `/?view=workspace`. Simplification is restricted to the new workspace: plain action labels (**Run query**, **Compare models**) and optional parameter controls under **More options**. The terminal, explorer, traces, predictions, prefetch, model comparison and files remain visible. Original main navigation is restored. Backend behavior, saved results and models are unchanged.

Final browser verification of the restored workspace ran SQL job `8999815f0dd4` and comparison job `101819b38621`, confirmed the terminal/editor visible, optional settings collapsed, all seven output tabs available, five primary navigation items, no mobile overflow and no JavaScript exceptions. TypeScript and the production build passed after the final UI changes.
