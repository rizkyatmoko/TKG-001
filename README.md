# Reviewer evidence: Admissibility Gates for R-GCN Evaluation on News-Derived TKGs

Every number in the manuscript (JISEBI version, `09_manuscript/`) maps to a file below.
Nothing here was re-run for this bundle: result files are copies of what the experiments
wrote, and `07_session_computations/` holds computations that were only ever printed in a
working session, extracted verbatim (script and output) from the session transcript.

News article bodies are **not** included: they are copyrighted by their publishers. Article
IDs, titles, URLs and dates are in `01_data_and_sampling/*_no_body.csv`.

| Section | Claim | Value | Evidence |
|---|---|---|---|
| III.A | Articles collected | 1,504,980 | `01_data_and_sampling/dataset_profile.json` |
| III.A | Scope filter / exact duplicates / pool | 30,836 / 149 / 30,687 | `01_data_and_sampling/pilot-sampling-report.json` |
| III.A | Pilot strata | 40, 40, 60, 50, 40, 40, 30 | `01_data_and_sampling/pilot-sampling-report.json` |
| III.A | Pilot articles (IDs, titles, URLs; no text) | 300 | `01_data_and_sampling/pilot_articles_no_body.csv` |
| III.A | Silver labels, human_verified false | pilot annotations | `02_annotation_and_extraction/llm-silver-pilot-v2.json` |
| III.A | Evidence-substring and date checks | V1-V4 | `02_annotation_and_extraction/ingest_chat_output.py` |
| III.B | Density selection vs commodity filter | 129 vs 68 cells >=4; 32 vs 8 cells >=8 | `01_data_and_sampling/dense-subcorpus-report.json` |
| III.B | Queue; qwen completed; density-selected among them | 1,952; 655; 433 of 443 | `07_session_computations/expansion_routes_id_overlap_433.txt` |
| III.B | Expansion queue (IDs, metadata; no text) | 1,952 | `01_data_and_sampling/expansion_queue_1952_no_body.csv` |
| III.B | Events per article by route; total events; 874 'no East Java link' | 1.20 / 0.63 / 0.26; 1,107; 874 | `07_session_computations/events_per_route_and_874.txt` |
| III.B | Chatbot calibration vs frontier pair | event Jaccard 0.391 vs 0.417 | `02_annotation_and_extraction/score-chatbot-calibration.json` |
| III.C / Table 1 | Pilot graph contents | 37,821 triples ... 87 policies | `03_ontology_and_graph/llm-silver-tkg-v2-report.json` |
| III.C / Table 1 | SHACL / core validation | zero core issues | `03_ontology_and_graph/llm-silver-tkg-v2-validation.json` |
| III.C / Table 1 | respondsTo 130, reportedCauseOf 29 | 130 / 29 | `05_gates_and_diagnostics/relation-classification-guarded.json` |
| III.C / Table 1 | implementsPolicy 52 | 52 | `07_session_computations/implementsPolicy_lags_and_targets.txt` |
| III.C | Ontology size | 46 classes, 48 object properties | `03_ontology_and_graph/foodjatim-tkg.ttl` |
| III.D | Expanded modeling view | 3,398 nodes, 10 relations, 6,489 edges | `03_ontology_and_graph/modeling_view_expanded` |
| III.E | R-GCN size | 71 features, 9 relations, 33,056 parameters | `06_cross_document_task/xdoc/eval-canon.json` |
| III.E | Gradient check; synthetic propagation task | 4.9e-8; 1.00 vs 0.80 | `04_model_and_statistics/rgcn-selftest.json` |
| III.F | Pilot two-hop neighborhoods | 2,266 / 1,277 / 1,199 | `07_session_computations/pilot_report_time_masking.txt` |
| III.F | Expanded neighborhoods | 405 / 207 / 200; 81% | `04_model_and_statistics/temporal-masking-report.json` |
| III.G | A4 attribute sharing | 87% / 51% / 26%; 21.5% two-hop | `07_session_computations/attribute_sharing_A4.txt` |
| III.H / IV.A | In-context arm | 0.743 / 0.856; diff -0.113; p = 0.50 | `05_gates_and_diagnostics/icl-semantic-ablation-test.json` |
| III.I | Significance self-tests | p = 1; exact = brute force; 6.5% at n = 25 | `04_model_and_statistics/paired-stats-selftest.json` |
| III.J | L0 lookup at 955 articles | 93% of 222 | `07_session_computations/L0_lookup_at_955_articles.txt` |
| III.J | L0 lookup, full corpus | 91% of 335 | `05_gates_and_diagnostics/viability-current.json` |
| III.J | L0 event-to-event labels | 159 of 159 | `07_session_computations/L0_event_to_event_159.txt` |
| III.J | Relation classification | 1.000 on 25 items; majority 0.92 | `05_gates_and_diagnostics/relation-classification-guarded.json` |
| III.J / Table 4 | Power table | 197 / 65 / 32 / 13 | `04_model_and_statistics/power-analysis.json` |
| III.J | Observed power | 0.69 vs 0.71 at n = 25, dz = 0.5 | `07_session_computations/observed_power_n25.txt` |
| III.K | Coreference | 510 mentions -> 462 incidents | `05_gates_and_diagnostics/coref-with-chat.json` |
| III.K | Judging | 278 groups, 802 pairs, 65 positives, 2,035 checks, kappa 0.800 | `06_cross_document_task/xdoc/RESULTS.md` |
| III.K | The 65 positives | xdoc-edges.json | `06_cross_document_task/xdoc/xdoc-edges.json` |
| III.K | Judge prompts and judgments | 24 batches | `06_cross_document_task/xdoc/judgments` |
| IV.A | Pilot test splits; equal-hardness baselines; density | 100%; 0.163 vs 0.333; 141 events, 8.6x/23.1x | `05_gates_and_diagnostics/pilot_split_and_density_RESULTS.md` |
| IV.A | Co-reporting | 87 of 130; 63 of 84; median gap 0 | `07_session_computations/co_reporting.txt` |
| IV.A | implementsPolicy lags and concentration | median 896; 14 of 87; 22 of 52 | `07_session_computations/implementsPolicy_lags_and_targets.txt` |
| IV.A | Deterministic coreference | 4 clusters of 197; median lead 6 days | `05_gates_and_diagnostics/deterministic-coref-w1.json` |
| IV.A | Coreference ensemble | 176 pairs; 45 re-judged; entropy 0.78-0.85 | `05_gates_and_diagnostics/coreference_ensemble_RESULTS.md` |
| IV.B / Table 5 | L1 over reports vs incidents | 29% / 3 vs 52% / 0 | `06_cross_document_task/xdoc/l1-incidents.json` |
| IV.B | L1 robustness to incident representation | V0-V4 | `06_cross_document_task/xdoc/l1-unit-sensitivity.json` |
| IV.B | L1 against the blocking rule | 72% strictly first, median 0 | `07_session_computations/L1_against_blocking_rule_72pct.txt` |
| IV.C / Table 6 | Ranking over incidents | district 0.921; R-GCN 0.259; dz -1.75 | `06_cross_document_task/xdoc/eval-canon.json` |
| IV.C / Table 6 | R-GCN + intra-article training | 0.252; 52 -> 120 positives per fold | `06_cross_document_task/xdoc/eval-canon-intra.json` |
| IV.C / Table 6 | Masking disabled | 0.361 | `06_cross_document_task/xdoc/eval-canon-leaktest.json` |
| IV.C / Table 6 | First protocol (reports), reproduced exactly | 0.717 / 0.259; 19 sets, 142 negatives | `06_cross_document_task/xdoc/eval-raw.json` |
| IV.C | Structure of the positives | all same district <=180 d; median lag 2; 60 of 65; 3.5% | `07_session_computations/positives_structure_60_of_65.txt` |
| IV.D / Table 7 | Task verdicts | L0 / L1 / L2 per task | `05_gates_and_diagnostics/viability-current.json` |
| References | Crossref / DataCite / arXiv records | 47 references | `08_reference_verification/reference_verification.csv` |

## Folders

- `01_data_and_sampling/` corpus profile, sampling report, density selection, article lists (no text)
- `02_annotation_and_extraction/` silver labels: pilot, 655 qwen3:8b checkpoints, 1,280 chatbot checkpoints; prompt, validators, calibration
- `03_ontology_and_graph/` ontology, materialized-graph reports, SHACL validation, expanded modeling view
- `04_model_and_statistics/` R-GCN implementation and self-test, paired statistics and self-test, power analysis, report-time masking
- `05_gates_and_diagnostics/` gate measurements, relation classification, in-context arm, coreference
- `06_cross_document_task/xdoc/` candidate generation, judge prompts, judgments, L1 measurements, evaluation script and all result files (including the first, report-level protocol)
- `07_session_computations/` verbatim extracts of session-only computations
- `08_reference_verification/` raw Crossref, DataCite and arXiv records for the reference list
- `09_manuscript/` the submitted manuscript and its build sources

`MANIFEST_SHA256.txt` fingerprints every file in this folder.
