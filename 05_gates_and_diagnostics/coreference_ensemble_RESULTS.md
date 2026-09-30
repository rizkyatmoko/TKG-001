# Event-coreference pool completion — full report

2026-08-15. Read-only against the ChatGPT repo; everything written here lives in
the Claude folder. Nothing was materialised into the RDF.

> **SUPERSEDED IN PART — read §Ensemble first.** Finding 1 below reports 9
> merges and 3 new canonical events from the single `strict_evidence` judge.
> Running the two other prompt variants collapses that to **1 merge and 1 new
> canonical event**. The single-judge number was a prompt artifact. The rest of
> the report stands.

## What was done

The shipped graph judged 47 mention→canonical links (42 accepted, 5 rejected
pairs). This work regenerated the candidate pool from all 359 event mentions
under two blocking rules, judged every undecided pair the repo's own rule
admits, and sampled the pairs only a looser rule admits.

**176 pairs judged**, by Claude (Opus 5) using the repo's own `event_identity` /
`mention_pair` schema and task prompt, prompt variant `strict_evidence`.
**Run 1 of an intended 3.** One model, one variant — not an ensemble, and not
statistically independent of any other Claude run.

| Tier | What it is | Pairs |
|---|---|---:|
| A | census of every undecided pair under the repo's documented blocking rule | 96 |
| B | random sample of pairs only the *relaxed* rule admits (from 840) | 80 |

## Corrections to earlier numbers

- The strict pool is **140 pairs, not 172**. The earlier count matched on RDF
  types, which include inherited superclasses (`ResponseEvent`,
  `PolicyImplementationEvent`), inflating subtype matches. The source
  annotation's single `event_subtype` gives 140.
- Prior coverage was **44/140 (31%)**, not 47/172 (27%).
- One already-rejected pair falls **outside** the documented blocking rule, so
  that rule is not exactly what generated the original 47.

## Finding 1 — recall was being lost, and it was worth ~19%

**9 pairs merge at p(same) ≥ 0.50, forming 3 new canonical events over 9
mentions.**

| New cluster | Mentions | What it is |
|---|---:|---|
| cooking-oil scarcity, early 2022 | **5** | one sustained national shortage reported from Ponorogo, Surabaya, Malang, Sumenep and Jakarta, Feb–Apr 2022 |
| refined-sugar scarcity, East Java 2021 | 2 | shortage hitting food and beverage SMEs |
| F&B business closures, East Java 2021 | 2 | the production-side consequence of that shortage |

Against the shipped 16 canonical events over 42 mentions, that is a **19%
increase in canonical events and 21% more clustered mentions**. The five-mention
cooking-oil cluster is now the **largest in the graph** (previous maximum: 4).

All three are **episode** merges — sustained commodity-shortage conditions
reported repeatedly across publishers. The original pass captured **incidents**
(one earthquake, one court case) and systematically missed episodes. That is a
nameable, correctable bias, not random error.

## Finding 2 — the blocking rule is fine; the judging was the bottleneck

This is what tier B was for, and the answer is unusually clean.

| | Tier A (strict rule) | Tier B (relaxed only) |
|---|---:|---:|
| pairs judged | 96 | 80 |
| merges at ≥0.50 | **9** | **0** |
| ambiguous (0.25–0.50) | 15 | 1 |
| highest p(same) seen | 0.70 | 0.25 |
| median normalized entropy | **0.41** | **0.21** |
| share with entropy > 0.5 | 35.4% | 3.7% |

The relaxed rule admits 979 pairs against the strict rule's 140 — a sevenfold
larger pool — and an 80-pair sample of the 840 extra pairs yields **zero**
merges. By the rule of three, the 95% upper bound on merges hiding in that whole
pool is about 32, and the point estimate is 0.

So the strict rule was **not** costing recall. All of the loss came from judging
only 31% of what that rule already admitted. This is a cheap and reassuring
result: completing the census is sufficient, and widening the blocking rule
would cost 7× the judging budget for close to nothing.

It also means blocking here is doing more than filtering for efficiency — it is
**selecting where label uncertainty lives**. Tier A carries nearly twice the
entropy of tier B and 10× the share of high-entropy pairs.

## Finding 3 — the soft labels exist; they were never sampled

Across all 176 judgments:

| p(same_event) band | pairs |
|---|---:|
| confident same (≥0.85) | 0 |
| leaning same (0.50–0.85) | 9 |
| **ambiguous (0.25–0.50)** | **16** |
| leaning distinct (0.10–0.25) | 20 |
| confident distinct (<0.10) | 131 |

The shipped graph's `sameEventProbability` values are {1.0, 0.99, 0.98, 0.97}
for accepted and {0.09, 0.04, 0.01, 0.01} for rejected — **zero values between
0.10 and 0.95**. That bimodality was never a property of the problem. It was a
property of *which pairs were looked at*: the original pass only judged pairs it
was already confident about, so it never sampled the middle of the distribution.

Notably, **no pair in the completed census reached 0.85**. The highest is 0.70.
The shipped graph's 42 accepted links all sit at 0.97–1.0. Those are not the
same population, and pooling them would misrepresent both.

The 16 ambiguous pairs form one coherent category — **episode versus instance**:

- a Blitar delivery to 8,069 households vs. the national programme to 21.35m;
- a Lumajang FMD count vs. province-wide and national counts;
- Banyuwangi chili damage vs. Tuban/Kediri/Blitar damage in one season;
- MinyaKita scarcity in Blitar City vs. across eight provinces.

Each is a real modelling decision about event granularity, not an extraction
error. A hard accept/reject destroys the information; a soft label keeps it.

## What this means for the GNN plan

1. **The uncertainty-aware direction now has direct evidence.** It was previously
   inferred from extractor disagreement (event Jaccard 0.417 between passes).
   It is now demonstrated on the resolution layer itself, from a full census
   rather than a convenience sample, and the ambiguity concentrates in one
   nameable phenomenon.
2. **The cross-document track gains material.** 3 new canonical events over 9
   mentions make new cross-document `respondsTo` entailments derivable — the
   only part of the benchmark carrying real report-time structure.
3. **Blocking should be reported as part of the method.** Tier B shows the rule
   is well calibrated; that is a defensible design choice to state, not an
   assumption to leave implicit.

## Ensemble — runs 2 and 3, and what they overturn

Run 2 `counterfactual` and run 3 `temporal_scope` were added over the 25
contested pairs (run-1 p(same) ≥ 0.25), plus a 20-pair audit of pairs run 1
placed below 0.25. Aggregation is the repo's own rule: unweighted mean of the
categorical distributions.

**Merges at p ≥ 0.50: 9 under the single judge → 1 under the ensemble.**

| pair | strict | counterfactual | temporal_scope | ensemble | entropy |
|---|---:|---:|---:|---:|---:|
| R062 sugar scarcity, EJ 2021 | 0.70 | 0.62 | 0.38 | **0.57** | 0.81 |
| R042 cooking oil, Surabaya×2 | 0.60 | 0.50 | 0.32 | 0.47 | 0.85 |
| R063 F&B closures | 0.60 | 0.48 | 0.30 | 0.46 | 0.85 |
| R030 / R041 cooking oil | 0.55 | 0.45 | 0.28 | 0.43 | 0.85 |
| R002 / R003 cooking oil | 0.60 | 0.42 | 0.25 | 0.42 | 0.85 |
| R031 / R032 cooking oil | 0.55 | 0.40 | 0.25 | 0.40 | 0.85 |

The spread is **systematic and directional**, never noisy: on all 45 re-judged
pairs `strict_evidence` > `counterfactual` > `temporal_scope`, median spread
0.15, max 0.35. The reason is plain in the prompt — `temporal_scope` instructs
*"Do not collapse related episodes into one event"*, and every one of the nine
single-judge merges was an episode merge.

**So the episode merges are prompt-determined, not evidence-determined.** The
one survivor, R062, is the only pair with explicit textual evidence of identity:
the second article refers back to the first complaint rather than merely
describing the same condition. The ensemble separated the evidence-backed merge
from the eight inference-backed ones, which is exactly what it is for.

Revised effect on the graph: **1 new canonical event over 2 mentions**
(16 → 17, +6%), not 3 over 9 (+19%).

The triage held: across the 20 audited pairs below 0.25, maximum ensemble
p(same) was 0.10 and none would merge.

### The uncomfortable implication

Ensemble entropy on the contested pairs is 0.78–0.85 — very high — but it is
driven by **prompt convention rather than evidential ambiguity**. Whether a
five-week national shortage is one event or five is a modelling decision the
prompt makes, not a fact the articles settle.

That matters for the uncertainty-aware GNN direction. Soft labels built this way
partly encode *which prompt variants were written*. Before training on them, the
episode-versus-instance convention has to be fixed in the ontology and stated,
so the remaining entropy reflects genuine evidential uncertainty. Otherwise a
model trained on these targets learns the annotation protocol.

The cheap fix is the same one flagged from the extractor-agreement work: decide
the granularity rule explicitly. It is currently the single largest source of
disagreement at both the extraction and resolution layers.

## Limits

- **Runs 2 and 3 cover 45 of 176 pairs, not all of them.** Pairs run 1 placed
  below 0.25 were triaged out; the 20-pair audit supports that choice but does
  not prove it.
- **All three runs are the same model** with different prompt families. They are
  prompt-decorrelated, not independent, and the systematic ordering between
  variants shows how far from independent they are. A second model family
  remains necessary.
- **One judge, one prompt variant, and the strictest of the three.**
  `strict_evidence` explicitly refuses credit for plausibility or co-occurrence,
  so these merges are a **lower bound** — `counterfactual` and `temporal_scope`
  would likely merge more, especially in the 0.25–0.50 band. Runs 2 and 3 should
  come from a different model family, or the ensemble reproduces the
  correlated-judge problem already flagged.
- My probabilities are subjective estimates, not calibrated frequencies. They
  are ensemble input, not ground truth.
- Tier B is an 80-pair sample, not a census. Zero merges is strong evidence, not
  proof.
- 45 pairs were inherited as already-decided and were not re-judged, so the
  shipped decisions are untouched and unverified by this run.
- Nothing has been written into the RDF.

## Files

| File | Contents |
|---|---|
| `build_coref_pool.py` | pool regeneration, both blocking rules, tiering |
| `coref-judging-packets.jsonl` | 176 packets in the repo's schema |
| `coref-pairs-readable.txt` | the same pairs, human-readable |
| `judgments-claude-strict/batch0{1..5}.json` | 176 judgments |
| `aggregate_judgments.py` | analysis |
| `coref-judgment-report.json` | per-pair probabilities, entropy, clusters |
| `by-tier-summary.json` | the tier A/B comparison |
| `coref-pool-report.json` | pool sizes and blocking statistics |
