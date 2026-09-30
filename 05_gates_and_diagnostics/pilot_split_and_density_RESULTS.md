# Option (b): shortcut-stratified evaluation — result

2026-08-15. Read-only against the ChatGPT repo. `per_tier_eval.py`,
`per-tier-eval.json` reproduce everything below. No GPU, no LLM.

## The circularity

`run_link_prediction_baselines_v2.py` defines its best-performing baseline as

```python
"structured_compatibility": lambda row: float(row["hardness_score"])
```

and `build_chronological_link_prediction_v2.py` writes that column as

```python
hardness = 4.0*same_commodity + 2.0*commodity_compatible \
         + 1.5*same_location + 0.5*same_food_system_stage + recency
```

**The baseline that reaches filtered MRR 1.000 is the candidate-construction
function ranking its own candidates.** "The benchmark is saturated by a
non-learned structured rule" and "the candidate universe was built by that rule"
are the same sentence.

## The diagnostic that replaces pooled MRR

For each positive, ask: how many negatives in its own candidate set are at least
as hard as it is?

| Task | Split | positives | positive is *strictly* the hardest | positives with **zero** negatives at their hardness | structured MRR |
|---|---|---:|---:|---:|---:|
| cross-document respondsTo | validation | 15 | 20% | 20% | 0.301 |
| cross-document respondsTo | **test** | 10 | **100%** | **100%** | **1.000** |
| resolved respondsTo | validation | 14 | 64% | 57% | 0.749 |
| resolved respondsTo | **test** | 10 | **100%** | **100%** | **1.000** |
| resolved implementsPolicy | validation | 6 | 17% | 17% | 0.287 |
| resolved implementsPolicy | test | 4 | 0% | 0% | 0.049 |

The share of positives that are strictly the hardest candidate tracks the
headline MRR almost exactly: 20%→0.30, 64%→0.75, 100%→1.00, 100%→1.00.

**Filtered MRR on this benchmark is a restatement of how often the positive
happens to be the maximum-hardness candidate.** That is a property of candidate
construction, not of any model.

## What survives when the easy negatives are removed

Restricting each query to negatives at or above its positive's hardness:

| Task | Split | condition | random | recency | structured | same_commodity | same_location |
|---|---|---|---:|---:|---:|---:|---:|
| cross-doc respondsTo | val | all candidates | 0.017 | 0.016 | 0.301 | 0.052 | 0.245 |
| cross-doc respondsTo | val | **at positive hardness** (2 q) | **0.333** | 0.163 | **0.163** | 0.267 | 0.489 |
| resolved respondsTo | val | all candidates | 0.056 | 0.674 | 0.749 | 0.240 | 0.493 |
| resolved respondsTo | val | **at positive hardness** (6 q) | **0.589** | 0.499 | **0.415** | 0.721 | 0.771 |
| cross-doc respondsTo | **test** | at positive hardness | — | — | — | — | — |
| resolved respondsTo | **test** | at positive hardness | — | — | — | — | — |

Two things to read off this:

1. **Both test splits are not evaluable at all under the hard-negative
   condition.** Not one query, out of 16, retains a single negative as hard as
   its positive. There are no hard negatives in the test sets — none.
2. Where the condition *is* evaluable, `structured_compatibility` falls
   **below deterministic random** (0.163 vs 0.333; 0.415 vs 0.589). Its apparent
   strength is entirely carried by easy negatives.

Truncating to the top 50% or top 25% hardest negatives changes almost nothing
(structured stays at 0.301 / 1.000 / 0.749 / 1.000), because it never removes
the decisive gap — the positive still tops every list.

There is a hint of residual real signal: at equal hardness, `same_location_only`
reaches 0.489 and 0.771, above random. But that rests on 2 and 6 queries.

## Consequence for the plan

The problem is **not** that a clever shortcut was discovered. It is that the
candidate universe contains no negative that competes with the positive, so
every ranking metric collapses to the construction rule. No architecture,
soft-labelled or otherwise, can be compared on this benchmark.

The fix cannot come from re-weighting or re-tiering the existing candidates —
`hard_only_top25pct` demonstrates that. It has to come from generating negatives
that **tie or exceed** the positive on commodity, location, stage and recency:
temporally valid disruptions matching the response on every structured feature
that simply are not the one responded to. Expansion-v3's 4,800 candidates carry
`structured_match_count` and `hardness_tier`, so they may do better — but their
rows are still article-level and marked `pending_event_extraction`.

## Hard-negative generation: attempted, and it fails on density

`build_hard_negatives.py` mines matched negatives directly — same commodity
mandatory, temporal eligibility enforced, optional lag window, and an optional
**normalized location** match (district tokens, so "Malang, East Java" and
"Malang and Lumajang, East Java" count as the same district; raw string equality
partly detects article identity, since positives are often intra-article and
share byte-identical location strings).

Four configurations, both gates, `resolved_respondsTo`:

| location | lag window | split | share strictly hardest | median negatives ≥ positive | queries with ≥1 | ≥5 | ≥10 |
|---|---|---|---:|---:|---:|---:|---:|
| exact | none | validation | 57% | 0 | 6/14 | 1 | 0 |
| exact | none | **test** | **100%** | **0** | **0/10** | 0 | 0 |
| exact | 3× positive | test | 100% | 0 | 0/10 | 0 | 0 |
| normalized | none | validation | 64% | 0 | 5/14 | 0 | 0 |
| normalized | none | **test** | 90% | **0** | 1/10 | 0 | 0 |

**Every configuration fails both gates.** Normalizing location helps the test
split slightly (100% → 90% strictly hardest) and helps validation not at all.

## Why: there is no pool to mine from

| | |
|---|---|
| disruption events in the whole graph | **141** |
| date span | 7.4 years |
| commodities | 13 |

Same-commodity disruptions within ±90 days of a positive: **median 5**
(validation), **median 2** (test).

Per-commodity density, events per 90-day window across the entire corpus:

| commodity | events | per 90d | ×corpus needed for median 10 |
|---|---:|---:|---:|
| rice | 35 | 1.17 | **8.6×** |
| broad_food | 22 | 0.73 | 13.6× |
| chili | 18 | 0.60 | 16.7× |
| beef | 13 | 0.43 | 23.1× |
| cooking_oil | 12 | 0.40 | 25× |

The benchmark is not repairable by better mining. There is no negative to mine:
at ~1 disruption per commodity per quarter, nothing competes with a positive that
matches on commodity, location, stage **and** sits at a 0–4 day lag.

## What this means for the expansion

The expansion is 1,952 articles against the pilot's 300 — **6.5×**. Set against
the table above, that is the right order of magnitude but lands sufficiency on
**rice alone**, and only if per-article event yield holds. Every other commodity
stays short by 2–4×.

The actionable consequence: **scope the benchmark to rice** (possibly plus
`broad_food`) rather than all 13 commodities. Rice is 25% of disruptions, the
densest commodity, and the most policy-relevant one. A defensible single-commodity
benchmark is reachable with the corpus that will exist on 18 August. A
thirteen-commodity one is not, at any plausible extraction quality.

Re-run `build_hard_negatives.py` against the expanded graph before committing to
that scope — it is the cheapest possible go/no-go, and it needs no GPU.

## Two release gates worth adopting

Cheap, deterministic, and they would have caught this at v1:

1. **`share_of_positives_that_are_strictly_hardest` must be well below 1.0.**
   At 1.0 the split is unusable regardless of what any model scores.
2. **Median negatives at or above positive hardness must be ≥ 1**, i.e. the
   hard-negative condition must be evaluable at all. Both current test splits
   fail this at zero.

Report both alongside MRR. A benchmark that cannot state them has not been shown
to measure anything.
