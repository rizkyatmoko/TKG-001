# Extractor calibration harness — build + first result

2026-08-15. Everything here is in the Claude folder; the ChatGPT repo was read
only. Nothing was written to it and nothing in it was modified by this work.

## What the harness is

A backend-agnostic way to score *any* extractor against the existing pilot
silver labels, so different models produce comparable numbers.

| File | Role |
|---|---|
| `build_pilot_packets.py` | deterministic stratified sample of pilot articles → blind packets |
| `score_run.py` | scores any run against the pilot reference |
| `run_ollama_extractor.py` | optional Qwen/Ollama backend |
| `checkpoints-claude/batch0*.json` | the Claude extraction pass, by batch |
| `run-claude.json`, `score-claude-opus-5.json` | merged run + its score |
| `packets-readable.txt` | the 40 packets in human-readable form |

Two things make the numbers trustworthy rather than decorative:

1. **The packets are the live pipeline's own format.** Sentence segmentation is
   imported from the repo's `llm_confidence_common.sentences`, not
   reimplemented, and `run_ollama_extractor.py` imports `SYSTEM_PROMPT_V3` and
   `response_schema_v3()` from `run_expansion_extraction_v3.py`. Same prompt,
   same schema, same `temperature=0` / `num_ctx=8192`. Anything else would
   measure a different prompt.
2. **The scorer is self-tested.** Scoring the reference against itself returns
   1.0 on every metric (`score-identity-selftest.json`). A harness that cannot
   recognise a perfect run cannot be trusted on an imperfect one.

Sample: 40 articles, 1,057 sentences, stratified to exercise the failure modes
that matter — 10 irrelevant articles (false positives), 2 multi-event
(over-fragmentation), 16 relation-bearing (predicate choice).
Reference content: 50 events, 24 relations.

## Result: Claude as second extractor

```
articles scored    : 40
relevance kappa    : 0.559  (raw agreement 0.80)
events run/ref     : 43/50   matched 28
event Jaccard      : 0.431   [frontier pair = 0.417]
subtype agreement  : 0.893   [frontier pair = 0.71]
relations run/ref  : 18/24   J=0.448  [frontier pair = 0.407]
false-positive arts: 2   over-fragmented: 3   under-extracted: 10
confidence         : 29 distinct values, not constant
```

Read against the two existing frontier passes over all 300 articles
(event J 0.417, subtype 0.71, relation J 0.407), this run is **in-family**:
same event and relation overlap, noticeably tighter subtype agreement. That is
the yardstick. A backend scoring near 0.43 is behaving like the corpus's
existing annotators; one scoring near 0.12 is not.

For reference, the earlier ad-hoc measurement of the Qwen runs against pilot
articles gave event J **0.125** (qwen8b, n=3) and **0.111** (qwen4b, n=4), with
0/1 subtype agreement each and constant confidence. Those came from a different
prompt and a 3–4 article sample, so they are **not** directly comparable to the
0.431 above — that is exactly what this harness now fixes. Run
`run_ollama_extractor.py` and the two numbers become comparable.

## What the disagreements actually are

Not noise. One axis dominates.

**The East Java scope gate.** Six of the eight relevance disagreements are
articles I called irrelevant and the reference called relevant — all national
stories whose only East Java link is a venue or a list mention: national sugar
production projected at a Surabaya meeting, a national sugar-absorption scheme
agreed in Surabaya, Bulog's rice-export discourse listing East Java among MRMP
sites, Jokowi asking governors to watch rice stock, a Bapanas regulation being
harmonised, a freight-traffic restriction that exempts sembako. I applied
"venue is not operational connection" strictly; the reference did not. Each of
those articles carries 1–2 reference events, so the rule choice alone accounts
for most of the 22 events I did not emit.

Two go the other way — I extracted where the reference did not: a flood
coordination meeting in Tuban (food link is one mention of logistics
distribution) and a CSR award naming a running Magetan farmer programme.

**Subtype and predicate choice are nearly settled.** Only 3 subtype
disagreements in 28 matched events (`OtherResponseAction`→`MarketOperationAction`,
`FoodSafetyIncident`→`InspectionEnforcement`,
`DistributionDisruption`→`SupplyShortage`), and predicate agreement on matched
relations is perfect (10 `respondsTo`, 3 `reportedCauseOf`, no crossings).

This sharpens the earlier recommendation about the stage-3 ballot. Between two
*strong* extractors the unstable axis is **article-level scope**, not
event typing — so the judging ballot needs a scope question ("does this article
report an East Java food-system occurrence?") at least as much as it needs
support/contradiction/insufficient on individual claims. A per-claim ballot
cannot recover an article that one extractor never opened.

## Honest limits

- 40 articles, 50 reference events. Wide error bars; treat gaps under ~0.05 as
  indistinguishable.
- The reference is LLM-only silver, not gold. This measures *agreement with the
  corpus's existing convention*, not correctness. A backend could disagree with
  the reference by being right.
- One Claude pass, no self-consistency check, so the run-to-run variance of this
  backend is unmeasured.
- The scope-rule divergence above is a genuine ambiguity in the prompt, not a
  defect in either party. Pinning it down would raise agreement for every
  backend at once — probably the cheapest single improvement available.

## To add the Qwen row

After the expansion job finishes (it shares the GPU; running this during the job
will slow it):

```
python run_ollama_extractor.py --model qwen3:8b
python score_run.py --run run-qwen3-8b.json --label qwen3-8b
```

Per-article checkpoints mean it resumes without repeating work.
