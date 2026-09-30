"""In-context-learning arm for respondsTo ranking, after Lee et al. (EMNLP 2023).

Two things make this worth running alongside the GNNs.

1. No training. The label count that blocks the GNNs (130 positives against a
   215 power requirement) does not constrain a method with zero fitted
   parameters, so it tests the graph rather than the sample size.

2. It carries its own shortcut audit. Lee et al.'s central finding is that TKG
   forecasting accuracy holds within +/-0.4% Hit@1 when entities and relations
   are replaced by arbitrary integers -- the model exploits symbolic patterns,
   not semantics. Running both modes here turns that into a diagnostic:

     mode="lexical"  full event descriptions
     mode="numeric"  identical structure, all descriptions replaced by IDs

   If numeric matches lexical, the model is pattern-matching temporal and
   structural regularities. Given that a four-feature rule already saturates
   these splits, that is the expected outcome, and it extends the Layer-1
   admissibility gate to LLM methods -- the gates are a property of the
   candidate sets, not of any model class.

Ranking protocol is the filtered rank used by the structured baselines, so the
numbers are directly comparable.

Requires a running Ollama. NOTE: shares CPU with the extraction job.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

VIEW = Path(r"C:\Users\HP\Documents\Claude\news-tkg-modeling-view\view")
LP = Path(r"C:\Users\HP\Documents\ChatGPT\News TKG\data\link_prediction\chronological-v2")
HERE = Path(__file__).parent

SYSTEM = (
    "You rank candidate events. You are given a query event and a numbered list of "
    "candidate events that occurred earlier. Exactly one candidate is the event the "
    "query responds to. Reply with the candidate numbers ordered from most to least "
    "likely, comma-separated, most likely first. Reply with numbers only."
)


def read(p: Path) -> list[dict]:
    lines = p.read_text(encoding="utf-8").splitlines()
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:] if ln.strip()]


def d(s):
    try:
        return date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def describe(e: dict, mode: str, alias: str) -> str:
    if mode == "numeric":
        # keep only structure: type id, commodity id, location id, date
        return (f"type={e.get('_subtype_id','?')} commodity={e.get('_com_id','?')} "
                f"place={e.get('_loc_id','?')} date={e.get('event_date','?')}")
    return (f"{e.get('event_subtype','?')} | {e.get('commodity_id','?')} | "
            f"{e.get('location_id','?')} | {e.get('event_date','?')}")


def ollama(url: str, model: str, prompt: str, timeout: int) -> str:
    payload = {"model": model, "stream": False,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt}],
               "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 200},
               "think": False, "keep_alive": "30m"}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())["message"]["content"]


def parse_ranking(txt: str, n: int) -> list[int]:
    nums = [int(x) for x in re.findall(r"\d+", txt or "")]
    seen, out = set(), []
    for x in nums:
        if 1 <= x <= n and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lexical", "numeric", "both"], default="both")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-candidates", type=int, default=20,
                    help="candidates shown per query; the positive is always included")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/chat")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    events = {r["event_id"]: r for r in read(VIEW / "nodes_event.tsv")}
    # stable symbolic ids for the numeric mode
    subs = sorted({r.get("event_subtype", "") for r in events.values()})
    coms = sorted({r.get("commodity_id", "") for r in events.values()})
    locs = sorted({r.get("location_id", "") for r in events.values()})
    for r in events.values():
        r["_subtype_id"] = f"T{subs.index(r.get('event_subtype',''))}"
        r["_com_id"] = f"C{coms.index(r.get('commodity_id',''))}"
        r["_loc_id"] = f"L{locs.index(r.get('location_id',''))}"

    cand_file = LP / "resolved_responds_to" / f"{args.split}_ranking_candidates.tsv"
    if not cand_file.exists():
        raise SystemExit(f"missing {cand_file}")
    rows = read(cand_file)
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["query_response_id"]].append(r)

    modes = ["lexical", "numeric"] if args.mode == "both" else [args.mode]
    results = {m: {"ranks": [], "queries": [], "failures": 0} for m in modes}
    queries = sorted(by_q)
    if args.limit:
        queries = queries[: args.limit]

    for qi, qid in enumerate(queries, 1):
        cands = by_q[qid]
        pos = [c for c in cands if c["known_positive"] == "true"]
        negs = [c for c in cands if c["known_positive"] != "true"]
        if not pos:
            continue
        # keep the hardest negatives, so the LLM faces the same difficulty the
        # structured baselines do rather than an easier random sample
        negs.sort(key=lambda r: -float(r["hardness_score"]))
        shown = pos[:1] + negs[: max(0, args.max_candidates - 1)]
        # deterministic shuffle by id so position carries no signal
        shown.sort(key=lambda r: r["candidate_disruption_id"])
        target_ix = 1 + shown.index(pos[0])

        qev = events.get(qid, {})
        for mode in modes:
            lines = [f"Query event (a government response): {describe(qev, mode, 'Q')}",
                     f"Query date: {qev.get('event_date','?')}", "",
                     "Candidate earlier events:"]
            for i, c in enumerate(shown, 1):
                ce = events.get(c["candidate_disruption_id"], {})
                lines.append(f"{i}. {describe(ce, mode, str(i))}")
            lines.append("")
            lines.append("Ranking (most likely first, numbers only):")
            prompt = "\n".join(lines)

            t0 = time.time()
            try:
                txt = ollama(args.ollama_url, args.model, prompt, args.timeout)
            except Exception as exc:
                results[mode]["failures"] += 1
                print(f"[{qi}/{len(queries)}] {mode:8s} FAILED {exc!r}")
                continue
            order = parse_ranking(txt, len(shown))
            rank = (order.index(target_ix) + 1) if target_ix in order else len(shown)
            results[mode]["ranks"].append(rank)
            results[mode]["queries"].append(
                {"query": qid, "rank": rank, "n_candidates": len(shown),
                 "seconds": round(time.time() - t0, 1), "raw": (txt or "")[:120]})
            print(f"[{qi}/{len(queries)}] {mode:8s} rank {rank}/{len(shown)}  "
                  f"{time.time()-t0:.0f}s")

    out = {"model": args.model, "split": args.split,
           "max_candidates": args.max_candidates, "modes": {}}
    for m in modes:
        rk = results[m]["ranks"]
        if not rk:
            out["modes"][m] = {"queries": 0}
            continue
        out["modes"][m] = {
            "queries": len(rk),
            "filtered_mrr": round(sum(1 / r for r in rk) / len(rk), 4),
            "hits_at_1": round(sum(r <= 1 for r in rk) / len(rk), 4),
            "hits_at_3": round(sum(r <= 3 for r in rk) / len(rk), 4),
            "mean_rank": round(sum(rk) / len(rk), 2),
            "failures": results[m]["failures"],
            "per_query": results[m]["queries"],
        }
    if len(modes) == 2 and all(out["modes"][m].get("queries") for m in modes):
        a, b = out["modes"]["lexical"]["filtered_mrr"], out["modes"]["numeric"]["filtered_mrr"]
        # A raw threshold on the MRR gap is the wrong test: it ignores both the
        # sign and the sampling variance. Decide with the paired permutation test
        # over per-query reciprocal ranks instead.
        try:
            from paired_stats import compare
            lex = [1 / q["rank"] for q in out["modes"]["lexical"]["per_query"]]
            num = [1 / q["rank"] for q in out["modes"]["numeric"]["per_query"]]
            st = compare(lex, num, "lexical", "numeric")
            sig = st["significant_at_alpha"]
            if not sig:
                interp = ("no detectable difference: stripping all semantics does not "
                          "measurably change ranking, so the model is exploiting symbolic "
                          "and temporal pattern (Lee et al. 2023). Admissibility is a "
                          "property of the candidate set, not of the model class.")
            elif b > a:
                interp = "removing semantics IMPROVED ranking -- inspect for prompt or parsing artefacts"
            else:
                interp = "semantics contributes; the result is not purely structural"
            out["semantic_ablation"] = {
                "lexical_mrr": a, "numeric_mrr": b, "delta": round(a - b, 4),
                "paired_test": st, "interpretation": interp,
            }
        except Exception as exc:                      # keep the raw numbers if the test fails
            out["semantic_ablation"] = {"lexical_mrr": a, "numeric_mrr": b,
                                        "delta": round(a - b, 4), "paired_test_error": repr(exc)}
    (HERE / f"icl-{args.split}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    for m in modes:
        o = out["modes"][m]
        if o.get("queries"):
            print(f"  {m:8s} q={o['queries']:3d}  MRR {o['filtered_mrr']:.3f}  "
                  f"H@1 {o['hits_at_1']:.3f}  mean rank {o['mean_rank']}")
    if "semantic_ablation" in out:
        s = out["semantic_ablation"]
        print(f"\n  semantic ablation: lexical {s['lexical_mrr']} vs numeric {s['numeric_mrr']} "
              f"(delta {s['delta']})\n  -> {s['interpretation']}")


if __name__ == "__main__":
    main()
