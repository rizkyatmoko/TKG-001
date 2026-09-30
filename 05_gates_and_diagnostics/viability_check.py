"""One command that says which GNN tasks the corpus can support.

Consumes the pilot annotations and the raw extraction checkpoints directly -- NOT
the RDF -- so it is independent of the ChatGPT-side resolution/graph-build
pipeline and can be run the moment extraction finishes.

Consolidates the diagnostics established on 2026-08-15:

  D1 yield                 events and relations per article
  D2 cell density          articles/events per (commodity x district x quarter)
  D3 coreference pool      cross-article pairs a blocking rule admits
  D4 hard-negative pool    same-commodity, temporally-eligible competitors per response
  D5 confidence spread     is the extractor's self-reported confidence a constant

Then prints a verdict per candidate task.

IMPORTANT: extraction relations are always intra-article (relation slots index
that article's own event list). Cross-document edges, and therefore any lead time
between a disruption and its response, come from the *coreference* step, not from
extracting more articles. D3 is the metric that predicts whether that step can
produce anything -- it is the real gate, and more extraction only helps through it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path

ROOT = Path(r"C:\Users\HP\Documents\ChatGPT\News TKG")
PILOT = ROOT / "data" / "annotation" / "llm-silver-pilot-v2.json"
CKPT = ROOT / ".artifacts" / "extraction-expansion-v3" / "qwen8b-verbose-primary"
HERE = Path(__file__).parent

DISTRICTS = ["surabaya", "bangkalan", "banyuwangi", "batu", "blitar", "bojonegoro",
             "bondowoso", "gresik", "jember", "jombang", "kediri", "lamongan", "lumajang",
             "madiun", "magetan", "malang", "mojokerto", "nganjuk", "ngawi", "pacitan",
             "pamekasan", "pasuruan", "ponorogo", "probolinggo", "sampang", "sidoarjo",
             "situbondo", "sumenep", "trenggalek", "tuban", "tulungagung", "madura"]
DISRUPTION = {"SupplyShortage", "ProductionDisruption", "DistributionDisruption",
              "WeatherDisruption", "DisasterEvent", "PestDiseaseEvent", "FoodSafetyIncident"}
WILD = {"broad_food", "unclear", "", None}


def d(s):
    try:
        return date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def district(loc: str) -> str:
    low = (loc or "").lower()
    for x in DISTRICTS:
        if re.search(rf"\b{x}\b", low):
            return x
    return "_other_"


def quarter(dt) -> str:
    return f"{dt.year}Q{(dt.month - 1)//3 + 1}" if dt else "_none_"


def load(pilot_only: bool) -> list[dict]:
    """Return a flat list of article annotations from both sources."""
    arts = list(json.loads(PILOT.read_text(encoding="utf-8"))["annotations"])
    for a in arts:
        a["_source"] = "pilot"
    if not pilot_only and CKPT.exists():
        for p in sorted(CKPT.glob("ann_*.json")):
            try:
                c = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if c.get("status") != "complete" or not c.get("annotation"):
                continue
            a = dict(c["annotation"])
            a["_source"] = "expansion"
            arts.append(a)
    # Opt-in only: the chatbot route is a different extraction regime and is
    # never pooled by default. TKG_WITH_CHAT=1 measures its marginal value.
    if not pilot_only and os.environ.get("TKG_WITH_CHAT"):
        chat = Path(__file__).parent.parent / "news-tkg-chat" / "checkpoints-chat"
        for p2 in sorted(chat.glob("ann_*.json")):
            try:
                c = json.loads(p2.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not c.get("annotation"):
                continue
            a = dict(c["annotation"])
            a["_source"] = "chat"
            arts.append(a)
    return arts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-only", action="store_true",
                    help="baseline: run against the 300-article pilot alone")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    arts = load(args.pilot_only)
    by_src = Counter(a["_source"] for a in arts)

    events = []
    n_rel = 0
    for a in arts:
        n_rel += len(a.get("relations") or [])
        for e in a.get("events") or []:
            dt = d(e.get("event_date"))
            events.append({
                "event_id": e.get("event_id") or f"{a.get('candidate_id')}:llm:event:{e.get('event_slot')}",
                "event_slot": e.get("event_slot"),
                "article": a.get("candidate_id"), "source": a["_source"],
                "kind": e.get("event_kind"), "subtype": e.get("event_subtype"),
                "date": dt, "pub": d(a.get("pub_date")),
                "commodity": e.get("commodity") or "unclear",
                "district": district(e.get("location")),
                "stage": e.get("food_system_stage"),
                "conf": e.get("confidence_pct"),
            })
    dis = [e for e in events if e["subtype"] in DISRUPTION]
    res = [e for e in events if e["subtype"] not in DISRUPTION]

    R: dict = {"articles": dict(by_src), "articles_total": len(arts)}

    # -- D1 yield ---------------------------------------------------------
    R["D1_yield"] = {
        "events": len(events), "relations": n_rel,
        "events_per_article": round(len(events) / max(1, len(arts)), 3),
        "relations_per_article": round(n_rel / max(1, len(arts)), 3),
        "disruptions": len(dis), "responses": len(res),
        "pilot_reference_events_per_article": 1.197,
    }

    # -- D2 cell density --------------------------------------------------
    cells = Counter((e["commodity"], e["district"], quarter(e["date"])) for e in dis)
    v = sorted(cells.values(), reverse=True)
    R["D2_cell_density"] = {
        "cells": len(v), "median_disruptions_per_cell": v[len(v) // 2] if v else 0,
        "cells_ge_4": sum(1 for x in v if x >= 4),
        "cells_ge_8": sum(1 for x in v if x >= 8),
        "max_cell": v[0] if v else 0,
        "top_cells": [{"cell": "|".join(map(str, k)), "n": n} for k, n in cells.most_common(8)],
    }

    # -- D3 coreference candidate pool ------------------------------------
    # different article, shared subtype, same commodity, |dd| <= 45
    pool = 0
    pool_by_subtype = Counter()
    idx = defaultdict(list)
    for e in dis:
        if e["date"]:
            idx[(e["subtype"], e["commodity"])].append(e)
    for key, group in idx.items():
        for x, y in combinations(group, 2):
            if x["article"] == y["article"]:
                continue
            if abs((x["date"] - y["date"]).days) > 45:
                continue
            pool += 1
            pool_by_subtype[key[0]] += 1
    R["D3_coreference_pool"] = {
        "blocked_cross_article_pairs": pool,
        "pilot_reference": 140,
        "by_subtype": dict(pool_by_subtype.most_common(8)),
        "note": "this is the upstream driver of cross-document edges and lead time",
    }

    # -- D4 hard-negative pool --------------------------------------------
    # for each response, how many temporally eligible same-commodity disruptions
    counts = []
    for r in res:
        if not r["date"]:
            continue
        n = sum(1 for x in dis
                if x["date"] and x["date"] <= r["date"]
                and x["commodity"] == r["commodity"]
                and (r["date"] - x["date"]).days <= 90)
        counts.append(n)
    counts.sort()
    R["D4_hard_negative_pool"] = {
        "responses_evaluated": len(counts),
        "median_same_commodity_competitors_within_90d": counts[len(counts) // 2] if counts else 0,
        "p75": counts[3 * len(counts) // 4] if counts else 0,
        "share_with_ge_5": round(sum(1 for x in counts if x >= 5) / max(1, len(counts)), 3),
        "share_with_ge_10": round(sum(1 for x in counts if x >= 10) / max(1, len(counts)), 3),
    }

    # -- D5 confidence spread ---------------------------------------------
    cs = [e["conf"] for e in events if e["conf"] is not None]
    R["D5_confidence"] = {"n": len(cs), "distinct": len(set(cs)),
                          "top": dict(Counter(cs).most_common(5)),
                          "is_effectively_constant": len(set(cs)) <= 3}

    # -- D6 benchmark integrity: is the positive simply the hardest candidate? --
    # Same hardness function the split builder uses:
    #   4*same_commodity + 2*compatible + 1.5*same_location + 0.5*same_stage + recency
    # Computed here directly from annotations so no split files are needed.
    def hardness(resp, dis):
        same_c = resp["commodity"] == dis["commodity"]
        compat = same_c or resp["commodity"] in WILD or dis["commodity"] in WILD
        same_l = resp["district"] == dis["district"] and resp["district"] != "_other_"
        same_s = resp["stage"] == dis["stage"]
        lag = (resp["date"] - dis["date"]).days
        rec = max(0.0, 2.0 - min(lag, 730) / 365.0)
        return 4.0 * same_c + 2.0 * compat + 1.5 * same_l + 0.5 * same_s + rec

    by_id = {e.get("event_id"): e for e in events if e.get("event_id")}
    strictly_hardest, competitors_at_or_above, pairs_scored = 0, [], 0
    for a in arts:
        slot = {e.get("event_slot"): e.get("event_id") for e in (a.get("events") or [])}
        for rel in a.get("relations") or []:
            if (rel.get("predicate") or rel.get("relation")) != "respondsTo":
                continue
            s = rel.get("subject_event_id") or slot.get(rel.get("subject_event_slot"))
            o = rel.get("object_event_or_policy_id") or slot.get(rel.get("object_event_slot"))
            r_, d_ = by_id.get(s), by_id.get(o)
            if not r_ or not d_ or not r_["date"] or not d_["date"] or d_["date"] > r_["date"]:
                continue
            pairs_scored += 1
            h_pos = hardness(r_, d_)
            n_ge = 0
            for c in dis:
                if c is d_ or not c["date"] or c["date"] > r_["date"] or c["article"] == r_["article"]:
                    continue
                if hardness(r_, c) >= h_pos:
                    n_ge += 1
            competitors_at_or_above.append(n_ge)
            strictly_hardest += (n_ge == 0)
    cao = sorted(competitors_at_or_above)
    R["D6_benchmark_integrity"] = {
        "respondsTo_pairs_scored": pairs_scored,
        "GATE_share_positives_strictly_hardest": round(strictly_hardest / max(1, pairs_scored), 4),
        "GATE_median_competitors_at_or_above": cao[len(cao) // 2] if cao else 0,
        "share_with_ge_1_competitor": round(sum(1 for x in cao if x >= 1) / max(1, len(cao)), 3),
        "note": "cross-article competitors only; intra-article candidates are excluded",
    }

    # -- D7 task non-degeneracy (gate L0) ----------------------------------
    # A task is degenerate if a lookup keyed on the argument types already
    # solves it. Relation type is the case in point: the ontology's domain and
    # range constraints determine the relation from the kinds of its arguments,
    # so "predicting" it restates the schema. This is invisible to L1 and L2 --
    # it passes on label count and never reaches a candidate set -- so it has to
    # be checked first, and it is only detectable at construction time.
    sig_counts = defaultdict(Counter)
    n_rel_labelled = 0
    for a in arts:
        slot = {e.get("event_slot"): e.get("event_id") for e in (a.get("events") or [])}
        for rel in a.get("relations") or []:
            pred = rel.get("predicate") or rel.get("relation")
            if not pred:
                continue
            s = rel.get("subject_event_id") or slot.get(rel.get("subject_event_slot"))
            o = rel.get("object_event_or_policy_id") or slot.get(rel.get("object_event_slot"))
            sk = by_id.get(s)
            ok = by_id.get(o)
            src_kind = ("disruption" if sk["subtype"] in DISRUPTION else "response") if sk else "unknown"
            if ok:
                dst_kind = "disruption" if ok["subtype"] in DISRUPTION else "response"
            else:
                dst_kind = "policy_or_other"
            sig_counts[(src_kind, dst_kind)][pred] += 1
            n_rel_labelled += 1
    lookup_correct = sum(c.most_common(1)[0][1] for c in sig_counts.values())
    L0_share = round(lookup_correct / max(1, n_rel_labelled), 4)
    R["D7_task_degeneracy"] = {
        "relations_labelled": n_rel_labelled,
        "GATE_share_recoverable_from_argument_types": L0_share,
        "signatures": {"|".join(k): dict(v) for k, v in
                       sorted(sig_counts.items(), key=lambda kv: -sum(kv[1].values()))},
        "note": "if this is at ceiling the relation label restates the ontology's "
                "domain/range constraints and relation classification is not a task",
    }

    # -- verdicts ----------------------------------------------------------
    # Three independent layers, all must hold.
    #   L0 non-degeneracy - is the label more than a restatement of the schema?
    #   L1 integrity  - is there a difference to detect, or does the candidate
    #                   rule already determine the answer?
    #   L2 power      - could a paired test detect it? Derived from the power
    #                   analysis in power-analysis.json: at alpha=0.05, 80% power
    #                   needs n>=32 test items for a medium effect (dz=0.5) and
    #                   n>=65 for dz=0.35. At a 70/15/15 chronological split,
    #                   32 test positives implies ~215 positives overall.
    POS_FOR_MEDIUM_EFFECT = 215          # -> ~32 test positives
    POS_FOR_SMALL_MEDIUM = 435           # -> ~65 test positives
    d2, d3, d4, d6 = (R["D2_cell_density"], R["D3_coreference_pool"],
                      R["D4_hard_negative_pool"], R["D6_benchmark_integrity"])
    n_responds = sum(1 for a in arts for rel in (a.get("relations") or [])
                     if (rel.get("predicate") or rel.get("relation")) == "respondsTo")

    l1_ok = (d6["GATE_share_positives_strictly_hardest"] < 0.80
             and d6["GATE_median_competitors_at_or_above"] >= 1)
    l2_ok = n_responds >= POS_FOR_MEDIUM_EFFECT

    verdicts = {
        "L1_benchmark_integrity": {
            "viable": l1_ok,
            "gate": "share of positives strictly hardest < 0.80 AND median cross-article "
                    "competitors at-or-above positive hardness >= 1",
            "observed": f"{d6['GATE_share_positives_strictly_hardest']:.0%} strictly hardest; "
                        f"median {d6['GATE_median_competitors_at_or_above']} competitors",
        },
        "L2_statistical_power": {
            "viable": l2_ok,
            "gate": f">={POS_FOR_MEDIUM_EFFECT} respondsTo positives (-> ~32 test items, "
                    f"80% power for dz=0.5); {POS_FOR_SMALL_MEDIUM} for dz=0.35",
            "observed": f"{n_responds} respondsTo positives",
        },
        "respondsTo_target_ranking": {
            "viable": l1_ok and l2_ok,
            "gate": "requires BOTH layers",
            "observed": f"L1 {'pass' if l1_ok else 'FAIL'}; L2 {'pass' if l2_ok else 'FAIL'}",
        },
        "response_occurrence_forecasting": {
            "viable": d3["blocked_cross_article_pairs"] >= 600 and l2_ok,
            "gate": ">=600 blocked coreference pairs (cross-document lead time) AND L2 power",
            "observed": f"{d3['blocked_cross_article_pairs']} pairs; "
                        f"L2 {'pass' if l2_ok else 'FAIL'}",
        },
        "L0_task_non_degeneracy": {
            "viable": L0_share < 0.90,
            "gate": "<90% of relation labels recoverable from the argument-type "
                    "signature alone (else the label restates the ontology)",
            "observed": f"{L0_share:.0%} recoverable by lookup",
        },
        "relation_type_classification": {
            "viable": (R["D1_yield"]["relations"] >= POS_FOR_MEDIUM_EFFECT
                       and L0_share < 0.90),
            "gate": f">={POS_FOR_MEDIUM_EFFECT} relations AND L0 non-degeneracy",
            "observed": f"{R['D1_yield']['relations']} relations; "
                        f"L0 {L0_share:.0%} recoverable "
                        f"({'pass' if L0_share < 0.90 else 'FAIL'})",
        },
    }
    R["power_basis"] = {
        "alpha": 0.05, "target_power": 0.80,
        "min_test_items": {"dz=0.80": 13, "dz=0.50": 32, "dz=0.35": 65, "dz=0.20": 197},
        "assumed_test_fraction": 0.15,
    }
    R["verdicts"] = verdicts
    R["any_viable"] = any(v["viable"] for k, v in verdicts.items()
                          if k not in ("L1_benchmark_integrity", "L2_statistical_power"))

    out = args.out or HERE / ("viability-pilot-only.json" if args.pilot_only else "viability-current.json")
    out.write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"articles: {dict(by_src)}  total {len(arts)}")
    print(f"D1 yield        events {R['D1_yield']['events']} ({R['D1_yield']['events_per_article']}/art)   "
          f"relations {R['D1_yield']['relations']} ({R['D1_yield']['relations_per_article']}/art)   "
          f"disruptions {len(dis)}  responses {len(res)}")
    print(f"D2 cells        {d2['cells']} cells   median {d2['median_disruptions_per_cell']}   "
          f">=4: {d2['cells_ge_4']}   >=8: {d2['cells_ge_8']}   max {d2['max_cell']}")
    print(f"D3 coref pool   {d3['blocked_cross_article_pairs']} blocked pairs (pilot: 140)")
    print(f"D4 hard-neg     median {d4['median_same_commodity_competitors_within_90d']} competitors/response   "
          f">=5: {d4['share_with_ge_5']:.0%}   >=10: {d4['share_with_ge_10']:.0%}")
    print(f"D5 confidence   {R['D5_confidence']['distinct']} distinct values   "
          f"constant={R['D5_confidence']['is_effectively_constant']}")
    print(f"D7 degeneracy   {R['D7_task_degeneracy']['relations_labelled']} relations   "
          f"lookup ceiling {R['D7_task_degeneracy']['GATE_share_recoverable_from_argument_types']:.0%}")
    print(f"D6 integrity    {d6['respondsTo_pairs_scored']} pairs scored   "
          f"strictly-hardest {d6['GATE_share_positives_strictly_hardest']:.0%}   "
          f"median competitors>=positive {d6['GATE_median_competitors_at_or_above']}")
    print()
    for name, v in verdicts.items():
        print(f"  {'VIABLE  ' if v['viable'] else 'BLOCKED '} {name}")
        print(f"            gate: {v['gate']}")
        print(f"            obs : {v['observed']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
