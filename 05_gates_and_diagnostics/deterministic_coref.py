"""Deterministic cross-document event coreference -- no model, no GPU.

Why this can work without an LLM
--------------------------------
The three-variant judging run showed the *ambiguous* coreference cases are all
episode-vs-instance: is a five-week national shortage one event or five. Those
are genuinely undecidable from the text and they produced ensemble entropy
0.78-0.85.

The *unambiguous* cases are incidents: one flood, one district, one day, reported
by two publishers. Those need no model. So this rule deliberately refuses the
episode cases and merges only datable incidents:

  same event_subtype, different article, compatible commodity,
  same district, |event_date delta| <= WINDOW days,
  and BOTH sides precisely dated (no range/month/year, no end date)

The last clause is what keeps episodes out, and it is the whole reason the rule
can be trusted.

Validation
----------
176 pairs were judged by a three-prompt-variant ensemble earlier today. Any pair
this rule merges that the ensemble scored as distinct is a false positive, and
any pair the ensemble merged that this rule misses is a false negative. Both are
reported, so the rule is measured rather than asserted.
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
ENSEMBLE = Path(r"C:\Users\HP\Documents\Claude\news-tkg-coref\coref-ensemble-report.json")
HERE = Path(__file__).parent

DISTRICTS = ["surabaya", "bangkalan", "banyuwangi", "batu", "blitar", "bojonegoro",
             "bondowoso", "gresik", "jember", "jombang", "kediri", "lamongan", "lumajang",
             "madiun", "magetan", "malang", "mojokerto", "nganjuk", "ngawi", "pacitan",
             "pamekasan", "pasuruan", "ponorogo", "probolinggo", "sampang", "sidoarjo",
             "situbondo", "sumenep", "trenggalek", "tuban", "tulungagung", "madura"]
DISRUPTION = {"SupplyShortage", "ProductionDisruption", "DistributionDisruption",
              "WeatherDisruption", "DisasterEvent", "PestDiseaseEvent", "FoodSafetyIncident"}
WILD = {"broad_food", "unclear", "", None}
PRECISE = {"day", "estimated_day"}          # excludes range / month / year


def d(s):
    try:
        return date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def district(loc: str) -> str:
    low = (loc or "").lower()
    hits = [x for x in DISTRICTS if re.search(rf"\b{x}\b", low)]
    return hits[0] if len(hits) == 1 else ("_multi_" if hits else "_none_")


def load_events() -> list[dict]:
    out = []
    src = [(json.loads(PILOT.read_text(encoding="utf-8"))["annotations"], "pilot")]
    if CKPT.exists():
        exp = []
        for p in sorted(CKPT.glob("ann_*.json")):
            try:
                c = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if c.get("status") == "complete" and c.get("annotation"):
                exp.append(c["annotation"])
        src.append((exp, "expansion"))
    # Opt-in only: chatbot route, a different extraction regime. TKG_WITH_CHAT=1.
    if os.environ.get("TKG_WITH_CHAT"):
        chat_dir = Path(__file__).parent.parent / "news-tkg-chat" / "checkpoints-chat"
        chat = []
        for p2 in sorted(chat_dir.glob("ann_*.json")):
            try:
                c = json.loads(p2.read_text(encoding="utf-8"))
            except Exception:
                continue
            if c.get("annotation"):
                chat.append(c["annotation"])
        src.append((chat, "chat"))
    for arts, tag in src:
        for a in arts:
            for e in a.get("events") or []:
                out.append({
                    "event_id": e.get("event_id") or f"{a['candidate_id']}:llm:event:{e.get('event_slot')}",
                    "article": a.get("candidate_id"), "publisher": (a.get("source") or "").split("_")[0],
                    "pub_date": d(a.get("pub_date")), "corpus": tag,
                    "kind": e.get("event_kind"), "subtype": e.get("event_subtype"),
                    "date": d(e.get("event_date")), "end": d(e.get("event_end_date")),
                    "precision": e.get("date_precision"),
                    "commodity": e.get("commodity") or "unclear",
                    "district": district(e.get("location")),
                    "location_raw": e.get("location"),
                })
    return out


def compatible(a: str, b: str) -> bool:
    return a == b or a in WILD or b in WILD


def mergeable(x: dict, y: dict, window: int, require_cross_publisher: bool) -> bool:
    if x["article"] == y["article"]:
        return False
    if x["subtype"] != y["subtype"]:
        return False
    if x["district"] in {"_none_", "_multi_"} or x["district"] != y["district"]:
        return False
    if not (x["date"] and y["date"]) or abs((x["date"] - y["date"]).days) > window:
        return False
    if x["precision"] not in PRECISE or y["precision"] not in PRECISE:
        return False           # refuse episodes: coarse dates are sustained conditions
    if x["end"] or y["end"]:
        return False           # refuse anything with an explicit interval
    if not compatible(x["commodity"], y["commodity"]):
        return False
    if require_cross_publisher and x["publisher"] == y["publisher"]:
        return False
    return True


def cluster(pairs, ids):
    parent = {i: i for i in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    groups = defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    return [sorted(v) for v in groups.values() if len(v) > 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=3, help="max |event_date| delta in days")
    ap.add_argument("--cross-publisher-only", action="store_true")
    args = ap.parse_args()

    events = load_events()
    by_id = {e["event_id"]: e for e in events}
    dis = [e for e in events if e["subtype"] in DISRUPTION]

    # block on (subtype, district) then test pairs
    blocks = defaultdict(list)
    for e in dis:
        blocks[(e["subtype"], e["district"])].append(e)
    pairs = []
    for group in blocks.values():
        for x, y in combinations(group, 2):
            if mergeable(x, y, args.window, args.cross_publisher_only):
                pairs.append((x["event_id"], y["event_id"]))

    clusters = cluster(pairs, [e["event_id"] for e in dis])
    members = sum(len(c) for c in clusters)
    cross_pub = sum(1 for c in clusters
                    if len({by_id[m]["publisher"] for m in c}) > 1)
    cross_corpus = sum(1 for c in clusters
                       if len({by_id[m]["corpus"] for m in c}) > 1)

    # lead time: first report of the cluster vs last report
    leads = []
    for c in clusters:
        pubs = sorted(by_id[m]["pub_date"] for m in c if by_id[m]["pub_date"])
        if len(pubs) > 1:
            leads.append((pubs[-1] - pubs[0]).days)
    leads.sort()

    # ---- validation against the 3-variant ensemble ------------------------
    val = {"available": ENSEMBLE.exists()}
    if ENSEMBLE.exists():
        ens = json.loads(ENSEMBLE.read_text(encoding="utf-8"))
        judged = {tuple(sorted(r["pair"])): r["ensemble_same"] for r in ens["rows"]}
        merged_set = {tuple(sorted(p)) for p in pairs}
        fp = [(k, v) for k, v in judged.items() if k in merged_set and v < 0.25]
        tp = [(k, v) for k, v in judged.items() if k in merged_set and v >= 0.50]
        fn = [(k, v) for k, v in judged.items() if k not in merged_set and v >= 0.50]
        overlap = [k for k in judged if k in merged_set]
        val.update({
            "judged_pairs": len(judged),
            "also_proposed_by_rule": len(overlap),
            "false_positives_rule_merges_ensemble_says_distinct": len(fp),
            "true_positives_both_merge": len(tp),
            "false_negatives_ensemble_merges_rule_misses": len(fn),
            "fp_examples": [f"{a} + {b} (ens={v})" for (a, b), v in fp[:5]],
            "fn_examples": [f"{a} + {b} (ens={v})" for (a, b), v in fn[:5]],
        })

    report = {
        "window_days": args.window,
        "cross_publisher_only": args.cross_publisher_only,
        "rule": "same subtype + same single district + compatible commodity + precise dates "
                "(day/estimated_day, no end date) + different article",
        # membership, so downstream tools can actually apply the merges
        "cluster_members": [
            {"canonical": c[0], "members": c,
             "articles": sorted({by_id[m]["article"] for m in c if by_id[m].get("article")}),
             "publishers": sorted({by_id[m]["publisher"] for m in c if by_id[m].get("publisher")}),
             "subtype": by_id[c[0]]["subtype"], "district": by_id[c[0]]["district"]}
            for c in clusters],
        "disruption_events": len(dis),
        "candidate_pairs_merged": len(pairs),
        "clusters": len(clusters),
        "clustered_events": members,
        "clusters_spanning_multiple_publishers": cross_pub,
        "clusters_spanning_pilot_and_expansion": cross_corpus,
        "cluster_size_hist": dict(Counter(len(c) for c in clusters)),
        "lead_time_days": {"n": len(leads),
                           "median": leads[len(leads) // 2] if leads else None,
                           "p75": leads[3 * len(leads) // 4] if leads else None,
                           "max": leads[-1] if leads else None,
                           "share_gt_0": round(sum(1 for x in leads if x > 0) / len(leads), 3) if leads else None},
        "validation_vs_llm_ensemble": val,
        "largest_clusters": [
            {"n": len(c), "subtype": by_id[c[0]]["subtype"], "district": by_id[c[0]]["district"],
             "dates": sorted({str(by_id[m]["date"]) for m in c}),
             "publishers": sorted({by_id[m]["publisher"] for m in c})}
            for c in sorted(clusters, key=len, reverse=True)[:6]],
    }
    out = HERE / f"deterministic-coref-w{args.window}{'-xpub' if args.cross_publisher_only else ''}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"window {args.window}d  cross-publisher-only={args.cross_publisher_only}")
    print(f"  disruptions {len(dis)}   merged pairs {len(pairs)}   clusters {len(clusters)}  "
          f"covering {members} events")
    print(f"  multi-publisher clusters {cross_pub}   pilot+expansion clusters {cross_corpus}")
    if leads:
        print(f"  lead time days: median {report['lead_time_days']['median']}  "
              f"max {report['lead_time_days']['max']}  >0: {report['lead_time_days']['share_gt_0']:.0%}")
    if val.get("available"):
        print(f"  validation vs LLM ensemble: overlap {val['also_proposed_by_rule']}  "
              f"TP {val['true_positives_both_merge']}  "
              f"FP {val['false_positives_rule_merges_ensemble_says_distinct']}  "
              f"FN {val['false_negatives_ensemble_merges_rule_misses']}")
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
