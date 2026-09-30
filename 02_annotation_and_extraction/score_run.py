"""Score any extraction run over the pilot calibration packets against the
existing pilot silver labels.

Backend-agnostic: give it a runs file mapping article_id -> raw v3 output
(the `article_relevance` / `events` / `relations` shape with
`evidence_sentence_id`). Works for Claude, qwen3:8b, or anything else, so runs
are directly comparable.

Yardstick: the two existing frontier passes over all 300 pilot articles agree at
event Jaccard 0.417 with 71% subtype agreement on matched events. A backend near
that is consistent with the corpus; far below it is not.

Read-only w.r.t. the ChatGPT folder. Writes its report next to this file.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
FRONTIER_BASELINE = {"event_jaccard": 0.417, "subtype_agreement": 0.71, "relation_jaccard": 0.407}
MATCH_THRESHOLD = 0.45


def norm(t: str) -> str:
    return re.sub(r"\W+", " ", (t or "").lower()).strip()


def jac(a: str, b: str) -> float:
    s1, s2 = set(norm(a).split()), set(norm(b).split())
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def kappa(pairs: list[tuple[str, str]]) -> dict:
    tab = Counter(pairs)
    n = sum(tab.values())
    if not n:
        return {}
    obs = sum(v for (x, y), v in tab.items() if x == y) / n
    ma, mb = Counter(), Counter()
    for (x, y), v in tab.items():
        ma[x] += v
        mb[y] += v
    exp = sum(ma[k] * mb[k] for k in set(ma) | set(mb)) / (n * n)
    return {
        "confusion": {f"run={x}|ref={y}": v for (x, y), v in sorted(tab.items(), key=lambda kv: -kv[1])},
        "observed_agreement": round(obs, 4),
        "cohens_kappa": round((obs - exp) / (1 - exp), 4) if exp < 1 else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="JSON: {article_id: raw v3 output}")
    ap.add_argument("--label", default=None, help="name for this backend in the report")
    ap.add_argument("--packets", type=Path, default=HERE / "pilot-calibration-packets.jsonl")
    ap.add_argument("--reference", type=Path, default=HERE / "pilot-calibration-reference.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    label = args.label or args.run.stem
    packets = {}
    with args.packets.open(encoding="utf-8") as fh:
        for line in fh:
            p = json.loads(line)
            packets[p["article_id"]] = {
                s["sentence_id"]: s["text"] for s in p["blind_payload"]["source_sentences"]
            }
    ref = json.loads(args.reference.read_text(encoding="utf-8"))["reference"]
    run = json.loads(args.run.read_text(encoding="utf-8"))
    run = run.get("annotations", run) if isinstance(run, dict) else run
    if isinstance(run, list):
        run = {r["article_id"]: r for r in run}

    ids = sorted(set(run) & set(ref))
    missing = sorted(set(ref) - set(run))

    def quote_of(ev: dict, aid: str) -> str:
        if ev.get("evidence_quote"):
            return str(ev["evidence_quote"])
        sid = ev.get("evidence_sentence_id")
        return packets.get(aid, {}).get(sid, "") if sid else ""

    rel_pairs, per_stratum = [], defaultdict(Counter)
    tot = Counter()
    subtype_conf, pred_conf = Counter(), Counter()
    conf_values, per_article = [], []

    for aid in ids:
        r, g = run[aid], ref[aid]
        st = g["stratum"]
        rr = r.get("article_relevance")
        gr = g.get("article_relevance")
        rel_pairs.append((str(rr), str(gr)))

        re_, ge = r.get("events", []) or [], g.get("events", []) or []
        rrel, grel = r.get("relations", []) or [], g.get("relations", []) or []
        tot["events_run"] += len(re_)
        tot["events_ref"] += len(ge)
        tot["relations_run"] += len(rrel)
        tot["relations_ref"] += len(grel)
        per_stratum[st]["articles"] += 1
        per_stratum[st]["events_run"] += len(re_)
        per_stratum[st]["events_ref"] += len(ge)

        for e in re_:
            c = e.get("confidence_pct")
            if c is not None:
                conf_values.append(c)
        if r.get("article_confidence_pct") is not None:
            conf_values.append(r["article_confidence_pct"])

        used, matched, sub_ok, date_ok = set(), 0, 0, 0
        for x in re_:
            best, bs = None, 0.0
            for i, y in enumerate(ge):
                if i in used:
                    continue
                s = jac(quote_of(x, aid), quote_of(y, aid))
                if s > bs:
                    best, bs = i, s
            if best is not None and bs >= MATCH_THRESHOLD:
                used.add(best)
                matched += 1
                y = ge[best]
                so = x.get("event_subtype") == y.get("event_subtype")
                do = str(x.get("event_date") or "") == str(y.get("event_date") or "")
                sub_ok += so
                date_ok += do
                if not so:
                    subtype_conf[f"{x.get('event_subtype')} -> {y.get('event_subtype')}"] += 1
        tot["events_matched"] += matched
        tot["matched_same_subtype"] += sub_ok
        tot["matched_same_date"] += date_ok
        per_stratum[st]["events_matched"] += matched

        rused, rmatched = set(), 0
        for x in rrel:
            best, bs = None, 0.0
            for i, y in enumerate(grel):
                if i in rused:
                    continue
                s = jac(quote_of(x, aid), quote_of(y, aid))
                if s > bs:
                    best, bs = i, s
            if best is not None and bs >= MATCH_THRESHOLD:
                rused.add(best)
                rmatched += 1
                pred_conf[f"{x.get('predicate')} -> {grel[best].get('predicate')}"] += 1
        tot["relations_matched"] += rmatched

        per_article.append({
            "article_id": aid, "stratum": st,
            "relevance_run": rr, "relevance_ref": gr,
            "events_run": len(re_), "events_ref": len(ge), "events_matched": matched,
            "relations_run": len(rrel), "relations_ref": len(grel), "relations_matched": rmatched,
        })

    ev_denom = tot["events_run"] + tot["events_ref"] - tot["events_matched"]
    rel_denom = tot["relations_run"] + tot["relations_ref"] - tot["relations_matched"]
    ev_j = round(tot["events_matched"] / ev_denom, 4) if ev_denom else None
    rel_j = round(tot["relations_matched"] / rel_denom, 4) if rel_denom else None

    # false positives on articles the reference calls irrelevant
    fp = [a for a in per_article if a["stratum"].startswith("not_relevant") and a["events_run"] > 0]
    # over-fragmentation: run emits strictly more events than reference
    frag = [a for a in per_article if a["events_run"] > a["events_ref"]]
    miss = [a for a in per_article if a["events_run"] < a["events_ref"]]

    report = {
        "backend": label,
        "run_file": str(args.run),
        "articles_scored": len(ids),
        "articles_missing_from_run": missing,
        "match_rule": f"evidence token Jaccard >= {MATCH_THRESHOLD}, greedy",
        "frontier_baseline_for_comparison": FRONTIER_BASELINE,
        "relevance": kappa(rel_pairs),
        "events": {
            "run_total": tot["events_run"], "ref_total": tot["events_ref"],
            "matched": tot["events_matched"],
            "jaccard_vs_reference": ev_j,
            "subtype_agreement_on_matched": round(tot["matched_same_subtype"] / tot["events_matched"], 4) if tot["events_matched"] else None,
            "date_agreement_on_matched": round(tot["matched_same_date"] / tot["events_matched"], 4) if tot["events_matched"] else None,
            "events_per_article_run": round(tot["events_run"] / len(ids), 3) if ids else None,
            "events_per_article_ref": round(tot["events_ref"] / len(ids), 3) if ids else None,
            "subtype_disagreements": dict(subtype_conf.most_common(15)),
        },
        "relations": {
            "run_total": tot["relations_run"], "ref_total": tot["relations_ref"],
            "matched": tot["relations_matched"],
            "jaccard_vs_reference": rel_j,
            "predicate_agreement_on_matched": dict(pred_conf.most_common(15)),
        },
        "failure_modes": {
            "false_positive_articles": len(fp),
            "false_positive_detail": fp[:10],
            "over_fragmented_articles": len(frag),
            "under_extracted_articles": len(miss),
        },
        "confidence": {
            "n": len(conf_values),
            "distinct_values": len(set(conf_values)),
            "distribution": dict(Counter(conf_values).most_common(10)),
            "is_effectively_constant": len(set(conf_values)) <= 2 if conf_values else None,
        },
        "per_stratum": {k: dict(v) for k, v in sorted(per_stratum.items())},
        "per_article": per_article,
    }

    out = args.out or HERE / f"score-{label}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    e, rl = report["events"], report["relations"]
    print(f"backend            : {label}")
    print(f"articles scored    : {len(ids)}" + (f"  (missing {len(missing)})" if missing else ""))
    print(f"relevance kappa    : {report['relevance'].get('cohens_kappa')}  "
          f"(agreement {report['relevance'].get('observed_agreement')})")
    print(f"events run/ref     : {e['run_total']}/{e['ref_total']}  matched {e['matched']}")
    print(f"event Jaccard      : {e['jaccard_vs_reference']}   [frontier pair = {FRONTIER_BASELINE['event_jaccard']}]")
    print(f"subtype agreement  : {e['subtype_agreement_on_matched']}   [frontier pair = {FRONTIER_BASELINE['subtype_agreement']}]")
    print(f"relations run/ref  : {rl['run_total']}/{rl['ref_total']}  J={rl['jaccard_vs_reference']}")
    print(f"false-positive arts: {report['failure_modes']['false_positive_articles']}  "
          f"over-fragmented: {report['failure_modes']['over_fragmented_articles']}  "
          f"under-extracted: {report['failure_modes']['under_extracted_articles']}")
    print(f"confidence distinct: {report['confidence']['distinct_values']} "
          f"(constant={report['confidence']['is_effectively_constant']})")
    print(f"report             : {out}")


if __name__ == "__main__":
    main()
