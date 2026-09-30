"""Ingest chatbot JSON replies back into pipeline-compatible checkpoints.

Save each reply as responses/batch-XXX.json (raw paste is fine — leading prose
and ``` fences are stripped). This validates every record and writes one
checkpoint per article in the same shape run_expansion_extraction_v3.py writes,
so viability_check.py, the exporters and everything downstream work unchanged.

Validation, because a chatbot will drift from the schema in ways a local runner
does not:

  V1  every evidence_quote is an exact substring of that article's body
  V2  no event date later than the publication date
  V3  event_kind and event_subtype agree (no response typed as a disruption)
  V4  relation slots resolve to emitted events; policy relations carry a policy_id
  V5  article ids match the batch that was sent, and none is missing

Records failing V1 or V2 are rejected outright; V3 and V4 failures are dropped at
the field level and counted. Nothing invalid is written.

Checkpoints go to a SEPARATE directory so the qwen extraction stays untouched and
the two regimes remain distinguishable by extraction_model.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\Users\HP\Documents\ChatGPT\News TKG")
QUEUE = ROOT / "data" / "link_prediction" / "expansion-v3" / "article_queue.jsonl"
HERE = Path(__file__).parent
RESP = HERE / "responses"
OUTCK = HERE / "checkpoints-chat"

DIS = {"WeatherDisruption", "DisasterEvent", "PestDiseaseEvent", "ProductionDisruption",
       "SupplyShortage", "DistributionDisruption", "FoodSafetyIncident"}
RES = {"ReserveRelease", "FoodAssistance", "MarketOperationAction", "ProcurementAction",
       "DistributionIntervention", "ImportExportAction", "InspectionEnforcement",
       "ProductionAssistance", "CrisisPreparednessAction", "OtherResponseAction"}
PREDS = {"respondsTo", "reportedCauseOf", "implementsPolicy", "citesPolicy"}


def norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(s or "")).split())


def extract_json(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    i, j = t.find("["), t.rfind("]")
    if i == -1 or j == -1:
        raise ValueError("no JSON array found")
    return json.loads(t[i:j + 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", type=Path, default=RESP)
    ap.add_argument("--out", type=Path, default=OUTCK)
    ap.add_argument("--model", default="chatbot-frontier")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    bodies, pubdates = {}, {}
    for line in QUEUE.open(encoding="utf-8"):
        r = json.loads(line)
        bodies[r["article_id"]] = norm(r.get("body"))
        pubdates[r["article_id"]] = str(r.get("pub_date"))[:10]

    index = {}
    idx_file = HERE / "batch-index.json"
    if idx_file.exists():
        for b in json.loads(idx_file.read_text(encoding="utf-8"))["batches"]:
            index[b["batch"]] = b["article_ids"]

    stats = Counter()
    problems = []
    written = 0

    for f in sorted(args.responses.glob("batch-*.json")) if args.responses.exists() else []:
        bno = int(re.search(r"(\d+)", f.stem).group(1))
        try:
            recs = extract_json(f.read_text(encoding="utf-8"))
        except Exception as exc:
            problems.append({"batch": bno, "error": f"unparseable: {exc!r}"}); stats["batch_unparseable"] += 1
            continue

        expected = index.get(bno)
        got = [r.get("candidate_id") for r in recs]
        if expected:
            missing = [a for a in expected if a not in got]
            extra = [a for a in got if a not in expected]
            if missing:
                problems.append({"batch": bno, "missing_articles": missing}); stats["V5_missing"] += len(missing)
            if extra:
                problems.append({"batch": bno, "unexpected_articles": extra}); stats["V5_unexpected"] += len(extra)

        for rec in recs:
            aid = rec.get("candidate_id")
            if aid not in bodies:
                problems.append({"batch": bno, "article": aid, "error": "unknown article_id"})
                stats["unknown_article"] += 1
                continue
            body, pub = bodies[aid], pubdates[aid]

            good_events, slots = [], {}
            reject = None
            for e in rec.get("events") or []:
                q = norm(e.get("evidence_quote"))
                if not q or q not in body:                                    # V1
                    reject = f"evidence quote not an exact substring: {q[:60]!r}"
                    break
                d = str(e.get("event_date") or "")[:10]
                if d and pub and d > pub:                                      # V2
                    reject = f"event_date {d} after publication {pub}"
                    break
                st = e.get("event_subtype")
                k = e.get("event_kind")
                if (k == "response" and st in DIS) or (k == "disruption" and st in RES):  # V3
                    stats["V3_kind_subtype_mismatch"] += 1
                    continue
                if st not in DIS and st not in RES:
                    stats["V3_unknown_subtype"] += 1
                    continue
                slot = e.get("event_slot") or (len(good_events) + 1)
                e = dict(e)
                e["event_slot"] = slot
                e["event_id"] = f"{aid}:llm:event:{slot}"
                slots[slot] = e["event_id"]
                good_events.append(e)
            if reject:
                problems.append({"batch": bno, "article": aid, "error": reject})
                stats["article_rejected"] += 1
                continue

            good_rels = []
            for rl in rec.get("relations") or []:
                p = rl.get("predicate")
                if p not in PREDS:
                    stats["V4_bad_predicate"] += 1; continue
                q = norm(rl.get("evidence_quote"))
                if q and q not in body:                                        # V1
                    stats["V4_relation_quote_not_substring"] += 1; continue
                s = slots.get(rl.get("subject_event_slot"))
                if not s:
                    stats["V4_unresolved_subject"] += 1; continue
                if p in ("implementsPolicy", "citesPolicy"):
                    if not (rl.get("policy_id") or "").strip():
                        stats["V4_policy_without_id"] += 1; continue
                    o = (rl.get("policy_id") or "").strip()
                else:
                    o = slots.get(rl.get("object_event_slot"))
                    if not o or o == s:
                        stats["V4_unresolved_or_self_object"] += 1; continue
                rl = dict(rl); rl["subject_event_id"] = s; rl["object_event_or_policy_id"] = o
                good_rels.append(rl)

            annotation = {
                "candidate_id": aid, "pub_date": pub,
                "source": rec.get("source", ""), "title": rec.get("title", ""),
                "article_relevance": rec.get("article_relevance", ""),
                "east_java_basis": rec.get("east_java_basis", ""),
                "llm_confidence_pct": rec.get("article_confidence_pct", ""),
                "events": good_events, "relations": good_rels,
                "extraction_model": args.model, "human_verified": False,
                "review_status": "complete",
            }
            (args.out / f"{aid}.json").write_text(json.dumps(
                {"article_id": aid, "status": "complete", "model": args.model,
                 "extraction_route": "chatbot", "annotation": annotation},
                indent=2, ensure_ascii=False), encoding="utf-8")
            written += 1
            stats["events"] += len(good_events); stats["relations"] += len(good_rels)

    report = {"checkpoints_written": written, "stats": dict(stats),
              "problems": problems[:40], "problem_count": len(problems),
              "output_dir": str(args.out)}
    (HERE / "ingest-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    print(f"checkpoints written : {written}")
    print(f"events / relations  : {stats['events']} / {stats['relations']}")
    print(f"problems            : {len(problems)}")
    for k, v in sorted(stats.items()):
        if k not in ("events", "relations"):
            print(f"   {k}: {v}")
    if problems:
        print("\nfirst problems:")
        for p in problems[:5]:
            print("  ", p)


if __name__ == "__main__":
    main()
