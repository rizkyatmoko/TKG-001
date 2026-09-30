"""Build the thin modelling view directly from raw annotations.

The RDF materialisation covers only the original 300-article pilot. The 553
expansion articles exist as raw extraction checkpoints and would need the
ChatGPT-side resolution and graph-build pipeline to reach RDF. That pipeline is
not ours to run, and depending on it would couple every downstream result to it.

So this exporter reads the pilot annotations and the extraction checkpoints
directly and emits the same node/edge tables that `export_modeling_view.py`
produces from RDF. Everything downstream -- rgcn.py, node_classification.py,
temporal_masking.py -- consumes the tables, so nothing else changes.

Differences from the RDF export, stated so results stay comparable:
  * no canonical events, since cross-document resolution has not been run on
    the expansion; every event is a mention;
  * no episode grouping;
  * relation edges come from the per-article relation slots, so they are
    intra-article by construction (as they are in the RDF export too, before
    resolution adds cross-document entailments).

Writes to view-full/ so the RDF-derived view/ stays intact for comparison.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(r"C:\Users\HP\Documents\ChatGPT\News TKG")
PILOT = ROOT / "data" / "annotation" / "llm-silver-pilot-v2.json"
CKPT = ROOT / ".artifacts" / "extraction-expansion-v3" / "qwen8b-verbose-primary"
POLICY = ROOT / "data" / "annotation" / "policy-lookup.json"
HERE = Path(__file__).parent
OUT = HERE / "view-full"
OUT.mkdir(parents=True, exist_ok=True)

DIS = {"SupplyShortage", "ProductionDisruption", "DistributionDisruption",
       "WeatherDisruption", "DisasterEvent", "PestDiseaseEvent", "FoodSafetyIncident"}
TARGETS = {"respondsTo", "implementsPolicy", "citesPolicy", "reportedCauseOf"}


def d(s):
    try:
        return date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def load_articles() -> list[dict]:
    arts = []
    for a in json.loads(PILOT.read_text(encoding="utf-8"))["annotations"]:
        a = dict(a); a["_source"] = "pilot"; arts.append(a)
    if CKPT.exists():
        for p in sorted(CKPT.glob("ann_*.json")):
            try:
                c = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if c.get("status") != "complete" or not c.get("annotation"):
                continue
            a = dict(c["annotation"])
            a["_source"] = "expansion"
            a["_model"] = c.get("model", "")
            arts.append(a)
    # Optional: fold in the chatbot route. Off by default — it is a different
    # extraction regime (different model, different prompt) and must never be
    # pooled silently. Set TKG_WITH_CHAT=1 to measure its marginal value.
    if os.environ.get("TKG_WITH_CHAT"):
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
            a["_model"] = c.get("model", "chatbot")
            arts.append(a)
    return arts


def write(name, header, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, delimiter="\t", extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    return len(rows)


arts = load_articles()
policies = json.loads(POLICY.read_text(encoding="utf-8"))
counts: dict[str, int] = {}

# ------------------------------------------------------------------ nodes
articles, events = [], []
actors, locations, commodities, publishers = set(), set(), set(), set()
for a in arts:
    pub = (a.get("source") or "").split("_")[0]
    publishers.add(pub)
    articles.append({
        "article_id": a.get("candidate_id"), "publisher_id": pub,
        "source_channel_key": a.get("source", ""),
        "publication_datetime": a.get("pub_date", ""),
        "article_relevance": a.get("article_relevance", ""),
        "extraction_source": a["_source"],
        "extraction_model": a.get("_model", "codex-semantic-pass" if a["_source"] == "pilot" else ""),
        "title": (a.get("title") or "").replace("\t", " ").replace("\n", " "),
    })
    for e in a.get("events") or []:
        eid = e.get("event_id") or f"{a['candidate_id']}:llm:event:{e.get('event_slot')}"
        act = (e.get("actor") or "").strip()
        loc = (e.get("location") or "").strip()
        com = (e.get("commodity") or "").strip()
        if act:
            actors.add(act)
        if loc:
            locations.add(loc)
        if com:
            commodities.add(com)
        sub = e.get("event_subtype") or ""
        events.append({
            "event_id": eid, "node_kind": "mention",
            "event_subtype": sub,
            "all_types": sub,
            "event_kind": e.get("event_kind", ""),
            "event_date": e.get("event_date", ""), "event_end_date": e.get("event_end_date", ""),
            "date_precision": e.get("date_precision", ""),
            "report_time": a.get("pub_date", ""), "available_date": a.get("pub_date", ""),
            "commodity_id": com, "location_id": loc, "stage_id": e.get("food_system_stage", ""),
            "actor_ids": act, "article_id": a.get("candidate_id"), "publisher_id": pub,
            "extraction_confidence": e.get("confidence_pct", ""),
            "extraction_source": a["_source"],
            "episode_id": "", "granularity_basis": "",
            "label_status": "silver", "human_verified": "false",
        })

counts["articles"] = write("nodes_article.tsv", list(articles[0]), articles)
counts["events"] = write("nodes_event.tsv", list(events[0]), events)
counts["publishers"] = write("nodes_publisher.tsv", ["publisher_id", "label"],
                             [{"publisher_id": p, "label": p} for p in sorted(publishers)])
counts["actor"] = write("nodes_actor.tsv", ["actor_id", "label"],
                        [{"actor_id": x, "label": x} for x in sorted(actors)])
counts["location"] = write("nodes_location.tsv", ["location_id", "label"],
                           [{"location_id": x, "label": x} for x in sorted(locations)])
counts["commodity"] = write("nodes_commodity.tsv", ["commodity_id", "label"],
                            [{"commodity_id": x, "label": x} for x in sorted(commodities)])
pol_rows = [{"policy_id": k,
             "enactment_date": (v or {}).get("enactment_date", "") if isinstance(v, dict) else "",
             "instrument_type": (v or {}).get("document_type", "") if isinstance(v, dict) else ""}
            for k, v in (policies.items() if isinstance(policies, dict) else [])]
counts["policies"] = write("nodes_policy.tsv", ["policy_id", "enactment_date", "instrument_type"],
                           pol_rows)
pol_ids = {r["policy_id"] for r in pol_rows}

# ------------------------------------------------------------------ edges
ev_by_id = {e["event_id"]: e for e in events}
edges = []


def add(src, st, rel, dst, dt, **kw):
    edges.append({"src_id": src, "src_type": st, "relation": rel, "dst_id": dst, "dst_type": dt,
                  "event_time": kw.get("et", ""), "report_time": kw.get("rt", ""),
                  "available_date": kw.get("ad", ""), "confidence": kw.get("conf", ""),
                  "assertion_derivation": "", "extraction_method": kw.get("src_tag", ""),
                  "support_assertion_count": "", "evidence_quote": kw.get("ev", "")[:400]})


for a in arts:
    slot = {e.get("event_slot"): (e.get("event_id") or f"{a['candidate_id']}:llm:event:{e.get('event_slot')}")
            for e in (a.get("events") or [])}
    for rel in a.get("relations") or []:
        pred = rel.get("predicate") or rel.get("relation")
        if pred not in TARGETS:
            continue
        s = rel.get("subject_event_id") or slot.get(rel.get("subject_event_slot"))
        o = rel.get("object_event_or_policy_id") or rel.get("policy_id") or \
            slot.get(rel.get("object_event_slot"))
        if not s or not o or s not in ev_by_id:
            continue
        dst_type = "policy" if (o in pol_ids or o not in ev_by_id) else "event"
        if dst_type == "event" and o not in ev_by_id:
            continue
        se = ev_by_id[s]
        add(s, "event", pred, o, dst_type, et=se["event_date"], rt=se["report_time"],
            ad=se["available_date"], conf=rel.get("confidence_pct", ""),
            src_tag=a["_source"], ev=(rel.get("evidence_quote") or "").replace("\t", " "))

for e in events:
    for col, rel, dt in (("commodity_id", "concernsCommodity", "commodity"),
                         ("location_id", "occursIn", "location"),
                         ("actor_ids", "hasActor", "actor"),
                         ("article_id", "reportedIn", "article"),
                         ("publisher_id", "sourcePublisher", "publisher")):
        if e[col]:
            add(e["event_id"], "event", rel, e[col], dt,
                et=e["event_date"], rt=e["report_time"], ad=e["available_date"],
                src_tag=e["extraction_source"])
for a in articles:
    if a["publisher_id"]:
        add(a["article_id"], "article", "publishedBy", a["publisher_id"], "publisher",
            rt=a["publication_datetime"], ad=a["publication_datetime"])

counts["edges"] = write("edges.tsv", list(edges[0]), edges)
rel_census = Counter(e["relation"] for e in edges)

manifest = {
    "built_from": "raw annotations (pilot JSON + extraction checkpoints), NOT RDF",
    "pilot": str(PILOT), "checkpoints": str(CKPT),
    "articles_by_source": dict(Counter(a["extraction_source"] for a in articles)),
    "counts": counts, "relation_census": dict(rel_census.most_common()),
    "caveats": ["no canonical events (resolution not run on the expansion)",
                "no episode grouping",
                "relation edges are intra-article by construction"],
    "human_verified": False, "label_status": "silver",
}
(HERE / "manifest-full.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                         encoding="utf-8")

print(json.dumps(counts, indent=2))
print("\narticles by source:", dict(Counter(a["extraction_source"] for a in articles)))
print("relations:", dict(rel_census.most_common()))
print(f"\nwrote {OUT}")
