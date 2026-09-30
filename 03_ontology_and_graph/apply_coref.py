"""Collapse coreferent disruption events in a modeling view into single nodes.

Why this exists
---------------
The exported view is article-local: every relation the extractor emits links two
event slots inside one article's annotation, so 100% of respondsTo edges are
intra-article and the graph has no cross-document structure at all. Deterministic
coreference finds the same incident reported by two publishers, but nothing was
applying those merges to the view.

Applying them is what creates cross-document structure. When disruption D is
reported in articles A and B, and B carries R -> respondsTo -> D_B, collapsing
D_A and D_B gives a node whose evidence spans both articles and which carries two
distinct report times. That is the only place in this pipeline where a
report-time-constrained neighbourhood is non-trivial.

Bitemporal care
---------------
A merged node keeps the EARLIEST event_date (the incident happened once) but the
LATEST available_date is NOT used -- each surviving edge keeps its own article's
report time, so a model restricted to information available at time t still sees
only the reports that had actually appeared by t. Collapsing report times would
leak the second publisher's coverage into the first publisher's neighbourhood.

Read-only w.r.t. its input; writes a new view directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent


def read(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write(p: Path, rows: list[dict], cols: list[str]) -> None:
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", type=Path, required=True)
    ap.add_argument("--coref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    clusters = json.loads(args.coref.read_text(encoding="utf-8")).get("cluster_members", [])
    if not clusters:
        raise SystemExit("coref report has no cluster_members -- regenerate it")

    # member -> canonical
    canon: dict[str, str] = {}
    for c in clusters:
        for m in c["members"]:
            canon[m] = c["canonical"]

    events = read(args.view / "nodes_event.tsv")
    by_id = {e["event_id"]: e for e in events}
    present = {c["canonical"]: [m for m in c["members"] if m in by_id] for c in clusters}
    applied = {k: v for k, v in present.items() if len(v) > 1}

    # ---- collapse event rows ------------------------------------------------
    kept, dropped = [], 0
    merged_articles: dict[str, list[str]] = {}
    for e in events:
        eid = e["event_id"]
        tgt = canon.get(eid, eid)
        if tgt != eid and tgt in by_id:
            dropped += 1
            merged_articles.setdefault(tgt, []).append(e.get("article_id", ""))
            continue
        kept.append(e)
    for e in kept:
        eid = e["event_id"]
        if eid in applied:
            members = [by_id[m] for m in applied[eid]]
            dates = sorted(d for d in (m.get("event_date") for m in members) if d)
            if dates:
                e["event_date"] = dates[0]          # incident happened once: earliest
            arts = sorted({m.get("article_id", "") for m in members if m.get("article_id")})
            pubs = sorted({m.get("publisher_id", "") for m in members if m.get("publisher_id")})
            e["coref_article_ids"] = "|".join(arts)
            e["coref_publisher_ids"] = "|".join(pubs)
            e["coref_size"] = str(len(members))
        else:
            e.setdefault("coref_article_ids", e.get("article_id", ""))
            e.setdefault("coref_publisher_ids", e.get("publisher_id", ""))
            e.setdefault("coref_size", "1")

    # ---- rewrite edges ------------------------------------------------------
    edges = read(args.view / "edges.tsv")
    alive = {e["event_id"] for e in kept}
    out_edges, self_loops = [], 0
    for r in edges:
        r = dict(r)
        for side in ("src_id", "dst_id"):
            v = r.get(side, "")
            if v in canon and canon[v] in alive:
                r[side] = canon[v]
        if r["src_id"] == r["dst_id"]:
            self_loops += 1                      # a merge made both ends the same event
            continue
        out_edges.append(r)

    # ---- how much cross-document structure did this create? -----------------
    ev_now = {e["event_id"]: e for e in kept}

    def arts_of(nid: str) -> set[str]:
        e = ev_now.get(nid)
        if not e:
            return set()
        return {a for a in (e.get("coref_article_ids") or "").split("|") if a}

    cross = intra = 0
    for r in out_edges:
        if r["relation"] != "respondsTo":
            continue
        a, b = arts_of(r["src_id"]), arts_of(r["dst_id"])
        if a and b and not (a & b):
            cross += 1
        elif a and b:
            # still cross-document if either endpoint itself spans articles
            cross += 1 if (len(a) > 1 or len(b) > 1) else 0
            intra += 0 if (len(a) > 1 or len(b) > 1) else 1

    args.out.mkdir(parents=True, exist_ok=True)
    for p in args.view.iterdir():
        if p.is_file() and p.name not in ("nodes_event.tsv", "edges.tsv"):
            shutil.copy2(p, args.out / p.name)
    ecols = list(events[0].keys()) + ["coref_article_ids", "coref_publisher_ids", "coref_size"]
    ecols = list(dict.fromkeys(ecols))
    write(args.out / "nodes_event.tsv", kept, ecols)
    write(args.out / "edges.tsv", out_edges, list(edges[0].keys()))

    multi = [e for e in kept if int(e.get("coref_size", 1)) > 1]
    multipub = [e for e in multi if len(set((e.get("coref_publisher_ids") or "").split("|"))) > 1]
    report = {
        "view_in": str(args.view), "view_out": str(args.out),
        "clusters_in_report": len(clusters), "clusters_applied": len(applied),
        "events_before": len(events), "events_after": len(kept), "events_absorbed": dropped,
        "merged_nodes": len(multi), "merged_nodes_multi_publisher": len(multipub),
        "edges_before": len(edges), "edges_after": len(out_edges),
        "edges_dropped_as_self_loops": self_loops,
        "respondsTo_cross_document": cross, "respondsTo_intra_article": intra,
        "relation_mix": dict(Counter(r["relation"] for r in out_edges).most_common()),
    }
    (args.out / "coref-apply-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"clusters applied      {len(applied)} of {len(clusters)}")
    print(f"events                {len(events)} -> {len(kept)}  ({dropped} absorbed)")
    print(f"merged nodes          {len(multi)}   multi-publisher {len(multipub)}")
    print(f"edges                 {len(edges)} -> {len(out_edges)}  ({self_loops} self-loops dropped)")
    print(f"respondsTo cross-doc  {cross}   intra-article {intra}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
