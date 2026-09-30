"""Report-time-constrained neighbourhood construction, and a test of whether it
actually matters on this graph.

The claim behind the proposed contribution is that a news-derived TKG must
restrict message passing to what was *reported* by the prediction time, not what
*happened* by then. Standard temporal GNNs mask on event time, which silently
admits facts the world had not yet published.

This module builds k-hop neighbourhoods under three regimes and measures the
difference. If report-time masking barely changes the neighbourhood, the
contribution is weak and should be dropped -- so this is written to be able to
falsify it.

  none        every edge (what an atemporal GNN sees)
  event_time  edges whose event_time <= tau
  report_time edges whose available_date <= tau   <- the correct one

Read-only. Output goes next to this file.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from datetime import date
from pathlib import Path

VIEW = Path(os.environ.get("TKG_VIEW", r"C:\Users\HP\Documents\Claude\news-tkg-modeling-view\view"))
HERE = Path(__file__).parent


def read(name: str) -> list[dict]:
    with (VIEW / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def d(s):
    try:
        return date.fromisoformat(str(s)[:10])
    except Exception:
        return None


events = {r["event_id"]: r for r in read("nodes_event.tsv")}
edges = read("edges.tsv")

# Undirected adjacency for message passing, with the edge's two times attached.
#
# COMPARABILITY: event_time is empty on structural edges (publishedBy, and any
# edge whose subject is an article rather than an event), while available_date is
# almost always present. Comparing the regimes over all edges would measure that
# missingness rather than the temporal criterion. So the regime comparison is
# restricted to edges carrying BOTH times; edges missing either are excluded from
# every regime alike, and counted separately.
adj: dict[str, list[tuple]] = defaultdict(list)
both_times = 0
missing_event_time = 0
for i, e in enumerate(edges):
    et, at = d(e["event_time"]), d(e["available_date"])
    if et is None or at is None:
        missing_event_time += 1
        continue
    both_times += 1
    adj[e["src_id"]].append((e["dst_id"], i, et, at))
    adj[e["dst_id"]].append((e["src_id"], i, et, at))


def neighbourhood(seed: str, tau: date, regime: str, hops: int = 2) -> tuple[set, set]:
    """Return (nodes, edge_indices) reachable within `hops` under `regime`."""
    seen_n, seen_e = {seed}, set()
    frontier = {seed}
    for _ in range(hops):
        nxt = set()
        for u in frontier:
            for v, ei, et, at in adj.get(u, ()):
                if regime == "event_time":
                    if et is None or et > tau:
                        continue
                elif regime == "report_time":
                    if at is None or at > tau:
                        continue
                seen_e.add(ei)
                if v not in seen_n:
                    seen_n.add(v)
                    nxt.add(v)
        frontier = nxt
        if not frontier:
            break
    return seen_n, seen_e


# ---------------------------------------------------------------- measurement
rows = []
for eid, ev in events.items():
    tau = d(ev["available_date"]) or d(ev["report_time"])
    if tau is None:
        continue
    n_all, e_all = neighbourhood(eid, tau, "none")
    n_evt, e_evt = neighbourhood(eid, tau, "event_time")
    n_rep, e_rep = neighbourhood(eid, tau, "report_time")
    rows.append({
        "event_id": eid, "tau": str(tau),
        "nodes_none": len(n_all), "nodes_event_time": len(n_evt), "nodes_report_time": len(n_rep),
        "edges_none": len(e_all), "edges_event_time": len(e_evt), "edges_report_time": len(e_rep),
        # leakage = admitted by the looser regime but not observable at tau
        "leak_edges_vs_none": len(e_all - e_rep),
        "leak_edges_vs_event_time": len(e_evt - e_rep),
        "leak_nodes_vs_event_time": len(n_evt - n_rep),
    })


def stats(key):
    v = sorted(r[key] for r in rows)
    n = len(v)
    return {"median": v[n // 2], "p25": v[n // 4], "p75": v[3 * n // 4], "max": v[-1],
            "mean": round(sum(v) / n, 2), "nonzero_share": round(sum(1 for x in v if x > 0) / n, 4)}


report = {
    "graph": str(VIEW),
    "events_measured": len(rows),
    "edges_with_both_times": both_times,
    "edges_excluded_missing_a_time": missing_event_time,
    "hops": 2,
    "prediction_time": "each event's own available_date",
    "neighbourhood_size": {k: stats(k) for k in
                           ("nodes_none", "nodes_event_time", "nodes_report_time",
                            "edges_none", "edges_event_time", "edges_report_time")},
    "leakage": {k: stats(k) for k in
                ("leak_edges_vs_none", "leak_edges_vs_event_time", "leak_nodes_vs_event_time")},
}

# ------------------------------------------------------------------ self-test
# A correct report_time neighbourhood must contain no edge whose availability is
# after tau. Verify directly rather than trusting the construction.
violations = 0
checked = 0
for r in rows[:200]:
    tau = d(r["tau"])
    _, e_rep = neighbourhood(r["event_id"], tau, "report_time")
    for ei in e_rep:
        at = d(edges[ei]["available_date"])
        checked += 1
        if at is None or at > tau:
            violations += 1
report["self_test"] = {"edges_checked": checked, "violations": violations,
                       "passes": violations == 0}

(HERE / "temporal-masking-report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"events measured: {len(rows)}   2-hop neighbourhoods\n")
print(f"{'regime':14s} {'nodes med':>10s} {'nodes p75':>10s} {'edges med':>10s} {'edges p75':>10s}")
for reg in ("none", "event_time", "report_time"):
    n, e = report["neighbourhood_size"][f"nodes_{reg}"], report["neighbourhood_size"][f"edges_{reg}"]
    print(f"{reg:14s} {n['median']:10d} {n['p75']:10d} {e['median']:10d} {e['p75']:10d}")
print()
for k in ("leak_edges_vs_none", "leak_edges_vs_event_time", "leak_nodes_vs_event_time"):
    s = report["leakage"][k]
    print(f"{k:28s} median {s['median']:4d}  p75 {s['p75']:4d}  max {s['max']:5d}  "
          f"affected {s['nonzero_share']:.0%} of events")
print()
print(f"self-test: {report['self_test']['edges_checked']} edges checked, "
      f"{report['self_test']['violations']} violations -> "
      f"{'PASS' if report['self_test']['passes'] else 'FAIL'}")
