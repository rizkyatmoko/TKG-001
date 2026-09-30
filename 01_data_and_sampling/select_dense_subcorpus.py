"""Select the extraction subset that maximises graph density per compute-hour.

A commodity filter (e.g. "all rice") is not optimal: 464 rice articles spread
thinly over 5 years and 30 districts still leave every commodity x district x
quarter cell nearly empty, which is the condition that produced all three
failure modes:

  * no negative reaches the positive's hardness      (needs same commodity+place+period)
  * responses co-reported with disruptions           (needs several articles per cell)
  * one policy takes 42% of implementsPolicy edges   (needs target diversity)

So select *cells*, not commodities. Score each (commodity, district, 90-day
window) bucket by how many queued articles it holds and how many distinct
publisher families cover it, then take the densest buckets until the budget is
spent.

Read-only. Writes the selected id list next to this file.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(r"C:\Users\HP\Documents\ChatGPT\News TKG")
QUEUE = ROOT / "data" / "link_prediction" / "expansion-v3" / "article_queue.jsonl"
CKPT = ROOT / ".artifacts" / "extraction-expansion-v3" / "qwen8b-verbose-primary"
HERE = Path(__file__).parent

SEC_PER_ARTICLE = 130
BUDGET_HOURS = 16.0
BUDGET = int(BUDGET_HOURS * 3600 / SEC_PER_ARTICLE)

rows = [json.loads(l) for l in QUEUE.open(encoding="utf-8")]
done = {p.stem for p in CKPT.glob("ann_*.json")}


def listify(v):
    if not v:
        return []
    return v if isinstance(v, list) else [v]


def window(s: str) -> str:
    d = date.fromisoformat(s[:10])
    q = (d.month - 1) // 3 + 1
    return f"{d.year}Q{q}"


# ---- build cells -----------------------------------------------------------
cells: dict[tuple, list[dict]] = defaultdict(list)
for r in rows:
    if not r.get("pub_date"):
        continue
    w = window(r["pub_date"])
    for com in listify(r.get("heuristic_commodities")) or ["_none_"]:
        for loc in listify(r.get("heuristic_locations")) or ["_none_"]:
            cells[(com, loc, w)].append(r)


def score(members: list[dict]) -> tuple:
    fams = {m.get("publisher_family") for m in members}
    n = len(members)
    ndone = sum(1 for m in members if m["article_id"] in done)
    # density x cross-source coverage; cells already partly extracted are cheaper
    return (min(n, 40) * min(len(fams), 4), n, ndone)


ranked = sorted(cells.items(), key=lambda kv: -score(kv[1])[0])

# Quotas. An unconstrained densest-first pass selects a flood monoculture: the
# biggest cells are all commodity-less DisasterEvent clusters. That maximises
# hard negatives but deletes the commodity dimension and leaves implementsPolicy
# with nothing, so cap it and reserve a share for commodity-bearing cells.
COMMODITY_QUOTA = int(BUDGET * 0.45)         # articles carrying >=1 commodity signal
DISASTER_CAP = int(BUDGET * 0.70)


def has_commodity(m) -> bool:
    return bool(listify(m.get("heuristic_commodities")))


def is_disaster(m) -> bool:
    t = listify(m.get("heuristic_disruption_types"))
    return t == ["DisasterEvent"] or (t and set(t) <= {"DisasterEvent", "WeatherDisruption"})


selected: dict[str, dict] = {}
chosen_cells = []
n_commodity = n_disaster = 0

# two passes: commodity-bearing cells first (to guarantee the quota), then the rest
for phase in ("commodity", "any"):
    for key, members in ranked:
        if len(selected) >= BUDGET:
            break
        if phase == "commodity" and key[0] == "_none_":
            continue
        if phase == "commodity" and n_commodity >= COMMODITY_QUOTA:
            break
        if len(members) < 4:                 # a cell smaller than this cannot host hard negatives
            continue
        fams = {m.get("publisher_family") for m in members}
        if len(fams) < 2:                    # needs >=2 sources for cross-document structure
            continue
        added = 0
        for m in members:
            aid = m["article_id"]
            if aid in done or aid in selected:
                continue
            if len(selected) >= BUDGET:
                break
            if is_disaster(m) and n_disaster >= DISASTER_CAP:
                continue
            selected[aid] = m
            added += 1
            n_commodity += has_commodity(m)
            n_disaster += is_disaster(m)
        if added:
            chosen_cells.append({"cell": "|".join(map(str, key)), "articles": len(members),
                                 "publisher_families": len(fams), "newly_selected": added,
                                 "phase": phase})

ids = list(selected)   # densest-first: preserve selection order, do NOT sort
(HERE / "ids-dense-subcorpus.txt").write_text(",".join(ids), encoding="utf-8")

com = Counter(c for m in selected.values() for c in listify(m.get("heuristic_commodities")))
fam = Counter(m.get("publisher_family") for m in selected.values())
yr = Counter(m["pub_date"][:4] for m in selected.values())
roles = Counter(m.get("roles") if isinstance(m.get("roles"), str) else
                "|".join(sorted(listify(m.get("roles")))) for m in selected.values())

# how dense are the selected cells, vs a plain rice filter of the same size?
rice_ids = [r["article_id"] for r in rows
            if "rice" in listify(r.get("heuristic_commodities")) and r["article_id"] not in done][:BUDGET]
def cell_profile(idset):
    c = defaultdict(int)
    idset = set(idset)
    for r in rows:
        if r["article_id"] not in idset or not r.get("pub_date"):
            continue
        w = window(r["pub_date"])
        for cm in listify(r.get("heuristic_commodities")) or ["_none_"]:
            for lc in listify(r.get("heuristic_locations")) or ["_none_"]:
                c[(cm, lc, w)] += 1
    v = sorted(c.values(), reverse=True)
    return {"cells": len(v), "median_articles_per_cell": v[len(v)//2] if v else 0,
            "cells_with_>=4": sum(1 for x in v if x >= 4),
            "cells_with_>=8": sum(1 for x in v if x >= 8), "max_cell": v[0] if v else 0}

report = {
    "budget_hours": BUDGET_HOURS, "seconds_per_article": SEC_PER_ARTICLE,
    "budget_articles": BUDGET, "selected": len(ids),
    "already_checkpointed": len(done),
    "selection_rule": "cells of (commodity, district, quarter) with >=4 queued articles "
                      "and >=2 publisher families, taken densest-first",
    "cells_used": len(chosen_cells),
    "top_cells": chosen_cells[:15],
    "commodity_mix": dict(com.most_common(10)),
    "publisher_mix": dict(fam.most_common()),
    "year_mix": dict(sorted(yr.items())),
    "role_mix": dict(roles.most_common()),
    "density_selected": cell_profile(ids),
    "density_plain_rice_same_budget": cell_profile(rice_ids),
}
(HERE / "dense-subcorpus-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"budget {BUDGET} articles ({BUDGET_HOURS}h at {SEC_PER_ARTICLE}s)   selected {len(ids)}")
print(f"cells used {len(chosen_cells)}")
print(f"\ncommodity mix: {dict(com.most_common(8))}")
print(f"publisher mix: {dict(fam.most_common())}")
print(f"years        : {dict(sorted(yr.items()))}")
print(f"roles        : {dict(roles.most_common())}")
print("\ncell density comparison (articles per commodity x district x quarter):")
for name, prof in (("dense selection", report["density_selected"]),
                   ("plain rice filter", report["density_plain_rice_same_budget"])):
    print(f"  {name:18s} cells={prof['cells']:4d}  median={prof['median_articles_per_cell']:2d}  "
          f">=4: {prof['cells_with_>=4']:3d}  >=8: {prof['cells_with_>=8']:3d}  max={prof['max_cell']}")
print(f"\nwrote ids-dense-subcorpus.txt ({len(ids)} ids)")
