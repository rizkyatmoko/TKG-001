"""Relation-type classification: the one task the corpus currently supports.

Given a pair of nodes known to be linked, predict which of the four relation
types holds (respondsTo, implementsPolicy, citesPolicy, reportedCauseOf). This
task is immune to the failure modes that block the others: there is no candidate
ranking, so the hardness degeneracy does not arise, and no forecasting, so
co-reporting is irrelevant. At 222 labels it clears the power gate.

It is however trivially corruptible, so three guards are enforced and the first
is empirically tested.

G1  Target-edge masking. The edge being classified is removed from the
    message-passing graph for its fold. Without this the model can read the
    label off the adjacency it is standing on. `--leak-test` disables the guard
    to show what that is worth.

G2  No evidence text. `evidence_quote` is never a feature. The labels were
    assigned by a model reading those quotes, so using them would measure
    whether we can re-read an annotator's mind, not whether graph structure
    carries relation type.

G3  Chronological split by availability, so no fold sees a later fold's edges.

Requires numpy; run with the system interpreter.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from rgcn import RGCN, VIEW, read, d

HERE = Path(__file__).parent
TARGETS = ["respondsTo", "implementsPolicy", "citesPolicy", "reportedCauseOf"]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def macro_f1(y_true, y_pred, k):
    fs = []
    for c in range(k):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(fs)), fs


def build_dataset(mask_targets: bool, fold_edges: set[int] | None = None,
                  event_only: bool = False):
    """Nodes, features, adjacency (with target edges optionally removed), and labels."""
    events = read(VIEW / "nodes_event.tsv")
    edges = read(VIEW / "edges.tsv")

    nodes, kinds = [], {}
    for r in events:
        nodes.append(r["event_id"]); kinds[r["event_id"]] = r
    for f, key, tag in (("nodes_policy.tsv", "policy_id", "policy"),
                        ("nodes_actor.tsv", "actor_id", "actor"),
                        ("nodes_location.tsv", "location_id", "location"),
                        ("nodes_commodity.tsv", "commodity_id", "commodity"),
                        ("nodes_publisher.tsv", "publisher_id", "publisher"),
                        ("nodes_article.tsv", "article_id", "article")):
        for r in read(VIEW / f):
            nodes.append(r[key]); kinds[r[key]] = {"_type": tag}
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    # G2: features are node type, event kind, commodity, stage, year. No text.
    types = ["event", "policy", "actor", "location", "commodity", "publisher", "article"]
    coms = sorted({r["commodity_id"] for r in read(VIEW / "nodes_commodity.tsv")})
    stages = sorted({r.get("stage_id", "") for r in events})
    F = len(types) + 2 + len(coms) + len(stages) + 1
    X = np.zeros((N, F))
    DIS = ("SupplyShortage", "DisasterEvent", "WeatherDisruption", "PestDiseaseEvent",
           "FoodSafetyIncident", "ProductionDisruption", "DistributionDisruption")
    for n, i in idx.items():
        r = kinds[n]
        t = r.get("_type", "event")
        X[i, types.index(t)] = 1.0
        if t == "event":
            st = r.get("all_types", "")
            isdis = any(k in st for k in DIS)
            X[i, len(types)] = float(isdis)
            X[i, len(types) + 1] = float(not isdis)
            c = r.get("commodity_id", "")
            if c in coms:
                X[i, len(types) + 2 + coms.index(c)] = 1.0
            s = r.get("stage_id", "")
            if s in stages:
                X[i, len(types) + 2 + len(coms) + stages.index(s)] = 1.0
            dt = d(r.get("event_date"))
            if dt:
                X[i, -1] = (dt.year - 2021) / 6.0

    # labels
    samples = []
    for ei, e in enumerate(edges):
        if e["relation"] not in TARGETS:
            continue
        # 87% of the 4-class labels are determined by (src_type, dst_type) alone:
        # (event,policy)->implementsPolicy, (article,policy)->citesPolicy. Only
        # event->event is genuinely ambiguous, so that is the real task.
        if event_only and not (e["src_type"] == "event" and e["dst_type"] == "event"):
            continue
        if e["src_id"] not in idx or e["dst_id"] not in idx:
            continue
        samples.append({"edge_ix": ei, "src": idx[e["src_id"]], "dst": idx[e["dst_id"]],
                        "y": TARGETS.index(e["relation"]),
                        "avail": d(e["available_date"]) or d(e["report_time"]) or date(2021, 1, 1)})

    # adjacency; G1 removes the fold's target edges from message passing
    rels = sorted({e["relation"] for e in edges})
    buckets = defaultdict(list)
    removed = 0
    for ei, e in enumerate(edges):
        if mask_targets and fold_edges is not None and ei in fold_edges:
            removed += 1
            continue
        if e["src_id"] in idx and e["dst_id"] in idx:
            buckets[e["relation"]].append((idx[e["src_id"]], idx[e["dst_id"]]))
    adj = {}
    for r in rels:
        A = np.zeros((N, N))
        for s, o in buckets.get(r, ()):
            A[o, s] = 1.0; A[s, o] = 1.0
        dg = A.sum(1, keepdims=True); dg[dg == 0] = 1.0
        adj[r] = A / dg
    return idx, X, adj, rels, samples, removed


def run(mode: str, mask_targets: bool, seed: int, epochs: int, lr: float, hidden: int,
        event_only: bool = False):
    _, X, _, rels, samples, _ = build_dataset(mask_targets=False, event_only=event_only)
    samples.sort(key=lambda s: s["avail"])                     # G3 chronological
    n = len(samples)
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    folds = {"train": samples[:n_tr], "val": samples[n_tr:n_tr + n_va], "test": samples[n_tr + n_va:]}

    # G1: message passing excludes every labelled target edge, so no fold can
    # read its own answer, and train edges cannot leak into val/test either.
    held = {s["edge_ix"] for s in samples}
    _, X, adj, rels, _, removed = build_dataset(mask_targets=mask_targets, fold_edges=held,
                                                event_only=event_only)

    K = len(TARGETS)   # label space kept at 4 so indices stay stable
    rng = np.random.default_rng(seed)
    net = RGCN(X.shape[1], hidden, rels, out=hidden, relations=mode, seed=seed)
    We = rng.normal(0, np.sqrt(2.0 / (3 * hidden)), (3 * hidden, K))
    be = np.zeros(K)

    def edge_feats(H, fold):
        s = np.array([f["src"] for f in fold]); o = np.array([f["dst"] for f in fold])
        return np.hstack([H[s], H[o], H[s] * H[o]]), s, o

    ytr = np.array([f["y"] for f in folds["train"]])
    best = (-1, None)
    for ep in range(epochs):
        H = net.forward_embed(X, adj)
        Z, si, oi = edge_feats(H, folds["train"])
        logits = Z @ We + be
        P = softmax(logits)
        Y = np.zeros_like(P); Y[np.arange(len(ytr)), ytr] = 1.0
        dlog = (P - Y) / len(ytr)

        gWe = Z.T @ dlog
        gbe = dlog.sum(0)
        dZ = dlog @ We.T
        dH = np.zeros_like(H)
        h = hidden
        np.add.at(dH, si, dZ[:, :h] + dZ[:, 2 * h:] * H[oi])
        np.add.at(dH, oi, dZ[:, h:2 * h] + dZ[:, 2 * h:] * H[si])
        g = net.backward_from_dH2(dH)
        net.step(g, lr)
        We -= lr * gWe; be -= lr * gbe

        if (ep + 1) % 25 == 0:
            Hv = net.forward_embed(X, adj)
            Zv, _, _ = edge_feats(Hv, folds["val"])
            yv = np.array([f["y"] for f in folds["val"]])
            pv = softmax(Zv @ We + be).argmax(1)
            f1, _ = macro_f1(yv, pv, K)
            if f1 > best[0]:
                best = (f1, (net.params().copy(), We.copy(), be.copy()))

    H = net.forward_embed(X, adj)
    out = {}
    for name in ("val", "test"):
        Z, _, _ = edge_feats(H, folds[name])
        y = np.array([f["y"] for f in folds[name]])
        p = softmax(Z @ We + be).argmax(1)
        f1, per = macro_f1(y, p, K)
        out[name] = {"n": len(y), "accuracy": round(float((p == y).mean()), 4),
                     "macro_f1": round(f1, 4),
                     "per_class_f1": {TARGETS[i]: round(per[i], 3) for i in range(K)},
                     "correct": (p == y).astype(int).tolist()}
    out["edges_removed_from_message_passing"] = removed
    out["class_counts"] = dict(Counter(TARGETS[s["y"]] for s in samples))
    maj = Counter(s["y"] for s in folds["train"]).most_common(1)[0][0]
    yte = np.array([f["y"] for f in folds["test"]])
    out["majority_baseline_test_acc"] = round(float((yte == maj).mean()), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--event-only", action="store_true",
                    help="restrict to event->event pairs: respondsTo vs reportedCauseOf")
    ap.add_argument("--leak-test", action="store_true",
                    help="disable G1 to demonstrate what target-edge masking is worth")
    args = ap.parse_args()

    report = {"guards": {"G1_target_edge_masking": not args.leak_test,
                         "G2_no_evidence_text": True,
                         "G3_chronological_split": True},
              "runs": {}}
    for mode in ("typed", "single", "none"):
        accs, f1s = [], []
        for s in range(args.seeds):
            r = run(mode, mask_targets=not args.leak_test, seed=s,
                    epochs=args.epochs, lr=args.lr, hidden=args.hidden,
                    event_only=args.event_only)
            accs.append(r["test"]["accuracy"]); f1s.append(r["test"]["macro_f1"])
            last = r
        report["runs"][mode] = {
            "test_accuracy_mean": round(float(np.mean(accs)), 4),
            "test_accuracy_sd": round(float(np.std(accs)), 4),
            "test_macro_f1_mean": round(float(np.mean(f1s)), 4),
            "per_seed_accuracy": accs, "per_seed_macro_f1": f1s,
            "test_n": last["test"]["n"], "per_class_f1": last["test"]["per_class_f1"],
            "majority_baseline": last["majority_baseline_test_acc"],
            "edges_removed": last["edges_removed_from_message_passing"],
            "class_counts": last["class_counts"],
        }

    tag = "leak" if args.leak_test else "guarded"
    (HERE / f"relation-classification-{tag}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"guards: G1 target-edge masking = {not args.leak_test}, "
          f"G2 no evidence text = True, G3 chronological = True")
    r0 = report["runs"]["typed"]
    print(f"labels: {r0['class_counts']}   test n={r0['test_n']}   "
          f"majority baseline {r0['majority_baseline']:.3f}")
    print(f"edges removed from message passing: {r0['edges_removed']}\n")
    print(f"{'model':10s} {'test acc':>18s} {'macro F1':>10s}")
    for mode in ("typed", "single", "none"):
        r = report["runs"][mode]
        print(f"{mode:10s} {r['test_accuracy_mean']:.3f} +/- {r['test_accuracy_sd']:.3f}"
              f"   {r['test_macro_f1_mean']:10.3f}")
    print(f"\nper-class F1 (typed): {report['runs']['typed']['per_class_f1']}")


if __name__ == "__main__":
    main()
