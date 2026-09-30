"""Minimal relational GCN for the FoodJatim TKG, in numpy.

No torch is installed and the graph is small (375 nodes now, ~750 projected), so
a hand-written R-GCN is both sufficient and faster to run than installing a
framework. The model is deliberately small: with 25-32 test items no
architecture claim is supportable, so the model is an instrument for the paired
ablations, not a contribution.

Layer (Schlichtkrull et al., 2018), with row-normalised per-relation adjacency:

    Z = X W_0 + sum_r  A_r X W_r
    H = relu(Z)

Gradients, derived by hand and verified against finite differences in the
self-test at the bottom:

    dW_0 = X^T dZ
    dW_r = (A_r X)^T dZ
    dX   = dZ W_0^T + sum_r A_r^T (dZ W_r^T)

Ablation switches are first-class, because the ablation table is the output that
matters:

    relations="typed"       per-relation weights (full R-GCN)
    relations="single"      one shared weight for all edges (homogeneous GCN)
    relations="none"        no message passing (MLP on node features)

Requires numpy. Run with the system interpreter, not the project venv.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

import os

# Override with TKG_VIEW to point at a different export. `view/` is the
# RDF-derived 300-article graph; `view-full/` is built straight from the raw
# annotations and covers all 853 articles.
VIEW = Path(os.environ.get(
    "TKG_VIEW", r"C:\Users\HP\Documents\Claude\news-tkg-modeling-view\view"))
HERE = Path(__file__).parent


# ----------------------------------------------------------------- data ----
def read(name: str) -> list[dict]:
    with (VIEW / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def d(s):
    try:
        return date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def build_graph(mask: str = "report_time", tau: date | None = None,
                require_both_times: bool = True):
    """Return (node_index, X, adj_by_relation, meta).

    mask: 'none' | 'event_time' | 'report_time'; tau is the prediction time.
    Edges failing the mask are dropped, so the same code serves ablation A3.
    """
    events = read("nodes_event.tsv")
    edges = read("edges.tsv")

    nodes, kinds = [], {}
    for r in events:
        nodes.append(r["event_id"])
        kinds[r["event_id"]] = r
    for f, key, tag in (("nodes_policy.tsv", "policy_id", "policy"),
                        ("nodes_actor.tsv", "actor_id", "actor"),
                        ("nodes_location.tsv", "location_id", "location"),
                        ("nodes_commodity.tsv", "commodity_id", "commodity"),
                        ("nodes_publisher.tsv", "publisher_id", "publisher"),
                        ("nodes_article.tsv", "article_id", "article")):
        for r in read(f):
            nodes.append(r[key])
            kinds[r[key]] = {"_type": tag}
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    # features: node-type one-hot | event-kind | commodity | normalised year
    types = ["event", "policy", "actor", "location", "commodity", "publisher", "article"]
    coms = sorted({r["commodity_id"] for r in read("nodes_commodity.tsv")})
    F = len(types) + 2 + len(coms) + 1
    X = np.zeros((N, F), dtype=np.float64)
    for n, i in idx.items():
        r = kinds[n]
        t = r.get("_type", "event")
        X[i, types.index(t)] = 1.0
        if t == "event":
            st = r.get("all_types", "")
            X[i, len(types)] = 1.0 if "Disruption" in st or any(
                k in st for k in ("SupplyShortage", "DisasterEvent", "WeatherDisruption",
                                  "PestDiseaseEvent", "FoodSafetyIncident")) else 0.0
            X[i, len(types) + 1] = 1.0 - X[i, len(types)]
            c = r.get("commodity_id", "")
            if c in coms:
                X[i, len(types) + 2 + coms.index(c)] = 1.0
            dt = d(r.get("event_date"))
            if dt:
                X[i, -1] = (dt.year - 2021) / 6.0

    rels = sorted({e["relation"] for e in edges})
    buckets = defaultdict(list)
    kept = dropped = excluded_missing = 0
    for e in edges:
        # COMPARABILITY for ablation A3: event_time is absent on structural edges
        # (publishedBy, article-subject edges) while available_date is not, so an
        # unrestricted comparison of the two regimes measures missingness rather
        # than the temporal criterion -- and makes report_time look *larger* than
        # event_time. Restrict every regime to edges carrying both times so the
        # only difference between conditions is the masking rule itself.
        if require_both_times:
            if d(e["event_time"]) is None or d(e["available_date"]) is None:
                excluded_missing += 1
                continue
        if mask != "none" and tau is not None:
            t = d(e["event_time"]) if mask == "event_time" else d(e["available_date"])
            if t is None or t > tau:
                dropped += 1
                continue
        if e["src_id"] not in idx or e["dst_id"] not in idx:
            dropped += 1
            continue
        kept += 1
        buckets[e["relation"]].append((idx[e["src_id"]], idx[e["dst_id"]]))

    adj = {}
    for r in rels:
        A = np.zeros((N, N), dtype=np.float64)
        for s, o in buckets.get(r, ()):
            A[o, s] = 1.0            # messages flow src -> dst
            A[s, o] = 1.0            # and back; the graph is treated as undirected
        deg = A.sum(1, keepdims=True)
        deg[deg == 0] = 1.0
        adj[r] = A / deg
    return idx, X, adj, {"N": N, "F": F, "relations": rels,
                         "edges_kept": kept, "edges_dropped": dropped,
                         "edges_excluded_missing_time": excluded_missing,
                         "require_both_times": require_both_times}


# ---------------------------------------------------------------- model ----
class RGCN:
    def __init__(self, F, H, rels, out=1, relations="typed", seed=0):
        rng = np.random.default_rng(seed)
        self.mode = relations
        self.rels = list(rels) if relations == "typed" else (["_all_"] if relations == "single" else [])
        sc1, sc2 = np.sqrt(2.0 / F), np.sqrt(2.0 / H)
        self.W0 = rng.normal(0, sc1, (F, H))
        self.Wr = {r: rng.normal(0, sc1, (F, H)) for r in self.rels}
        self.U0 = rng.normal(0, sc2, (H, H))
        self.Ur = {r: rng.normal(0, sc2, (H, H)) for r in self.rels}
        self.wo = rng.normal(0, sc2, (H, out))
        self.bo = np.zeros(out)

    def _agg(self, adj):
        if self.mode == "typed":
            return adj
        if self.mode == "single":
            A = sum(adj.values())
            deg = A.sum(1, keepdims=True); deg[deg == 0] = 1.0
            return {"_all_": A / deg}
        return {}

    def forward(self, X, adj):
        A = self._agg(adj)
        self._A = A
        Z1 = X @ self.W0 + sum(A[r] @ X @ self.Wr[r] for r in self.rels) if self.rels else X @ self.W0
        H1 = np.maximum(Z1, 0)
        Z2 = H1 @ self.U0 + sum(A[r] @ H1 @ self.Ur[r] for r in self.rels) if self.rels else H1 @ self.U0
        H2 = np.maximum(Z2, 0)
        logit = H2 @ self.wo + self.bo
        self._cache = (X, Z1, H1, Z2, H2)
        return logit

    def forward_embed(self, X, adj):
        """Node embeddings H2, for tasks with their own head (e.g. edge classification)."""
        A = self._agg(adj)
        self._A = A
        Z1 = X @ self.W0 + sum(A[r] @ X @ self.Wr[r] for r in self.rels) if self.rels else X @ self.W0
        H1 = np.maximum(Z1, 0)
        Z2 = H1 @ self.U0 + sum(A[r] @ H1 @ self.Ur[r] for r in self.rels) if self.rels else H1 @ self.U0
        H2 = np.maximum(Z2, 0)
        self._cache = (X, Z1, H1, Z2, H2)
        return H2

    def backward_from_dH2(self, dH2):
        """Backprop starting from the gradient w.r.t. node embeddings."""
        X, Z1, H1, Z2, H2 = self._cache
        A = self._A
        g = {"wo": np.zeros_like(self.wo), "bo": np.zeros_like(self.bo)}
        dZ2 = dH2 * (Z2 > 0)
        g["U0"] = H1.T @ dZ2
        g["Ur"] = {r: (A[r] @ H1).T @ dZ2 for r in self.rels}
        dH1 = dZ2 @ self.U0.T + sum(A[r].T @ (dZ2 @ self.Ur[r].T) for r in self.rels) if self.rels \
            else dZ2 @ self.U0.T
        dZ1 = dH1 * (Z1 > 0)
        g["W0"] = X.T @ dZ1
        g["Wr"] = {r: (A[r] @ X).T @ dZ1 for r in self.rels}
        return g

    def backward(self, dlogit):
        X, Z1, H1, Z2, H2 = self._cache
        A = self._A
        g = {}
        g["wo"] = H2.T @ dlogit
        g["bo"] = dlogit.sum(0)
        dH2 = dlogit @ self.wo.T
        dZ2 = dH2 * (Z2 > 0)
        g["U0"] = H1.T @ dZ2
        g["Ur"] = {r: (A[r] @ H1).T @ dZ2 for r in self.rels}
        dH1 = dZ2 @ self.U0.T + sum(A[r].T @ (dZ2 @ self.Ur[r].T) for r in self.rels) if self.rels \
            else dZ2 @ self.U0.T
        dZ1 = dH1 * (Z1 > 0)
        g["W0"] = X.T @ dZ1
        g["Wr"] = {r: (A[r] @ X).T @ dZ1 for r in self.rels}
        return g

    def params(self):
        p = {"W0": self.W0, "U0": self.U0, "wo": self.wo, "bo": self.bo}
        for r in self.rels:
            p[f"Wr::{r}"] = self.Wr[r]
            p[f"Ur::{r}"] = self.Ur[r]
        return p

    def grads_flat(self, g):
        out = {"W0": g["W0"], "U0": g["U0"], "wo": g["wo"], "bo": g["bo"]}
        for r in self.rels:
            out[f"Wr::{r}"] = g["Wr"][r]
            out[f"Ur::{r}"] = g["Ur"][r]
        return out

    def step(self, g, lr):
        f = self.grads_flat(g)
        for k, p in self.params().items():
            p -= lr * f[k]


def bce(logit, y, m):
    """Masked binary cross-entropy; returns (loss, dlogit)."""
    p = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
    n = max(1, int(m.sum()))
    loss = -(m * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))).sum() / n
    return loss, (m * (p - y)) / n


# ------------------------------------------------------------ self-tests ----
def _gradient_check() -> dict:
    rng = np.random.default_rng(3)
    N, F, H = 12, 5, 4
    X = rng.normal(size=(N, F))
    rels = ["a", "b"]
    adj = {}
    for r in rels:
        A = (rng.random((N, N)) < 0.25).astype(float)
        np.fill_diagonal(A, 0)
        dg = A.sum(1, keepdims=True); dg[dg == 0] = 1
        adj[r] = A / dg
    y = (rng.random((N, 1)) < 0.5).astype(float)
    m = np.ones((N, 1))
    out = {}
    for mode in ("typed", "single", "none"):
        net = RGCN(F, H, rels, relations=mode, seed=1)
        logit = net.forward(X, adj)
        loss, dl = bce(logit, y, m)
        g = net.grads_flat(net.backward(dl))
        worst = 0.0
        for k, P in net.params().items():
            G = g[k]
            for _ in range(12):                       # sample entries
                i = tuple(rng.integers(0, s) for s in P.shape)
                eps = 1e-6
                orig = P[i]
                P[i] = orig + eps
                lp, _ = bce(net.forward(X, adj), y, m)
                P[i] = orig - eps
                lm, _ = bce(net.forward(X, adj), y, m)
                P[i] = orig
                num = (lp - lm) / (2 * eps)
                den = max(1e-8, abs(num) + abs(G[i]))
                worst = max(worst, abs(num - G[i]) / den)
        out[mode] = {"max_relative_error": float(worst), "pass": worst < 1e-4}
    return out


def _learnability() -> dict:
    """The model must fit a label that is only recoverable through the graph."""
    rng = np.random.default_rng(7)
    N, F, H = 60, 4, 12
    X = rng.normal(size=(N, F)) * 0.1
    A = np.zeros((N, N))
    grp = rng.integers(0, 2, N)
    for i in range(N):
        for j in range(N):
            if i != j and grp[i] == grp[j] and rng.random() < 0.3:
                A[i, j] = 1.0
    marker = np.zeros(N)
    for gv in (0, 1):
        members = np.where(grp == gv)[0]
        marker[members[0]] = 1.0 if gv == 1 else -1.0
    X[:, 0] = marker                     # only 2 nodes carry the signal
    dg = A.sum(1, keepdims=True); dg[dg == 0] = 1
    adj = {"link": A / dg}
    y = grp.reshape(-1, 1).astype(float)
    m = np.ones((N, 1))
    res = {}
    for mode in ("typed", "none"):
        net = RGCN(F, H, ["link"], relations=mode, seed=2)
        for _ in range(600):
            logit = net.forward(X, adj)
            loss, dl = bce(logit, y, m)
            net.step(net.backward(dl), lr=0.5)
        p = 1 / (1 + np.exp(-net.forward(X, adj)))
        acc = float((((p > 0.5).astype(float) == y).mean()))
        res[mode] = round(acc, 3)
    res["graph_beats_mlp"] = bool(res["typed"] > res["none"] + 0.05)
    return res


if __name__ == "__main__":
    gc = _gradient_check()
    ln = _learnability()
    print("gradient check (analytic vs finite differences):")
    for mode, r in gc.items():
        print(f"   {'PASS' if r['pass'] else 'FAIL'}  relations={mode:7s} "
              f"max rel. err {r['max_relative_error']:.2e}")
    print(f"\nlearnability (label recoverable only via message passing):")
    print(f"   typed R-GCN accuracy {ln['typed']}   feature-only MLP {ln['none']}   "
          f"graph helps: {ln['graph_beats_mlp']}")

    idx, X, adj, meta = build_graph(mask="none")
    print(f"\nreal graph: N={meta['N']} F={meta['F']} relations={len(meta['relations'])} "
          f"edges kept={meta['edges_kept']}")
    net = RGCN(meta["F"], 32, meta["relations"], relations="typed", seed=0)
    out = net.forward(X, adj)
    print(f"forward pass OK: logits {out.shape}, finite={np.isfinite(out).all()}")

    Path(HERE / "rgcn-selftest.json").write_text(json.dumps(
        {"gradient_check": gc, "learnability": ln, "graph": meta}, indent=2, default=str), encoding="utf-8")
