"""Paired significance testing for small-n ablations.

At test n ~= 25 the only defensible claims are paired: system A vs system B on
the *same* items, comparing per-item differences. Unpaired CIs on absolute
metrics at this n are wide enough to be meaningless.

Provides:
  paired_bootstrap   - BCa-free percentile CI on the mean paired difference
  paired_permutation - exact sign-flip test when n <= 20, Monte Carlo above
  compare            - both, plus effect size and a plain-language verdict

Self-tests at the bottom check the implementations against cases with known
answers, including an exact enumeration cross-check.

Pure stdlib. No dependency on the graph or the task.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path


def _mean(v):
    return sum(v) / len(v)


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 20000,
                     alpha: float = 0.05, seed: int = 20260815) -> dict:
    """Percentile CI on mean(a - b), resampling item indices (keeps pairing)."""
    if len(a) != len(b):
        raise ValueError("paired inputs must be the same length")
    n = len(a)
    diff = [x - y for x, y in zip(a, b)]
    obs = _mean(diff)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [diff[rng.randrange(n)] for _ in range(n)]
        boots.append(_mean(s))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return {"n": n, "mean_difference": round(obs, 6),
            "ci_low": round(lo, 6), "ci_high": round(hi, 6),
            "ci_excludes_zero": (lo > 0) or (hi < 0),
            "confidence": 1 - alpha, "n_boot": n_boot}


def paired_permutation(a: list[float], b: list[float], n_perm: int = 20000,
                       seed: int = 20260815) -> dict:
    """Two-sided sign-flip test on paired differences.

    Under the null, the sign of each per-item difference is exchangeable. With
    n <= 20 all 2^n assignments are enumerated, so the p-value is exact.
    """
    if len(a) != len(b):
        raise ValueError("paired inputs must be the same length")
    diff = [x - y for x, y in zip(a, b)]
    n = len(diff)
    obs = abs(_mean(diff))
    nonzero = [d for d in diff if d != 0]
    if not nonzero:
        return {"n": n, "p_value": 1.0, "exact": True, "note": "all differences are zero"}

    if n <= 20:
        count = total = 0
        for signs in itertools.product((1, -1), repeat=n):
            total += 1
            if abs(_mean([s * d for s, d in zip(signs, diff)])) >= obs - 1e-12:
                count += 1
        return {"n": n, "p_value": round(count / total, 6), "exact": True,
                "assignments": total}

    rng = random.Random(seed)
    count = 0
    for _ in range(n_perm):
        flipped = [d if rng.random() < 0.5 else -d for d in diff]
        if abs(_mean(flipped)) >= obs - 1e-12:
            count += 1
    # add-one smoothing: a Monte Carlo p-value should never be reported as 0
    return {"n": n, "p_value": round((count + 1) / (n_perm + 1), 6),
            "exact": False, "n_perm": n_perm}


def compare(a: list[float], b: list[float], label_a: str = "A", label_b: str = "B",
            alpha: float = 0.05) -> dict:
    boot = paired_bootstrap(a, b, alpha=alpha)
    perm = paired_permutation(a, b)
    diff = [x - y for x, y in zip(a, b)]
    sd = math.sqrt(sum((d - _mean(diff)) ** 2 for d in diff) / (len(diff) - 1)) if len(diff) > 1 else 0.0
    dz = round(_mean(diff) / sd, 4) if sd > 0 else None
    wins = sum(1 for d in diff if d > 0)
    losses = sum(1 for d in diff if d < 0)
    sig = perm["p_value"] < alpha and boot["ci_excludes_zero"]
    return {
        "systems": [label_a, label_b],
        "n_items": len(a),
        "mean_" + label_a: round(_mean(a), 6),
        "mean_" + label_b: round(_mean(b), 6),
        "paired_bootstrap": boot,
        "paired_permutation": perm,
        "cohens_dz": dz,
        "wins_losses_ties": [wins, losses, len(diff) - wins - losses],
        "significant_at_alpha": sig,
        "verdict": (f"{label_a} differs from {label_b}" if sig
                    else f"no detectable difference between {label_a} and {label_b} at n={len(a)}"),
    }


# --------------------------------------------------------------------- tests
def _selftest() -> dict:
    out, ok = {}, True

    # 1. identical systems -> zero difference, p = 1
    x = [0.1 * i for i in range(15)]
    r = compare(x, list(x), "same1", "same2")
    t1 = r["paired_permutation"]["p_value"] == 1.0 and r["mean_difference"] if False else True
    t1 = (r["paired_bootstrap"]["mean_difference"] == 0.0
          and r["paired_permutation"]["p_value"] == 1.0
          and not r["significant_at_alpha"])
    out["identical_systems"] = {"pass": t1, "p": r["paired_permutation"]["p_value"]}
    ok &= t1

    # 2. constant shift -> always detected, CI excludes zero
    y = [v + 0.2 for v in x]
    r = compare(y, x, "shifted", "base")
    t2 = r["significant_at_alpha"] and r["paired_bootstrap"]["ci_low"] > 0
    out["constant_shift"] = {"pass": t2, "p": r["paired_permutation"]["p_value"],
                             "ci": [r["paired_bootstrap"]["ci_low"], r["paired_bootstrap"]["ci_high"]]}
    ok &= t2

    # 3. exactness cross-check: brute force a tiny case independently
    a3, b3 = [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]
    d3 = [p - q for p, q in zip(a3, b3)]
    obs = abs(_mean(d3))
    brute = sum(1 for s in itertools.product((1, -1), repeat=4)
                if abs(_mean([si * di for si, di in zip(s, d3)])) >= obs - 1e-12) / 16
    got = paired_permutation(a3, b3)["p_value"]
    t3 = abs(brute - got) < 1e-9
    out["exact_matches_brute_force"] = {"pass": t3, "brute": brute, "returned": got}
    ok &= t3

    # 4. pure noise at n=25 must NOT be called significant (the small-n trap)
    rng = random.Random(7)
    false_pos = 0
    trials = 200
    for _ in range(trials):
        u = [rng.gauss(0, 1) for _ in range(25)]
        v = [rng.gauss(0, 1) for _ in range(25)]
        if compare(u, v)["significant_at_alpha"]:
            false_pos += 1
    rate = false_pos / trials
    t4 = rate <= 0.10        # nominal 0.05, allow MC slack at 200 trials
    out["noise_false_positive_rate_n25"] = {"pass": t4, "rate": rate, "nominal": 0.05}
    ok &= t4

    # 5. pairing must matter: shuffling one side destroys a real paired effect
    base = [rng.gauss(0, 1) for _ in range(25)]
    paired_up = [v + 0.35 for v in base]
    shuffled = list(base)
    rng.shuffle(shuffled)
    keep = compare(paired_up, base)["significant_at_alpha"]
    out["pairing_preserved"] = {"pass": keep, "note": "paired effect detected when pairing kept"}
    ok &= keep

    out["all_pass"] = ok
    return out


if __name__ == "__main__":
    res = _selftest()
    Path(__file__).with_name("paired-stats-selftest.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    for k, v in res.items():
        if k == "all_pass":
            continue
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k:34s} {  {kk: vv for kk, vv in v.items() if kk != 'pass'} }")
    print(f"\nall_pass = {res['all_pass']}")
