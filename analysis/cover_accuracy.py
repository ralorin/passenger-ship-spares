"""
Numerical accuracy of the exact joint cover J and its effect on the certified
optima of Tables I, II and V.

J (plan_cover.Zone) sums over a lattice of PER_SD points per standard
deviation of the smaller latent term, cut at HALF_WIDTH standard deviations,
with discretised normal weights. Two checks:

  error     every certified optimum and its one-unit-down neighbours (the
            vectors closest to the 99 % threshold from below) are evaluated on
            the base lattice and on a reference lattice with FINE_PER_SD points
            per sd cut at FINE_HALF_WIDTH sds; the largest difference estimates
            the numerical error of J, and the optimum's own margin above 0.99
            is compared with it.
  certify   every certified optimum is searched again by branch and bound (or
            dynamic programming where Table I used it) at level 0.99 - DELTA,
            starting from the optimum. If no cheaper vector reaches that lower
            level, an error in J smaller than DELTA cannot make a cheaper
            vector feasible, so the certificate stands.

    python cover_accuracy.py --worker I --nworkers N     # one share of the tasks
    python cover_accuracy.py --summary                   # data/v2/cover_accuracy.json
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import pandas as pd

import plan_cover
from plan_cover import Zone
from stock_exact import LEVEL, exact_joint, split

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")
RAW = os.path.join(OUT, "raw", "cover_accuracy")
BASE = (plan_cover.PER_SD, plan_cover.HALF_WIDTH)
FINE = (64, 10.0)
DELTA = 1e-6


def zone(lams, tau, omega, smax=None, grid=BASE):
    plan_cover.PER_SD, plan_cover.HALF_WIDTH = grid
    try:
        return Zone(lams, tau, omega, smax=smax)
    finally:
        plan_cover.PER_SD, plan_cover.HALF_WIDTH = BASE


def tasks():
    """(name, builder, cost, optimum, time limit, method) for every certified optimum."""
    out = []
    # Table I: grid of stock_exact.py, optimum at the true f and at f = 0
    T = pd.read_csv(os.path.join(OUT, "stock_exact_part1.csv"))
    dp = {(d["P"], d["total"], d["means"], d["costs"]): d
          for d in json.load(open(os.path.join(OUT, "stock_f0_dp.json"), encoding="utf-8"))}
    for _, r in T.iterrows():
        P = int(r.P)
        a = np.ones(P) if r.means == "equal" else 2.0 ** np.arange(P)
        lams = r.total * a / a.sum()
        c = np.ones(P) if r.costs == "equal" else 5.0 ** (np.arange(P) % 2)
        key = (P, r.total, r.means, r.costs)
        tag = f"t1_P{P}_{r.total:g}_{r.means}_{r.costs}"
        if r.proved:
            s = np.array(r.plan_exact.split(), int)
            out.append((f"{tag}_f{r.f:g}", ("zone", lams.tolist(), *split(r.f), None), c.tolist(),
                        s.tolist(), 600, "bb"))
        elif r.f == 0 and key in dp:
            out.append((f"{tag}_f0", ("zone", lams.tolist(), *split(0.0), None), c.tolist(),
                        dp[key]["dp_vector"], 0, "dp"))
    # Table II: the large ship (ship_cases.py)
    S = json.load(open(os.path.join(OUT, "ship_cases.json"), encoding="utf-8"))
    cost = S["cost"]
    for r in S["cases"]:
        lams = (np.array(S["failures_per_day"]) * r["days"]).tolist()
        sg = r["sigma"]
        tag = f"t2_{r['days']}d_s{sg:g}"
        out.append((f"{tag}_robust", ("zone", lams, *split(0.0, sg), None), cost, r["robust"], 600, "bb"))
        out.append((f"{tag}_f1", ("zone", lams, *split(1.0, sg), None), cost, r["know_f1"], 600, "bb"))
        out.append((f"{tag}_event", ("mixed", lams, sg, r["event_q"], S["installed"], S["event_share"]),
                    cost, r["event_plan"], 600, "bb"))
    # Table V: the bridge (bridge.py), unit costs
    B = json.load(open(os.path.join(OUT, "bridge.json"), encoding="utf-8"))
    rows = B["rows"] if isinstance(B, dict) else B
    seen = set()
    for r in rows:
        lams = r["marg_mean"]
        sg, H, f = r["sigma"], r["horizon"], r["f"]
        smax = int(sum(lams) + 12 * math.sqrt(sum(lams) + sum(lams) ** 2 * (math.exp(sg ** 2) - 1)) + 60)
        ones = [1.0] * len(lams)
        if (sg, H) not in seen:
            seen.add((sg, H))
            out.append((f"t5_s{sg:g}_{H}d_ind", ("zone", lams, *split(0.0, sg), smax), ones, r["ind"], 900, "bb"))
        if f > 0:
            out.append((f"t5_s{sg:g}_{H}d_f{f:g}", ("zone", lams, *split(f, sg), smax), ones, r["dep"], 900, "bb"))
    return out


def build(spec, grid):
    if spec[0] == "zone":
        _, lams, tau, om, smax = spec
        return zone(np.array(lams), tau, om, smax, grid)
    from ship_cases import Mixed
    _, lams, sg, q, N, a = spec
    zb = zone(np.array(lams), *split(0.0, sg), grid=grid)
    zb = zone(np.array(lams), *split(0.0, sg), smax=zb.smax + 40, grid=grid)
    return Mixed(zb, np.array(N), a, q)


def dp_cost(z, cost, level, bmax):
    from stock_f0_dp import dp_optimum
    r = dp_optimum(z, np.asarray(cost), level, bmax)
    return None if r is None else float(r[1])


def run(task):
    name, spec, cost, s, limit, method = task
    s, cost = np.array(s, int), np.array(cost, float)
    t0 = time.time()
    zb, zf = build(spec, BASE), build(spec, FINE)
    down = [s - np.eye(len(s), dtype=int)[p] for p in range(len(s)) if s[p] > 0]
    err = max(abs(zb.joint(v) - zf.joint(v)) for v in [s] + down)
    r = dict(name=name, method=method, cost=float(cost @ s), J_base=zb.joint(s), J_fine=zf.joint(s),
             down_max_base=max((zb.joint(v) for v in down), default=0.0), err=float(err))
    if method == "dp":
        r["cost_lower_level"] = dp_cost(zb, cost, LEVEL - DELTA, int(cost @ s) + 5)
        r["proved"] = True
    else:
        best, nodes, proved = exact_joint(zb, LEVEL - DELTA, cost, s, seconds=limit)
        r.update(cost_lower_level=float(cost @ best), proved=bool(proved), nodes=int(nodes))
    r["same"] = bool(r["proved"] and abs(r["cost_lower_level"] - r["cost"]) < 1e-9)
    r["seconds"] = time.time() - t0
    return r


def summary():
    R = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(RAW, "*.json")))]
    D = pd.DataFrame(R)
    out = dict(base=BASE, fine=FINE, delta=DELTA, optima=len(D),
               by_table={t: int(D.name.str.startswith(t).sum()) for t in ("t1", "t2", "t5")},
               err_max=float(D.err.max()),
               margin_min=float((D.J_base - LEVEL).min()),
               margin_fine_min=float((D.J_fine - LEVEL).min()),
               below_fine=int((D.J_fine < LEVEL).sum()),
               margin_below_delta=int((D.J_base - LEVEL < DELTA).sum()),
               rechecked=int(D.proved.sum()), same=int(D.same.sum()),
               changed=D[D.proved & ~D.same].name.tolist(),
               not_finished=D[~D.proved].name.tolist())
    json.dump(dict(summary=out, rows=R), open(os.path.join(OUT, "cover_accuracy.json"), "w",
                                              encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        return summary()
    os.makedirs(RAW, exist_ok=True)
    T = tasks()
    for i, t in enumerate(T):
        if i % a.nworkers != a.worker:
            continue
        path = os.path.join(RAW, t[0] + ".json")
        if os.path.exists(path):
            continue
        r = run(t)
        json.dump(r, open(path, "w", encoding="utf-8"), indent=1)
        print(r["name"], r["same"], f"{r['err']:.1e}", f"{r['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
