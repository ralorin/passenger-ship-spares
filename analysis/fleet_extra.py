"""
Summaries of the fleet study beyond fleet.py --summary: cover with Monte Carlo
intervals over fleets, paired comparisons between methods within fleets,
cover and shortfalls on missions with a ship-wide supply event, and an
explicit cost of stock and shortage.

    python fleet_extra.py
"""
import json
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")
RATIOS = (1, 10, 100)          # shortage cost per unit, in units of holding cost
rng = np.random.default_rng(20261104)


def boot_mean(x, B=2000):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    R = pd.read_csv(os.path.join(OUT, "fleet_v4_reps.csv.gz"))
    # keep the replicates finished in every scenario, so all scenarios share fleets 0..n-1
    done = R.groupby("rep").scenario.nunique()
    full = done[done == R.scenario.nunique()].index
    n = 0
    while n in full:
        n += 1
    R = R[R.rep < n]
    out = {"fleets_per_scenario": n}
    for scn, g in R.groupby("scenario"):
        o = {}
        for kind in ("known", "new", "known_shock", "new_shock"):
            h = g[g.kind == kind]
            if not len(h):
                continue
            o[kind] = {}
            for m, hm in h.groupby("method"):
                n = hm.n.sum()
                o[kind][m] = dict(missions=int(n), cover=float(hm.covered.sum() / n),
                                  fleets=len(hm), units=float(hm.units.sum() / n),
                                  cost=float(hm.cost.sum() / n), short=float(hm.short.sum() / n),
                                  short_cost=float(hm.short_cost.sum() / n),
                                  log_score=float(hm.log_score.mean()) if hm.log_score.notna().any() else None)
                if kind in ("known", "new"):
                    o[kind][m]["cover_ci"] = boot_mean(hm.covered / hm.n)[1:]
                    o[kind][m]["total_cost"] = {str(r): float((hm.cost.sum() + r * hm.short_cost.sum()) / n)
                                                for r in RATIOS}
        # paired, per fleet, hierarchy against the alternatives
        for kind in ("known", "new"):
            h = g[g.kind == kind].pivot_table(index="rep", columns="method",
                                              values=["log_score", "units", "covered", "n"])
            for alt in ("nb_pooled", "poisson_glm", "mtbf_scaled", "eb_gamma"):
                key = f"pair_{kind}_{alt}"
                du = h["units"]["hier_pred"] / h["n"]["hier_pred"] - h["units"][alt] / h["n"][alt]
                dc = h["covered"]["hier_pred"] / h["n"]["hier_pred"] - h["covered"][alt] / h["n"][alt]
                o[key] = dict(units=boot_mean(du), cover=boot_mean(dc), fewer_units=float((du < 0).mean()))
                if alt != "mtbf_scaled":
                    o[key]["log_score"] = boot_mean(h["log_score"]["hier_pred"] - h["log_score"][alt])
        out[scn] = o
    sig = R.drop_duplicates(["scenario", "rep"])
    caps = {}
    for col in ("sigma_hier", "sigma_shock", "sigma_agg"):
        a = np.array([json.loads(x) for x in sig[col]])
        caps[col] = dict(fleets_at_cap=int((a >= 2.999).any(axis=1).sum()), fleets=len(a))
    out["caps"] = caps
    with open(os.path.join(OUT, "fleet_extra.json"), "w") as f:
        json.dump(out, f, indent=1)
    for scn in out:
        if scn in ("caps", "fleets_per_scenario"):
            continue
        k = out[scn]["known"]
        print(scn, {m: (round(100 * v["cover"], 1), round(v["units"], 1)) for m, v in k.items()})
    print(caps)


if __name__ == "__main__":
    main()
