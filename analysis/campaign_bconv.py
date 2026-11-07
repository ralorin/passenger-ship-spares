"""
Is B = 20 enough? For four cells of campaign_boot.py, each campaign is refitted
on 80 bootstrap replicates, and the theta-integrated plan is built from the
first 20, the first 40 and all 80, each from its own draws; a second plan from
all 80 with fresh draws measures the Monte Carlo error of selecting the vector.
Every plan is scored exactly under the true model.

    python campaign_bconv.py [--workers 7]
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import hglmm
from campaign import RULE, SUFFIX, fit_all, simulate, truth_zone
from campaign_boot import LEVEL, cells, pred_plan

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")
SEED = 20261104
REPS = 40
CELLS = (1, 2, 22, 52)
BS = (20, 40, 80)


def one(args):
    c, rep = args
    rng = np.random.default_rng([SEED, c["cell"], rep])
    d = simulate(c, rng)
    full = fit_all(d, c["P"])[0]
    info = {}
    thetas = hglmm.bootstrap_theta(full, max(BS), rng, rule=RULE, info=info)
    fits = [hglmm.at_theta(full, th) for th in thetas]
    z = truth_zone(c)
    r = dict(cell=c["cell"], rep=rep, fit_accepted=int(full.accepted), fit_crit_gain=full.crit_gain,
             refits_failed=info.get("failed", 0), n_refits=len(thetas))
    for b in BS:
        sj = pred_plan(fits[:b], c["P"], rng)[0]
        r[f"cover_b{b}"], r[f"units_b{b}"] = z.joint(sj), int(sj.sum())
    sj = pred_plan(fits, c["P"], rng)[0]
    r["cover_b80_again"], r["units_b80_again"] = z.joint(sj), int(sj.sum())
    sj = pred_plan([full], c["P"], rng)[0]
    r["cover_pred"], r["units_pred"] = z.joint(sj), int(sj.sum())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    cl = {c["cell"]: c for c in cells()}
    jobs = [(cl[k], r) for r in range(REPS) for k in CELLS]
    with Pool(a.workers) as pool:
        R = pd.DataFrame(pool.map(one, jobs, chunksize=1))
    R.to_csv(os.path.join(OUT, f"campaign_bconv{SUFFIX}_reps.csv"), index=False)
    out = {}
    for k, g in R.groupby("cell"):
        c = cl[k]
        e = dict(S=c["S"], M=c["M"], tau=c["tau"], omega=c["omega"], reps=len(g))
        for b in BS:
            e[f"cover_b{b}"] = g[f"cover_b{b}"].mean()
            e[f"units_b{b}"] = g[f"units_b{b}"].median()
        e["cover_pred"] = g.cover_pred.mean()
        d20 = g.cover_b20 - g.cover_b80
        e["d20_80_mean"], e["d20_80_mcse"] = d20.mean(), d20.std(ddof=1) / np.sqrt(len(g))
        e["d20_80_absmed"] = d20.abs().median()
        rep = g.cover_b80_again - g.cover_b80
        e["redraw_absmed"], e["redraw_absmax"] = rep.abs().median(), rep.abs().max()
        e["redraw_units_absmax"] = int((g.units_b80_again - g.units_b80).abs().max())
        out[int(k)] = {kk: float(v) for kk, v in e.items()}
    json.dump(out, open(os.path.join(OUT, f"campaign_bconv{SUFFIX}.json"), "w"), indent=1)
    print(pd.DataFrame(out).T.round(4).to_string())


if __name__ == "__main__":
    main()
