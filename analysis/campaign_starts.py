"""
How often does the single default start of the design study (campaign.py)
stop at a lower local maximum of the Laplace criterion?

For the first REPS campaigns of every cell of the main grid (S x M x split,
P = 4, m = 1, eps = 0.3), regenerated with campaign.py's seeds, the three
fits of each campaign (both terms, without the ship term, without the
ship-reference term) are repeated with the three-start rule of
hglmm.Fit.fit(rule=True). Recorded: the criterion gained, the estimates, and
whether the boundary tests at 5 % reach the same decision.

    python campaign_starts.py [--reps 10] [--workers 6]
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import hglmm
from campaign import SEED, build_cells, fixed_cols, simulate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")
CRIT = 2.71


def fits(d, P, rule):
    fx = fixed_cols(P)
    full = hglmm.Fit(hglmm.Design(d, fx, ["ship", "sp"])).fit(rule=rule)
    t, o, lp = full.theta
    ns = hglmm.Fit(hglmm.Design(d, fx, ["sp"])).fit(theta0=np.r_[o, lp], rule=rule)
    nw = hglmm.Fit(hglmm.Design(d, fx, ["ship"])).fit(theta0=np.r_[t, lp], rule=rule)
    return full, ns, nw


def one(args):
    c, rep = args
    rng = np.random.default_rng([SEED, c["cell"], rep])
    d = simulate(c, rng)
    r = dict(cell=c["cell"], rep=rep, S=c["S"], M=c["M"], tau=c["tau"], omega=c["omega"])
    for tag, rule in (("one", False), ("three", True)):
        f, ns, nw = fits(d, c["P"], rule)
        r[f"{tag}_ll"] = f.loglik
        r[f"{tag}_tau"], r[f"{tag}_omega"] = f.sigma
        r[f"{tag}_phi"] = f.phi
        r[f"{tag}_lrt_tau"] = max(0.0, 2 * (f.loglik - ns.loglik))
        r[f"{tag}_lrt_omega"] = max(0.0, 2 * (f.loglik - nw.loglik))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    cells = [c for c in build_cells() if c["P"] == 4 and c["m"] == 1.0 and c["eps"] == 0.3
             and not c["unbal"]]
    jobs = [(c, r) for c in cells for r in range(a.reps)]
    with Pool(a.workers) as pool:
        R = pd.DataFrame(pool.map(one, jobs, chunksize=4))
    R.to_csv(os.path.join(OUT, "campaign_starts.csv.gz"), index=False)
    gain = R.three_ll - R.one_ll
    out = dict(campaigns=len(R), cells=len(cells),
               higher=int((gain > 1e-3).sum()), higher_01=int((gain > 0.1).sum()),
               gain_max=float(gain.max()),
               tau_decision_changed=int(((R.one_lrt_tau > CRIT) != (R.three_lrt_tau > CRIT)).sum()),
               omega_decision_changed=int(((R.one_lrt_omega > CRIT) != (R.three_lrt_omega > CRIT)).sum()))
    for M in sorted(R.M.unique()):
        g = R[R.M == M]
        gg = g.three_ll - g.one_ll
        out[f"higher_M{M}"] = int((gg > 1e-3).sum())
        out[f"n_M{M}"] = len(g)
        out[f"omega_changed_M{M}"] = int(((g.one_lrt_omega > CRIT) != (g.three_lrt_omega > CRIT)).sum())
    json.dump(out, open(os.path.join(OUT, "campaign_starts.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
