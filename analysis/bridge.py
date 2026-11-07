"""
Bridge between the proposition and the fleet comparator, for the large ship
of ship_cases.py.

For each horizon the marginal law of every reference is held fixed while the
ship-wide share f varies: expected failures are the daily rate times the
horizon, and the latent multiplier exp(u + w_p - sigma^2/2) is drawn once per
horizon, so it persists over every cruise before the next resupply. The mean
grows in proportion to the horizon and the variance as
lambda + lambda^2 (exp(sigma^2) - 1).

Two contrasts, both unit-minimal at 99 % and both scored by the exact cover
under the reference law (the true f), never on the draws used to choose:

  exact   optimum at the true f against the optimum at f = 0 (the independence
          plan, robust by the proposition), both by branch and bound;
  draws   the certified optimum for R draws of the true law against the same
          draws with each column permuted independently (the fixed-marginal
          comparator of the fleet study), repeated over seeds.

At f = 0 the draw contrast measures only Monte Carlo error of selection.

    python bridge.py [--seeds 40] [--draws 6000]
"""
import argparse
import json
import math
import os
import time

import numpy as np

from plan_cover import Zone
from plan_exact import opt_draws
from ship_cases import greedy, ship
from stock_exact import exact_joint, split

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")
SEED = 20261106
LEVEL = 0.99
SIGMAS = (0.42, 0.6)    # 0.42: the ship and ship-reference spreads of the fleet; 0.6: Table II
HORIZONS = (7, 14, 28, 56)
SHARES = (0.0, 0.5, 1.0)


def draws(lams, f, sigma, R, rng):
    tau, om = split(f, sigma)
    u = rng.normal(0, tau, (R, 1))
    w = rng.normal(0, om, (R, len(lams)))
    return rng.poisson(lams[None, :] * np.exp(u + w - sigma ** 2 / 2))


def permute(Y, rng):
    return np.column_stack([Y[rng.permutation(len(Y)), j] for j in range(Y.shape[1])])


def cover_draws(Y, s):
    return float(np.mean(np.all(Y <= s[None, :], axis=1)))


def cover_product(Y, s):
    return float(np.prod([np.mean(Y[:, j] <= s[j]) for j in range(Y.shape[1])]))


def exact_opt(z):
    ones = np.ones(z.P)
    s, _, proved = exact_joint(z, LEVEL, ones, greedy(z, LEVEL, ones), seconds=900)
    return np.asarray(s, int), bool(proved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--draws", type=int, default=6000)
    a = ap.parse_args()
    names, N, rate_day, cost = ship()
    rows = []
    for SIGMA, H in [(sg, h) for sg in SIGMAS for h in HORIZONS]:
        lams = rate_day * H
        smax = int(lams.sum() + 12 * math.sqrt(lams.sum() + lams.sum() ** 2 * (math.exp(SIGMA ** 2) - 1)) + 60)
        z0 = Zone(lams, *split(0.0, SIGMA), smax=smax)
        ind, pr0 = exact_opt(z0)
        for f in SHARES:
            t0 = time.time()
            zf = Zone(lams, *split(f, SIGMA), smax=smax)
            dep, prf = exact_opt(zf) if f > 0 else (ind, pr0)
            r = dict(horizon=H, f=f, sigma=SIGMA, Lambda=float(lams.sum()),
                     marg_mean=lams.tolist(),
                     marg_var=(lams + lams ** 2 * (math.exp(SIGMA ** 2) - 1)).tolist(),
                     dep=dep.tolist(), ind=ind.tolist(), proved=pr0 and prf,
                     dep_units=int(dep.sum()), ind_units=int(ind.sum()),
                     dep_cost=float(cost @ dep), ind_cost=float(cost @ ind),
                     J_dep=zf.joint(dep), J_ind=zf.joint(ind), J_ind_f0=z0.joint(ind))
            sd = []
            for k in range(a.seeds):
                rng = np.random.default_rng([SEED, int(100 * SIGMA), H, int(10 * f), k])
                Y = draws(lams, f, SIGMA, a.draws, rng)
                Yp = permute(Y, rng)
                sj, cj, _, _ = opt_draws(Y, LEVEL)
                sp, cp, _, _ = opt_draws(Yp, LEVEL)
                sd.append(dict(uj=int(sj.sum()), up=int(sp.sum()), cj=int(cj), cp=int(cp),
                               Jj=zf.joint(sj), Jp=zf.joint(sp),
                               cost_j=float(cost @ sj), cost_p=float(cost @ sp),
                               resid=cover_draws(Yp, sp) - cover_product(Y, sp),
                               same=int(np.array_equal(sj, sp))))
            D = {k: np.array([x[k] for x in sd], float) for k in sd[0]}
            du, dJ = D["uj"] - D["up"], D["Jj"] - D["Jp"]
            dc = D["cost_j"] - D["cost_p"]

            def ci(x):
                se = x.std(ddof=1) / math.sqrt(len(x))
                return [float(x.mean() - 1.96 * se), float(x.mean() + 1.96 * se)]

            r.update(seeds=a.seeds, R=a.draws, cert=float(np.mean(np.r_[D["cj"], D["cp"]])),
                     du=float(du.mean()), du_ci=ci(du), du_p=[float(np.percentile(du, 2.5)), float(np.percentile(du, 97.5))],
                     dJ=float(dJ.mean()), dJ_ci=ci(dJ),
                     dcost=float(dc.mean()), dcost_ci=ci(dc),
                     Jj=float(D["Jj"].mean()), Jj_min=float(D["Jj"].min()),
                     Jp=float(D["Jp"].mean()), Jp_min=float(D["Jp"].min()),
                     uj=float(D["uj"].mean()), up=float(D["up"].mean()),
                     resid_max=float(np.abs(D["resid"]).max()), resid_mean=float(D["resid"].mean()),
                     same=float(D["same"].mean()), secs=time.time() - t0)
            print(H, f, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                         if k not in ("marg_mean", "marg_var")}, flush=True)
            rows.append(r)
    json.dump(dict(products=names, cost=cost.tolist(), level=LEVEL, sigmas=SIGMAS, seed=SEED, rows=rows),
              open(os.path.join(OUT, "bridge.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
