"""
Campaigns again, now carrying the uncertainty of the variance components.

For a subset of the cells of campaign.py, each campaign is fitted once and
then refitted on B parametric-bootstrap replicates (hglmm.bootstrap_theta).
The bootstrap thetas are used as approximate posterior draws: for each one the
coefficients are redrawn from the Laplace approximation on the observed data at
that theta, and the unseen ship's counts follow. Plans compared, all exact
under the true model:

  predictive    theta fixed at its estimate, as in campaign.py
  boot          theta integrated over the bootstrap draws
  conservative  campaign.py's conservative plan

Both predictive plans pick the vector from draws with the same rule
(plan_cover.plan_joint_draws_robust), which stays fast when a bootstrap theta
has a very small NB shape.

The cell list deliberately includes small campaigns, components at zero and
single-mission designs, where the estimate of the allocation is weakest.

    python campaign_boot.py --worker I --nworkers N [--hours 3.6]
    python campaign_boot.py --summary

With SPARES_POLISH=1 or SPARES_KEEP_FAILED=1 (sensitivity checks of the rule,
see hglmm.py) and --only list.csv, a subset of campaigns is rerun with the same
seeds into raw/campaign_boot_rule_polish or raw/campaign_boot_rule_keep.
"""
import argparse
import glob
import math
import os
import time

import numpy as np
import pandas as pd

import hglmm
from campaign import (LEVEL, RULE, SUFFIX, build_cells, fit_all, plans, simulate,
                      truth_zone, zone_draws)
from plan_cover import plan_aggregate_draws, plan_joint_draws_robust

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")
RAW = os.path.join(OUT, "raw", "campaign_boot" + SUFFIX)
SEED = 20261103
REPS = 40
B = 20
R_DRAWS = 40000
TRUTHS = [(0.42, 0.42), (0.6, 0.0), (0.0, 0.6), (0.0, 0.0)]


def cells():
    out = []
    for c in build_cells():
        if c["P"] != 4 or c["eps"] != 0.3 or c["unbal"] or (c["tau"], c["omega"]) not in TRUTHS:
            continue
        small = c["S"] in (5, 10, 20) and c["M"] in (1, 4) and c["m"] == 1.0
        busy = c["S"] in (10, 20) and c["M"] == 4 and c["m"] == 3.0     # where the plug-in thetas fall short
        if small or busy:
            out.append(c)
    return out


def pred_plan(fits, P, rng):
    per = R_DRAWS // len(fits)
    Y = []
    for g in fits:
        L = np.linalg.cholesky(g.cov_a[:g.d.p, :g.d.p] + 1e-12 * np.eye(g.d.p))
        bd = g.beta + (L @ rng.standard_normal((g.d.p, per))).T
        tau, om = g.sigma
        Y.append(zone_draws(g.beta, tau, om, g.phi, P, per, rng, beta_draws=bd))
    Y = np.vstack(Y)
    return plan_joint_draws_robust(Y, LEVEL), plan_aggregate_draws(Y, LEVEL)


def one_rep(c, rep):
    rng = np.random.default_rng([SEED, c["cell"], rep])
    d = simulate(c, rng)
    P = c["P"]
    full, no_ship, no_sp = fit_all(d, P)
    t0 = time.time()
    info = {}
    thetas = hglmm.bootstrap_theta(full, B, rng, rule=RULE, info=info)
    sig = np.exp(thetas[:, :2])
    tot = sig[:, 0] ** 2 + sig[:, 1] ** 2
    share = np.where(tot > 0, sig[:, 0] ** 2 / np.maximum(tot, 1e-300), 0.5)
    r = dict(cell=c["cell"], rep=rep, tau_hat=full.sigma[0], omega_hat=full.sigma[1],
             p_tau=hglmm.lrt_boundary(full.loglik, no_ship.loglik)[1],
             p_omega=hglmm.lrt_boundary(full.loglik, no_sp.loglik)[1],
             tau_b_q10=np.quantile(sig[:, 0], 0.1), tau_b_q90=np.quantile(sig[:, 0], 0.9),
             omega_b_q10=np.quantile(sig[:, 1], 0.1), omega_b_q90=np.quantile(sig[:, 1], 0.9),
             share_b_q10=np.quantile(share, 0.1), share_b_q90=np.quantile(share, 0.9),
             tau_b_zero=float(np.mean(sig[:, 0] < 0.02)), omega_b_zero=float(np.mean(sig[:, 1] < 0.02)),
             boot_seconds=time.time() - t0, fit_accepted=int(full.accepted),
             fit_starts=full.n_starts, fit_grad=full.grad_max, fit_start_won=full.start_won,
             fit_crit_gain=full.crit_gain,
             refits_failed=info.get("failed", 0), refits_restarted=info.get("restarted", 0),
             refits_grad_max=info.get("grad_max", 0.0))
    z = truth_zone(c)
    allp = plans(full, d, P, rng)
    allp["predictive"] = pred_plan([full], P, rng)
    allp["boot"] = pred_plan([hglmm.at_theta(full, th) for th in thetas], P, rng)
    allp["oracle"] = (z.plan_joint(LEVEL), z.plan_aggregate(LEVEL))
    for name in ("predictive", "boot", "conservative", "oracle"):
        sj, sa = allp[name]
        r[f"joint_units_{name}"] = int(np.sum(sj))
        r[f"joint_cover_{name}"] = z.joint(sj)
        r[f"agg_units_{name}"] = int(sa)
        r[f"agg_cover_{name}"] = z.aggregate(sa)
    return r


def tasks():
    return [(c, r) for r in range(REPS) for c in cells()]


def worker(i, n, hours, only=None):
    os.makedirs(RAW, exist_ok=True)
    t0 = time.time()
    todo = tasks()
    if only:
        keep = set(map(tuple, pd.read_csv(only)[["cell", "rep"]].values.tolist()))
        todo = [(c, rep) for c, rep in todo if (c["cell"], rep) in keep]
    for k, (c, rep) in enumerate(todo):
        if k % n != i:
            continue
        path = os.path.join(RAW, f"c{c['cell']:03d}_r{rep:03d}.csv")
        if os.path.exists(path):
            continue
        if time.time() - t0 > hours * 3600:
            print("time limit reached", flush=True)
            return
        t1 = time.time()
        pd.DataFrame([one_rep(c, rep)]).to_csv(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
        print(f"cell {c['cell']} rep {rep} {time.time() - t1:.0f}s", flush=True)


def summary():
    cl = pd.DataFrame(cells()).set_index("cell")
    R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RAW, "*.csv")))])
    R.to_csv(os.path.join(OUT, f"campaign_boot{SUFFIX}_reps.csv.gz"), index=False)
    rows = []
    for cell, g in R.groupby("cell"):
        c = cl.loc[cell]
        n = len(g)
        r = dict(cell=cell, S=c.S, M=c.M, tau=c.tau, omega=c.omega, reps=n,
                 share_width=float((g.share_b_q90 - g.share_b_q10).median()),
                 tau_b_zero=float(g.tau_b_zero.mean()), omega_b_zero=float(g.omega_b_zero.mean()))
        for crit in ("joint", "agg"):
            for name in ("predictive", "boot", "conservative", "oracle"):
                cv = g[f"{crit}_cover_{name}"]
                r[f"{crit}_{name}_cover"] = float(cv.mean())
                r[f"{crit}_{name}_cover_mcse"] = float(cv.std(ddof=1) / math.sqrt(n))
                r[f"{crit}_{name}_p10"] = float(cv.quantile(0.10))
                r[f"{crit}_{name}_units_med"] = float(g[f"{crit}_units_{name}"].median())
                r[f"{crit}_{name}_units_mean"] = float(g[f"{crit}_units_{name}"].mean())
            diff = g[f"{crit}_cover_boot"] - g[f"{crit}_cover_predictive"]
            r[f"{crit}_boot_minus_pred"] = float(diff.mean())
            r[f"{crit}_boot_minus_pred_mcse"] = float(diff.std(ddof=1) / math.sqrt(n))
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, f"campaign_boot{SUFFIX}_summary.csv"), index=False)
    print(T[["S", "M", "tau", "omega", "joint_predictive_cover", "joint_boot_cover",
             "joint_predictive_units_med", "joint_boot_units_med", "share_width"]].round(3).to_string())
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--hours", type=float, default=3.6)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", help="csv with cell and rep columns: run only those campaigns")
    a = ap.parse_args()
    if a.list:
        print(len(cells()), "cells,", len(tasks()), "campaigns")
    elif a.summary:
        summary()
    else:
        worker(a.worker, a.nworkers, a.hours, a.only)


if __name__ == "__main__":
    main()
