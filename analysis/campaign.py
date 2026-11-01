"""
Campaign design study (simulation).

For ship s, reference p and mission t

    y_spt ~ Poisson(m * a_p * exp(u_s + w_sp + e_spt - (tau^2 + omega^2 + eps^2)/2))

with a_p = 1 (balanced) or proportional to 1:2:4:8 (unbalanced), scaled to mean 1.
Every campaign is fitted three times with hglmm (ship + ship-product, ship only,
ship-product only; reference as fixed effect), which gives the estimates and
the boundary LRT for each component. Then a 99 % plan for an unseen ship's
zone is built from the fit in four ways and its cover is computed exactly
under the true model:

  oracle        true parameters (the target)
  plugin        estimated parameters, latent draws only
  predictive    as plugin, plus draws of the coefficients
  conservative  plugin with the estimated total spread put all on the ship
                term and all on the interaction, taking the larger plan
  poisson       per-reference Poisson quantiles at the observed means

for both the joint and the aggregate criterion. Optionally (--ci) profile
likelihood intervals for tau and omega are added.

    python campaign.py --list
    python campaign.py --worker I --nworkers N [--hours 3.6]
    python campaign.py --summary
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import poisson

import hglmm
from plan_cover import (Zone, plan_aggregate_draws, plan_joint_draws,
                        plan_marginal_draws)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")
RAW = os.path.join(OUT, "raw", "campaign")
SEED = 20261101
REPS = 1000
BLOCK = 50
CI_REPS = 300
LEVEL = 0.99
R_DRAWS = 40000
BOUND = 0.02

TRUTHS = [(0.6, 0.0), (0.0, 0.6), (0.42, 0.42), (0.3, 0.3), (0.0, 0.0)]
BASE = dict(P=4, m=1.0, eps=0.3, unbal=False)


def build_cells():
    cells = []

    def add(**kw):
        for tau, om in TRUTHS:
            c = dict(BASE)
            c.update(kw, tau=tau, omega=om)
            c["ci"] = (c["S"] in (10, 20, 40) and c["M"] == 4 and c["P"] == 4
                       and c["m"] == 1.0 and c["eps"] == 0.3 and not c["unbal"])
            if c not in cells:
                cells.append(c)

    for S in (5, 10, 20, 40, 80):
        for M in (1, 2, 4, 8):
            add(S=S, M=M)
    for S in (10, 20):
        for m in (0.1, 0.3, 3.0):
            add(S=S, M=4, m=m)
    for P in (2, 8):
        add(S=20, M=4, P=P)
    for eps in (0.0, 0.6):
        add(S=20, M=4, eps=eps)
    add(S=20, M=4, unbal=True)
    for i, c in enumerate(cells):
        c["cell"] = i
    return cells


def weights(P, unbal):
    a = 2.0 ** np.arange(P) if unbal else np.ones(P)
    return a / a.mean()


def simulate(c, rng):
    S, M, P = c["S"], c["M"], c["P"]
    a = weights(P, c["unbal"])
    tau, om, eps = c["tau"], c["omega"], c["eps"]
    u = rng.normal(0, tau, S)
    w = rng.normal(0, om, (S, P))
    e = rng.normal(0, eps, (S, P, M))
    lam = (c["m"] * a[None, :, None]
           * np.exp(u[:, None, None] + w[:, :, None] + e - (tau ** 2 + om ** 2 + eps ** 2) / 2))
    y = rng.poisson(lam)
    s, p, t = np.meshgrid(np.arange(S), np.arange(P), np.arange(M), indexing="ij")
    d = pd.DataFrame(dict(ship=s.ravel(), product=p.ravel(), mission=t.ravel(),
                          failures=y.ravel().astype(float), hours=1.0))
    d["sp"] = d.ship * P + d["product"]
    for k in range(1, P):
        d[f"pr{k}"] = (d["product"] == k).astype(float)
    return d


def fixed_cols(P):
    return [f"pr{k}" for k in range(1, P)]


def fit_all(d, P):
    fx = fixed_cols(P)
    full = hglmm.Fit(hglmm.Design(d, fx, ["ship", "sp"])).fit()
    t, o, lp = full.theta
    no_ship = hglmm.Fit(hglmm.Design(d, fx, ["sp"])).fit(theta0=np.r_[o, lp])
    no_sp = hglmm.Fit(hglmm.Design(d, fx, ["ship"])).fit(theta0=np.r_[t, lp])
    return full, no_ship, no_sp


def zone_draws(beta, tau, om, phi, P, R, rng, beta_draws=None):
    """Counts for one mission of an unseen ship, from the fitted structure."""
    eta = np.empty((R, P))
    b = beta if beta_draws is None else beta_draws
    if b.ndim == 1:
        b = np.tile(b, (R, 1))
    eta[:, 0] = b[:, 0]
    eta[:, 1:] = b[:, :1] + b[:, 1:P]
    eta += rng.normal(0, 1, (R, 1)) * tau + rng.normal(0, 1, (R, P)) * om
    mu = np.exp(eta)
    if phi is not None:
        mu *= rng.gamma(phi, 1.0 / phi, (R, P))
    return rng.poisson(np.minimum(mu, 1e8))


def plans(full, d, P, rng):
    beta = full.beta
    tau, om = full.sigma
    phi = full.phi
    out = {}
    Y = zone_draws(beta, tau, om, phi, P, R_DRAWS, rng)
    out["plugin"] = (plan_joint_draws(Y, LEVEL), plan_aggregate_draws(Y, LEVEL))

    L = np.linalg.cholesky(full.cov_a[:full.d.p, :full.d.p] + 1e-12 * np.eye(full.d.p))
    bd = beta + (L @ rng.standard_normal((full.d.p, R_DRAWS))).T
    Y = zone_draws(beta, tau, om, phi, P, R_DRAWS, rng, beta_draws=bd)
    out["predictive"] = (plan_joint_draws(Y, LEVEL), plan_aggregate_draws(Y, LEVEL))

    tot = math.hypot(tau, om)
    js, ags = [], []
    for t2, o2 in ((tot, 0.0), (0.0, tot)):
        Y = zone_draws(beta, t2, o2, phi, P, R_DRAWS, rng)
        js.append(plan_joint_draws(Y, LEVEL))
        ags.append(plan_aggregate_draws(Y, LEVEL))
    out["conservative"] = (np.maximum(js[0], js[1]), max(ags))

    means = d.groupby("product").failures.mean().reindex(range(P)).to_numpy()
    out["poisson"] = (np.array([int(poisson.ppf(LEVEL, mu)) for mu in means]),
                      int(poisson.ppf(LEVEL, means.sum())))
    return out


def profile_ci(fobj, k, ll_reduced, level_chi=3.841, lower=-6.0):
    """Profile-likelihood interval for sigma_k; the lower end is 0 when the LRT keeps it."""
    th_hat = fobj.theta.copy()
    target = fobj.loglik - level_chi / 2
    others = [i for i in range(len(th_hat)) if i != k]
    cache = {"z": fobj.z.copy(), "x": th_hat[others]}

    def prof(x):
        def obj(o):
            th = th_hat.copy()
            th[k] = x
            th[others] = np.clip(o, lower, 12.0)
            v, z, _ = fobj.laplace(th, cache["z"])
            if not np.isfinite(v):
                return 1e12
            cache["z"] = z
            return -v
        r = optimize.minimize(obj, cache["x"], method="L-BFGS-B",
                              bounds=[(lower, 12.0)] * len(others),
                              options=dict(maxiter=200, ftol=1e-10))
        cache["x"] = r.x
        return -r.fun

    def bisect(a, b, fa_above):
        for _ in range(14):
            mid = 0.5 * (a + b)
            if (prof(mid) >= target) == fa_above:
                a = mid
            else:
                b = mid
        return 0.5 * (a + b)

    hi = th_hat[k] + 0.5
    while prof(hi) >= target and hi < math.log(5.0):
        hi += 0.5
    up = math.exp(bisect(th_hat[k], hi, True))
    cache["x"], cache["z"] = th_hat[others], fobj.z.copy()
    if 2 * (fobj.loglik - ll_reduced) <= level_chi:
        lo = 0.0
    else:
        lo = math.exp(bisect(th_hat[k], lower, True))
    return lo, up


_truth = {}


def truth_zone(c):
    key = c["cell"]
    if key not in _truth:
        lams = c["m"] * weights(c["P"], c["unbal"])
        _truth[key] = Zone(lams, c["tau"], math.hypot(c["omega"], c["eps"]))
    return _truth[key]


def one_rep(c, rep):
    rng = np.random.default_rng([SEED, c["cell"], rep])
    d = simulate(c, rng)
    P = c["P"]
    r = dict(cell=c["cell"], rep=rep, failures=int(d.failures.sum()))
    try:
        full, no_ship, no_sp = fit_all(d, P)
    except Exception as exc:
        r["error"] = type(exc).__name__
        return r
    r.update(tau_hat=full.sigma[0], omega_hat=full.sigma[1], phi_hat=full.phi,
             beta0_hat=full.beta[0], converged=full.converged and no_ship.converged and no_sp.converged,
             ll_full=full.loglik, ll_no_ship=no_ship.loglik, ll_no_sp=no_sp.loglik)
    r["lrt_tau"], r["p_tau"] = hglmm.lrt_boundary(full.loglik, no_ship.loglik)
    r["lrt_omega"], r["p_omega"] = hglmm.lrt_boundary(full.loglik, no_sp.loglik)

    z = truth_zone(c)
    allp = plans(full, d, P, rng)
    allp["oracle"] = (z.plan_joint(LEVEL), z.plan_aggregate(LEVEL))
    for name, (sj, sa) in allp.items():
        r[f"joint_units_{name}"] = int(np.sum(sj))
        r[f"joint_cover_{name}"] = z.joint(sj)
        r[f"agg_units_{name}"] = int(sa)
        r[f"agg_cover_{name}"] = z.aggregate(sa)
    if c["ci"] and rep < CI_REPS:
        r["ci_tau_lo"], r["ci_tau_hi"] = profile_ci(full, 0, no_ship.loglik)
        r["ci_omega_lo"], r["ci_omega_hi"] = profile_ci(full, 1, no_sp.loglik)
    return r


def chunks():
    return [(c, b) for c in build_cells() for b in range(REPS // BLOCK)]


def chunk_path(c, b):
    return os.path.join(RAW, f"c{c['cell']:03d}_b{b:02d}.csv")


def worker(i, n, hours):
    os.makedirs(RAW, exist_ok=True)
    t0 = time.time()
    todo = [x for k, x in enumerate(chunks()) if k % n == i]
    for c, b in todo:
        path = chunk_path(c, b)
        if os.path.exists(path):
            continue
        if time.time() - t0 > hours * 3600:
            print("time limit reached, leaving the rest for the next job", flush=True)
            return
        t1 = time.time()
        rows = [one_rep(c, rep) for rep in range(b * BLOCK, (b + 1) * BLOCK)]
        pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
        print(f"cell {c['cell']} block {b} {time.time() - t1:.0f}s", flush=True)


def summary():
    import glob
    cells = pd.DataFrame(build_cells()).set_index("cell")
    R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RAW, "*.csv")))])
    R.to_csv(os.path.join(OUT, "campaign_reps.csv.gz"), index=False)
    ok = R[R.get("error").isna()] if "error" in R else R
    rows = []
    for cell, g in ok.groupby("cell"):
        c = cells.loc[cell]
        n = len(g)
        r = dict(cell=cell, **{k: c[k] for k in ("S", "M", "P", "m", "eps", "unbal", "tau", "omega")},
                 reps=n, failed=int((R.cell == cell).sum() - n),
                 failures_median=g.failures.median())

        def prop(x):
            p = float(np.mean(x))
            return p, math.sqrt(p * (1 - p) / n)

        for comp, true in (("tau", c.tau), ("omega", c.omega)):
            est = g[f"{comp}_hat"]
            r[f"{comp}_bias"] = float(est.mean() - true)
            r[f"{comp}_rmse"] = float(np.sqrt(((est - true) ** 2).mean()))
            r[f"{comp}_at_bound"] = float((est <= BOUND).mean())
            r[f"reject_{comp}"], r[f"reject_{comp}_mcse"] = prop(g[f"p_{comp}"] < 0.05)
            if f"ci_{comp}_lo" in g and g[f"ci_{comp}_lo"].notna().any():
                h = g.dropna(subset=[f"ci_{comp}_lo"])
                r[f"ci_{comp}_cover"], r[f"ci_{comp}_cover_mcse"] = prop(
                    (h[f"ci_{comp}_lo"] <= true + 1e-9) & (true <= h[f"ci_{comp}_hi"]))
                r[f"ci_{comp}_n"] = len(h)
        if c.tau != c.omega:
            big = "tau" if c.tau > c.omega else "omega"
            got = np.where(g.tau_hat > g.omega_hat, "tau", "omega")
            r["order_ok"], r["order_ok_mcse"] = prop(got == big)
        for crit in ("joint", "agg"):
            for name in ("oracle", "plugin", "predictive", "conservative", "poisson"):
                cv = g[f"{crit}_cover_{name}"]
                r[f"{crit}_{name}_cover"] = float(cv.mean())
                r[f"{crit}_{name}_cover_mcse"] = float(cv.std(ddof=1) / math.sqrt(n))
                r[f"{crit}_{name}_cover_p10"] = float(cv.quantile(0.10))
                r[f"{crit}_{name}_short"] = float((cv < LEVEL - 0.005).mean())
                r[f"{crit}_{name}_units"] = float(g[f"{crit}_units_{name}"].mean())
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "campaign_summary.csv"), index=False)
    print(f"{len(R)} campaigns, {len(T)} cells")
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--worker", type=int)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--hours", type=float, default=3.6)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.list:
        cl = build_cells()
        print(f"{len(cl)} cells, {len(chunks())} chunks of {BLOCK} reps, "
              f"{sum(c['ci'] for c in cl)} cells with profile intervals")
    elif a.worker is not None:
        worker(a.worker, a.nworkers, a.hours)
    elif a.summary:
        summary()
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
