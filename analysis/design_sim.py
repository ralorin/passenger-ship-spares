"""
Identifiability and campaign design for a shipboard failure-count model.

SIMULATION. Every number produced here comes from data generated under the
model of the manuscript with ASSUMED parameters. Nothing in this file is a
measurement of shipboard equipment, and no value should be quoted as one. The
purpose is to answer, before any fleet spends money collecting records, two
questions that can be answered without data:

  Q1 (identifiability) Can a campaign separate ship-wide heterogeneity (tau)
     from ship-product interaction (omega)? These two allocations of the same
     total spread imply different multi-product spares plans (multiproduct.py),
     so a fleet that cannot separate them cannot choose its plan on evidence.

  Q2 (design) How many ships, missions and failures are needed before the
     estimates are usable, judged by the cover a 99% plan then delivers to a
     ship the fit has never seen?

Generating model, for ship s, product p, mission m:

    y_spm ~ Poisson(H_spm * exp(beta0 + u_s + w_sp + e_spm))
    u_s  ~ N(0, tau^2)        ship-wide deviation, shared by all its products
    w_sp ~ N(0, omega^2)      ship-product interaction
    e_spm~ N(0, eps^2)        mission-to-mission dispersion

Exposure is held equal across cells and set so that each (ship, product,
mission) expects `m_exp` failures. The fit uses the same structure, with the
ship and interaction terms as crossed random effects.

    python design_sim.py [--quick]
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from multiprocessing import Pool

from functools import lru_cache

import numpy as np
import pandas as pd

from glmm import Design, HGLMM
from multiproduct import quantile_from_pmf, total_pmf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")
os.makedirs(OUT, exist_ok=True)

BOUND = 0.02          # an estimate at or below this is treated as at the boundary
SEED = 20260927


def simulate(S, M, P, m_exp, tau, omega, eps, rng):
    """One campaign. Returns the cell table and the realised latent effects."""
    u = rng.normal(0, tau, S) if tau > 0 else np.zeros(S)
    w = rng.normal(0, omega, (S, P)) if omega > 0 else np.zeros((S, P))
    rows = []
    for s in range(S):
        for p in range(P):
            for t in range(M):
                e = rng.normal(0, eps) if eps > 0 else 0.0
                lam = m_exp * np.exp(u[s] + w[s, p] + e
                                     - (tau ** 2 + omega ** 2 + eps ** 2) / 2)
                rows.append(dict(ship=f"S{s:03d}", product=f"P{p}", mission=t,
                                 sp=f"S{s:03d}|P{p}", hours=1.0,
                                 failures=float(rng.poisson(lam))))
    return pd.DataFrame(rows)


def fit_campaign(d, with_interaction=True):
    random = ["ship"] + (["sp"] if with_interaction else [])
    random = [r for r in random if d[r].nunique() > 1]
    f = HGLMM(Design(d, [], random), "nb").fit()
    sig = dict(zip(f.d.random, f.sigma.tolist()))
    return sig, f


def one_rep(args):
    (S, M, P, m_exp, tau, omega, eps, rep) = args
    rng = np.random.default_rng([SEED, S, M, P, int(m_exp * 100),
                                 int(tau * 100), int(omega * 100), rep])
    d = simulate(S, M, P, m_exp, tau, omega, eps, rng)
    out = dict(S=S, M=M, P=P, m_exp=m_exp, tau=tau, omega=omega, eps=eps, rep=rep,
               failures=int(d.failures.sum()))
    try:
        sig, _ = fit_campaign(d, with_interaction=True)
        out["tau_hat"] = float(sig.get("ship", np.nan))
        out["omega_hat"] = float(sig.get("sp", np.nan))
        out["tau_bound"] = bool(out["tau_hat"] <= BOUND)
        out["omega_bound"] = bool(out["omega_hat"] <= BOUND)
        out["ok"] = True
    except Exception as exc:                     # a campaign the estimator cannot fit
        out["ok"] = False
        out["error"] = type(exc).__name__
    return out


@lru_cache(maxsize=200000)
def _pmf_cached(P, total_mean, tau, omega):
    """total_pmf on rounded arguments; the estimates are binned to 0.02 in the
    latent sd, far finer than their own sampling error, so the cache changes no
    reported figure and makes the grid affordable."""
    pmf, _ = total_pmf(np.full(P, total_mean / P), tau, omega, nodes=201)
    return np.cumsum(pmf)


def delivered_cover(tau_hat, omega_hat, tau, omega, P, total_mean, level=99):
    """Cover a plan built on the ESTIMATES delivers under the TRUE structure."""
    rnd = lambda x: round(max(0.0, float(x)) / 0.02) * 0.02
    c_hat = _pmf_cached(P, total_mean, rnd(tau_hat), rnd(omega_hat))
    q = int(np.argmax(c_hat >= level / 100.0))
    c_true = _pmf_cached(P, total_mean, rnd(tau), rnd(omega))
    return q, float(c_true[min(q, len(c_true) - 1)])


CONFIGS = [
    # S ships, M missions, P products, expected failures per (ship, product, mission)
    dict(S=5,  M=4, P=4, m_exp=1.0),
    dict(S=10, M=1, P=4, m_exp=1.0),
    dict(S=10, M=4, P=4, m_exp=1.0),
    dict(S=10, M=4, P=4, m_exp=3.0),
    dict(S=20, M=4, P=4, m_exp=1.0),
    dict(S=20, M=8, P=4, m_exp=1.0),
    dict(S=40, M=4, P=4, m_exp=1.0),
]
TRUTHS = [
    dict(tau=0.6, omega=0.0),      # ship-wide only
    dict(tau=0.0, omega=0.6),      # interaction only
    dict(tau=0.42, omega=0.42),    # both, equal split of the same total
]
EPS = 0.3


def main():
    quick = "--quick" in sys.argv
    reps = 40 if quick else 200
    tasks = []
    for c in CONFIGS:
        for t in TRUTHS:
            for r in range(reps):
                tasks.append((c["S"], c["M"], c["P"], c["m_exp"],
                              t["tau"], t["omega"], EPS, r))
    print(f"{len(tasks)} campaigns ({reps} replicates per cell)", flush=True)
    t0 = time.time()
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        res = pool.map(one_rep, tasks, chunksize=8)
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(OUT, "design_sim_reps.csv"), index=False)
    print(f"{time.time() - t0:.0f}s, {R.ok.mean():.1%} fitted", flush=True)

    rows = []
    for (S, M, P, m_exp, tau, omega), g in R.groupby(["S", "M", "P", "m_exp", "tau", "omega"]):
        ok = g[g.ok]
        if not len(ok):
            continue
        r = dict(S=S, M=M, P=P, m_exp=m_exp, tau=tau, omega=omega,
                 reps=int(len(g)), fitted=float(ok.ok.mean()),
                 failures_median=float(g.failures.median()),
                 tau_bound=float(ok.tau_bound.mean()), omega_bound=float(ok.omega_bound.mean()),
                 tau_median=float(ok.tau_hat.median()), omega_median=float(ok.omega_hat.median()),
                 tau_rmse=float(np.sqrt(((ok.tau_hat - tau) ** 2).mean())),
                 omega_rmse=float(np.sqrt(((ok.omega_hat - omega) ** 2).mean())))
        # can the campaign tell the two structures apart? A campaign "recovers the
        # allocation" when the larger estimate belongs to the larger true term.
        if tau != omega:
            bigger_true = "tau" if tau > omega else "omega"
            got = np.where(ok.tau_hat > ok.omega_hat, "tau", "omega")
            r["allocation_recovered"] = float((got == bigger_true).mean())
        else:
            r["allocation_recovered"] = float(np.nan)
        # the cover a 99% plan built on the estimates delivers to a new ship
        cov, q = [], []
        for _, row in ok.iterrows():
            qq, cc = delivered_cover(row.tau_hat, row.omega_hat, tau, omega, P,
                                     total_mean=m_exp * P)
            q.append(qq)
            cov.append(cc)
        r["plan99_median"] = float(np.median(q))
        r["cover_mean"] = float(np.mean(cov))
        r["cover_p10"] = float(np.quantile(cov, 0.10))
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "design_sim_summary.csv"), index=False)
    json.dump(dict(reps=reps, eps=EPS, bound=BOUND, configs=CONFIGS, truths=TRUTHS,
                   summary=T.to_dict("records")),
              open(os.path.join(OUT, "design_sim.json"), "w"), indent=1, default=float)

    pd.set_option("display.width", 200)
    print("\nEvery value below is simulated under assumed parameters.\n")
    print(T[["S", "M", "m_exp", "tau", "omega", "failures_median", "tau_bound",
             "omega_bound", "tau_rmse", "omega_rmse", "allocation_recovered",
             "plan99_median", "cover_mean", "cover_p10"]].to_string(index=False))


if __name__ == "__main__":
    main()
