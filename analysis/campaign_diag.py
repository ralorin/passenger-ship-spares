"""
Why do the extreme bootstrap plans of the small campaigns differ between runs
with the same seeds?

For a list of campaigns of campaign_boot.py / campaign_reg.py this script
regenerates each campaign with its seed and follows the chain step by step,
recording what a run on another machine can be compared against:

  data       a hash of the simulated counts (fixed by the seed alone)
  fit        theta, the Laplace criterion, the optimiser's exit status and
             iterations, the projected gradient by central differences, the
             parameters at a bound, and the criterion and theta reached from
             three other starting points (flatness of the criterion)
  refits     the same for each of the B bootstrap refits, with a hash of
             each simulated data set
  stream     whether a change of 1e-9 in the point estimate changes the
             bootstrap data sets drawn from the same random state
  plan       the predictive and bootstrap vectors with the stream of Table IV,
             the refits that put draws above the known-parameter vector, and
             the bootstrap vector over K fresh streams at the same estimate

    python campaign_diag.py [--k 10] [--out ../data/v2/campaign_diag.json]
"""
import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
from scipy import optimize

import hglmm
from campaign import fit_all, plans, simulate, truth_zone, zone_draws
from campaign_boot import B, LEVEL, SEED, cells, pred_plan

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2", "campaign_diag.json")
LOWER, UPPER = -6.0, 12.0
# the S = 5, M = 1 campaigns whose bootstrap plan exceeded 1 000 units in either run
CASES = [(0, 29), (1, 4), (1, 6), (1, 9), (1, 19), (1, 28), (1, 37), (2, 11), (2, 29), (2, 38)]


def h(y):
    return hashlib.sha1(np.ascontiguousarray(y, dtype=np.float64).tobytes()).hexdigest()[:12]


def crit(fobj, th):
    return float(fobj.laplace(np.clip(th, LOWER, UPPER))[0])


def proj_grad(fobj, th, eps=1e-5):
    """Central-difference gradient of the criterion, projected on the box."""
    g = np.zeros(len(th))
    for i in range(len(th)):
        e = np.zeros(len(th))
        e[i] = eps
        g[i] = (crit(fobj, th + e) - crit(fobj, th - e)) / (2 * eps)
        if th[i] <= LOWER + 1e-8 and g[i] < 0:      # maximising: at the lower bound a
            g[i] = 0.0                               # negative slope is not a defect
        if th[i] >= UPPER - 1e-8 and g[i] > 0:
            g[i] = 0.0
    return g


def fit_record(design, family, theta0):
    """hglmm.Fit.fit with the optimiser result kept."""
    f = hglmm.Fit(design, family)
    nth = f.k + 1
    cache = {"z": None}

    def obj(th):
        th = np.clip(th, LOWER, UPPER)
        v, z, _ = f.laplace(th, cache["z"])
        if not np.isfinite(v):
            v, z, _ = f.laplace(th, None)
        if not np.isfinite(v):
            return 1e12
        cache["z"] = z
        return -v

    res = optimize.minimize(obj, theta0, method="L-BFGS-B",
                            bounds=[(LOWER, UPPER)] * nth,
                            options=dict(maxiter=500, ftol=1e-11, gtol=1e-6))
    g = proj_grad(f, res.x)
    return dict(theta=res.x.tolist(), crit=crit(f, res.x), success=bool(res.success),
                message=str(res.message), nit=int(res.nit), nfev=int(res.nfev),
                grad=g.tolist(), grad_max=float(np.max(np.abs(g))),
                at_lower=[bool(t <= LOWER + 1e-6) for t in res.x],
                at_upper=[bool(t >= UPPER - 1e-6) for t in res.x],
                sd_total=float(math.hypot(*np.exp(res.x[:2]))))


def boot_data(full, rng):
    d = hglmm.Design.__new__(hglmm.Design)
    d.__dict__.update(full.d.__dict__)
    d.y = hglmm.simulate_counts(full, rng)
    return d


def one(c, rep, K):
    t0 = time.time()
    rng = np.random.default_rng([SEED, c["cell"], rep])
    d = simulate(c, rng)
    P = c["P"]
    out = dict(cell=c["cell"], rep=rep, S=c["S"], M=c["M"], tau=c["tau"], omega=c["omega"],
               data_hash=h(d.failures.to_numpy()))
    full = fit_all(d, P)[0]
    design = full.d
    th0 = np.r_[math.log(0.5), math.log(0.5), math.log(5.0)]
    out["fit"] = fit_record(design, "nb", th0)
    out["fit"]["theta_hglmm"] = full.theta.tolist()
    starts = {"small": np.r_[-3.0, -3.0, 0.0], "large": np.r_[0.5, 0.5, 3.0],
              "swap": np.r_[full.theta[1], full.theta[0], full.theta[2]]}
    out["fit_starts"] = {k: fit_record(design, "nb", v) for k, v in starts.items()}

    # bootstrap with the stream of Table IV, refits instrumented
    state = rng.bit_generator.state
    refits, hashes, thetas = [], [], []
    for _ in range(B):
        dd = boot_data(full, rng)
        hashes.append(h(dd.y))
        r = fit_record(dd, "nb", full.theta.copy())
        refits.append(r)
        thetas.append(r["theta"])
    thetas = np.array(thetas)
    after = rng.bit_generator.state
    out["refits"] = refits
    out["refit_hashes"] = hashes

    # does a change of 1e-9 in the estimate change the bootstrap data sets?
    rng_b = np.random.default_rng(0)
    rng_b.bit_generator.state = state
    pert = hglmm.at_theta(full, full.theta + 1e-9)
    same = 0
    for k in range(B):
        dd = boot_data(pert, rng_b)
        same += int(h(dd.y) == hashes[k])
        if h(dd.y) != hashes[k]:
            break
    out["stream_same_before_divergence"] = same

    # plans with the stream of Table IV
    rng.bit_generator.state = after
    plans(full, d, P, rng)
    z = truth_zone(c)
    so = z.plan_joint(LEVEL)
    sp = pred_plan([full], P, rng)[0]
    bfits = [hglmm.at_theta(full, t) for t in thetas]
    sb = pred_plan(bfits, P, rng)[0]
    out.update(oracle_units=int(so.sum()), pred_units=int(sp.sum()), boot_units=int(sb.sum()),
               boot_vector=sb.tolist(), boot_cover=z.joint(sb))
    # which refits put draws above the known-parameter vector
    per = 40000 // B
    rng2 = np.random.default_rng([SEED, c["cell"], rep, 21])
    Y = zone_draws
    contrib = []
    for g in bfits:
        L = np.linalg.cholesky(g.cov_a[:g.d.p, :g.d.p] + 1e-12 * np.eye(g.d.p))
        bd = g.beta + (L @ rng2.standard_normal((g.d.p, per))).T
        Yg = Y(g.beta, g.sigma[0], g.sigma[1], g.phi, P, per, rng2, beta_draws=bd)
        contrib.append(float(np.mean(np.any(Yg > sb[None, :], axis=1))))
    out["exceed_by_refit"] = contrib
    # the bootstrap vector over K fresh streams, the point estimate held fixed
    reps = []
    for k in range(K):
        rk = np.random.default_rng([SEED, c["cell"], rep, 31, k])
        th = hglmm.bootstrap_theta(full, B, rk)
        sk = pred_plan([hglmm.at_theta(full, t) for t in th], P, rk)[0]
        tot = np.hypot(*np.exp(th[:, :2]).T)
        reps.append(dict(units=int(sk.sum()), cover=z.joint(sk), sd_max=float(tot.max()),
                         n_wide=int((tot > 1.5).sum())))
    out["fresh_streams"] = reps
    out["seconds"] = time.time() - t0
    print(c["cell"], rep, out["boot_units"], [r["units"] for r in reps],
          f"{out['seconds']:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--cases", default="")
    a = ap.parse_args()
    cl = {c["cell"]: c for c in cells()}
    cases = CASES if not a.cases else [tuple(map(int, s.split(":"))) for s in a.cases.split(",")]
    res = [one(cl[c], r, a.k) for c, r in cases]
    import numpy, scipy, platform
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(dict(machine=platform.platform(), numpy=numpy.__version__,
                       scipy=scipy.__version__, B=B, cases=res), f, indent=1)


if __name__ == "__main__":
    main()
