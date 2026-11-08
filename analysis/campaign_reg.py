"""
Small campaigns: where do the extreme bootstrap plans come from, and does an
explicit regularisation of the latent spread remove them?

Each campaign of campaign_boot.py is regenerated with its seed and the same
sequence of random numbers, so the predictive and bootstrap plans are those of
Table IV. New plans, from fresh streams:

  boot_cap      the bootstrap plan with at most CAP units per reference, an
                assumed storage limit; flagged infeasible when the capped
                vector misses the target under the bootstrap draws
  pen{s}        penalised criterion l_LA(theta) - (tau^2 + omega^2 + 1/phi) / (2 s^2),
                the mode of a half-normal prior of scale s on each of the three
                spreads: the two latent sds and the coefficient of variation
                1/sqrt(phi) of the gamma multiplier, followed by a parametric
                bootstrap of that estimator (data from the penalised fit, each
                replicate refitted with the same penalty); the predictive law
                integrates over the refits
  penpred{s}    the penalised estimate held fixed (no bootstrap)

For the trace, each bootstrap refit records its sds and NB shape, and the
draws that exceed the oracle vector are attributed to the refits that produced
them. Every plan is scored exactly under the true model.

    python campaign_reg.py --worker I --nworkers N [--hours 3.6]
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd

import hglmm
from campaign import RULE, SUFFIX, fit_all, plans, simulate, truth_zone, zone_draws
from campaign_boot import B, LEVEL, R_DRAWS, SEED, cells, pred_plan
from plan_cover import plan_joint_draws_robust

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "v2", "raw", "campaign_reg" + SUFFIX)
REPS = 40
SCALES = (0.5, 1.0, 2.0)
CAP = 15
CELLS = (0, 1, 2, 4, 10, 11, 12, 14, 20, 21, 22, 24)    # S = 5 or 10, M = 1; S = 5, M = 4


class PenFit(hglmm.Fit):
    """Laplace criterion plus the log density of a half-normal prior of scale s
    on every latent sd and on the coefficient of variation of the gamma
    multiplier (constants dropped)."""

    def __init__(self, design, scale, family="nb"):
        super().__init__(design, family)
        self.scale = scale

    def laplace(self, theta, z0=None):
        val, z, blocks = super().laplace(theta, z0)
        th = np.asarray(theta)
        spread2 = float(np.sum(np.exp(2 * th[:self.k])))
        if self.family == "nb":
            spread2 += math.exp(-th[self.k])
        return val - spread2 / (2 * self.scale ** 2), z, blocks


def pen_fit(d, P, scale, theta0=None):
    from campaign import fixed_cols
    return PenFit(hglmm.Design(d, fixed_cols(P), ["ship", "sp"]), scale).fit(theta0=theta0, rule=RULE)


def pen_boot(f, scale, nb, rng, info=None):
    return hglmm.bootstrap_theta(f, nb, rng, rule=RULE, info=info,
                                 fit_cls=lambda dd: PenFit(dd, scale, f.family))


def draws_by_fit(fits, P, rng):
    per = R_DRAWS // len(fits)
    Y, src = [], []
    for i, g in enumerate(fits):
        L = np.linalg.cholesky(g.cov_a[:g.d.p, :g.d.p] + 1e-12 * np.eye(g.d.p))
        bd = g.beta + (L @ rng.standard_normal((g.d.p, per))).T
        tau, om = g.sigma
        Y.append(zone_draws(g.beta, tau, om, g.phi, P, per, rng, beta_draws=bd))
        src.append(np.full(per, i))
    return np.vstack(Y), np.concatenate(src)


def one_rep(c, rep):
    rng = np.random.default_rng([SEED, c["cell"], rep])
    d = simulate(c, rng)
    P = c["P"]
    full = fit_all(d, P)[0]
    info = {}
    thetas = hglmm.bootstrap_theta(full, B, rng, rule=RULE, info=info)
    plans(full, d, P, rng)                           # keeps the stream of campaign_boot.py
    z = truth_zone(c)
    so = z.plan_joint(LEVEL)
    r = dict(cell=c["cell"], rep=rep, S=c["S"], M=c["M"], tau=c["tau"], omega=c["omega"],
             tau_hat=full.sigma[0], omega_hat=full.sigma[1], phi_hat=full.phi,
             oracle_units=int(so.sum()), oracle_cover=z.joint(so),
             fit_accepted=int(full.accepted), fit_starts=full.n_starts, fit_grad=full.grad_max,
             fit_start_won=full.start_won, fit_crit_gain=full.crit_gain,
             refits_failed=info.get("failed", 0), refits_restarted=info.get("restarted", 0),
             refits_grad_max=info.get("grad_max", 0.0), n_refits=len(thetas))

    def score(key, s):
        r[f"{key}_units"] = int(np.sum(s))
        r[f"{key}_cover"] = z.joint(s)

    score("pred", pred_plan([full], P, rng)[0])
    bfits = [hglmm.at_theta(full, th) for th in thetas]
    score("boot", pred_plan(bfits, P, rng)[0])      # the plan of Table IV

    # trace, on fresh draws from the same refits
    rng2 = np.random.default_rng([SEED, c["cell"], rep, 11])
    Y, src = draws_by_fit(bfits, P, rng2)
    sb = plan_joint_draws_robust(Y, LEVEL)
    sig = np.exp(thetas[:, :2])
    tot = np.hypot(sig[:, 0], sig[:, 1])
    phi = np.exp(thetas[:, 2])
    over = np.any(Y > so[None, :], axis=1)
    wide = np.flatnonzero(tot > 1.5)                 # latent sd far above any truth
    flat = np.flatnonzero(phi < 1.0)                 # gamma multiplier with cv above 1
    r.update(boot2_units=int(sb.sum()), boot2_cover=z.joint(sb),
             bsig_max=float(tot.max()), bsig_q90=float(np.quantile(tot, 0.9)),
             bsig_med=float(np.median(tot)), bphi_min=float(phi.min()),
             n_wide=len(wide), n_flat=len(flat), over_share=float(over.mean()),
             over_from_wide=float(np.isin(src[over], wide).mean()) if over.any() else np.nan,
             over_from_flat=float(np.isin(src[over], flat).mean()) if over.any() else np.nan,
             over_from_either=float(np.isin(src[over], np.union1d(wide, flat)).mean())
             if over.any() else np.nan)
    cap = np.minimum(sb, CAP)
    r["bootcap_infeasible"] = int(np.mean(np.all(Y <= cap[None, :], axis=1)) < LEVEL)
    score("bootcap", cap)

    for s in SCALES:
        rng3 = np.random.default_rng([SEED, c["cell"], rep, 12, int(10 * s)])
        f = pen_fit(d, P, s, theta0=full.theta.copy())
        pinfo = {}
        th = pen_boot(f, s, B, rng3, pinfo)
        tag = f"pen{s:g}"
        r[f"{tag}_fit_accepted"] = int(f.accepted)
        r[f"{tag}_refits_failed"] = pinfo.get("failed", 0)
        r[f"{tag}_tau"], r[f"{tag}_omega"] = f.sigma
        r[f"{tag}_phi"] = f.phi
        r[f"{tag}_bsig_max"] = float(np.hypot(*np.exp(th[:, :2]).T).max())
        r[f"{tag}_bphi_min"] = float(np.exp(th[:, 2]).min())
        score(f"penpred{s:g}", pred_plan([f], P, rng3)[0])
        score(tag, pred_plan([hglmm.at_theta(f, t) for t in th], P, rng3)[0])
    return r


def tasks():
    cl = {c["cell"]: c for c in cells()}
    return [(cl[k], r) for r in range(REPS) for k in CELLS]


def worker(i, n, hours):
    os.makedirs(RAW, exist_ok=True)
    t0 = time.time()
    for k, (c, rep) in enumerate(tasks()):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--hours", type=float, default=3.6)
    ap.add_argument("--one", nargs=2, type=int)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        import glob
        R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RAW, "*.csv")))])
        R.to_csv(os.path.join(HERE, "..", "data", "v2", f"campaign_reg{SUFFIX}_reps.csv.gz"), index=False)
        print(len(R), "campaigns")
    elif a.one:
        cl = {c["cell"]: c for c in cells()}
        t = time.time()
        print(one_rep(cl[a.one[0]], a.one[1]), f"{time.time() - t:.0f}s")
    else:
        worker(a.worker, a.nworkers, a.hours)


if __name__ == "__main__":
    main()
