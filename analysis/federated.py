"""
Federated Laplace fit with the whole local block eliminated by each holder.

Holder s keeps v_s = (u_s, w_s1, ..., w_sP): its ship effect and the
ship-product effects of its own references. None of them appears in another
holder's cells, so the joint Hessian is block-arrowhead in (gamma, v_1..v_S)
and each holder can send

    f_s, grad_gamma f_s, S_s = H_gg - H_gv H_vv^-1 H_vg, log det H_vv

for the global coordinates gamma it touches. The coordinator adds them up and
takes the Newton step of the profiled objective. This generalises
glmm.FederatedFit, where v_s was the ship effect alone.

The experiment checks the fit against the centralised hglmm.Fit and counts what
is transmitted, against the one-shot alternative of sending the cell table.

    python federated.py
"""
import json
import math
import os
import time

import numpy as np
import pandas as pd
from scipy import optimize

import hglmm
from glmm import BETA_PRIOR_SD, LOG2PI, Design, _nb_terms

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")


class BlockFederatedFit:
    """Random terms must be listed as [global terms..., ship, ship_product]."""

    def __init__(self, design, family="nb"):
        d = self.d = design
        self.family = family
        k = len(d.random)
        self.k = k
        self.ig = list(range(k - 2))                    # global random terms
        self.gcols = np.r_[np.arange(d.p)] if not self.ig else np.r_[
            np.arange(d.p), np.concatenate([np.arange(d.offsets[i], d.offsets[i] + d.q[i])
                                            for i in self.ig])]
        Jg = d.J[:, self.gcols]
        ship = d.idx[d.random[-2]]
        sp = d.idx[d.random[-1]]
        self.holders = []
        for s in range(d.q[-2]):
            rows = np.where(ship == s)[0]
            Js = Jg[rows]
            touched = np.unique(Js.indices)
            sp_lev, sp_loc = np.unique(sp[rows], return_inverse=True)
            Jv = np.zeros((len(rows), 1 + len(sp_lev)))
            Jv[:, 0] = 1.0
            Jv[np.arange(len(rows)), 1 + sp_loc] = 1.0
            self.holders.append(dict(rows=rows, J=Js[:, touched].toarray(), touched=touched,
                                     Jv=Jv, P=len(sp_lev)))
        self.rounds = self.floats_sent = self.messages = 0

    def _unpack(self, theta):
        sig = np.exp(theta[:self.k])
        phi = math.exp(theta[self.k]) if self.family == "nb" else None
        return sig, phi

    def local(self, h, zg_local, tau, om, phi, v0):
        y = self.d.y[h["rows"]]
        off = self.d.log_off[h["rows"]]
        base = h["J"] @ zg_local
        Jv = h["Jv"]
        D = np.r_[1 / tau ** 2, np.full(h["P"], 1 / om ** 2)]
        v = v0.copy()
        for _ in range(100):
            ll, g, w = _nb_terms(y, base + Jv @ v, off, phi)
            gv = Jv.T @ g - D * v
            Hv = Jv.T @ (w[:, None] * Jv) + np.diag(D)
            dv = np.linalg.solve(Hv, gv)
            dv *= min(1.0, 2.0 / max(np.max(np.abs(dv)), 1e-300))    # damped far from the mode
            v += dv
            if np.max(np.abs(dv)) < 1e-12:
                break
        ll, g, w = _nb_terms(y, base + Jv @ v, off, phi)
        Hv = Jv.T @ (w[:, None] * Jv) + np.diag(D)
        Hgv = h["J"].T @ (w[:, None] * Jv)
        Hgg = h["J"].T @ (w[:, None] * h["J"])
        S = Hgg - Hgv @ np.linalg.solve(Hv, Hgv.T)
        f = (ll.sum() - 0.5 * np.sum(D * v ** 2) - math.log(tau) - h["P"] * math.log(om)
             - 0.5 * (1 + h["P"]) * LOG2PI)
        m = len(h["touched"])
        self.floats_sent += 1 + m + m * (m + 1) // 2 + 1
        self.messages += 1
        return f, h["J"].T @ g, S, np.linalg.slogdet(Hv)[1], v

    def laplace(self, theta, zg0=None, v0=None, tol=1e-9, maxit=100):
        d = self.d
        sig, phi = self._unpack(theta)
        tau, om = sig[-2], sig[-1]
        ng = len(self.gcols)
        prec = np.zeros(ng)
        prec[1:d.p] = 1.0 / BETA_PRIOR_SD ** 2
        pos = d.p
        for i in self.ig:
            prec[pos:pos + d.q[i]] = 1.0 / sig[i] ** 2
            pos += d.q[i]
        zg = np.zeros(ng) if zg0 is None else zg0.copy()
        if zg0 is None:
            zg[0] = math.log(max(d.y.sum(), 0.5) / np.exp(d.log_off).sum())
        vs = [np.zeros(1 + h["P"]) for h in self.holders] if v0 is None else [v.copy() for v in v0]

        def evaluate(zg):
            F, G, Hs, L, vnew = 0.0, np.zeros(ng), np.zeros((ng, ng)), 0.0, []
            for h, v in zip(self.holders, vs):
                f, g, S, lh, vh = self.local(h, zg[h["touched"]], tau, om, phi, v)
                t = h["touched"]
                F += f
                G[t] += g
                Hs[np.ix_(t, t)] += S
                L += lh
                vnew.append(vh)
            self.rounds += 1
            return (F - 0.5 * np.sum(prec * zg ** 2), G - prec * zg, Hs + np.diag(prec), L, vnew)

        F, G, Hs, L, vs = evaluate(zg)
        for _ in range(maxit):
            step = np.linalg.solve(Hs, G)
            t = 1.0
            while True:
                F2, G2, H2, L2, v2 = evaluate(zg + t * step)
                if F2 >= F - 1e-10 or t < 1e-6:
                    break
                t *= 0.5
            zg = zg + t * step
            done = abs(F2 - F) < tol and np.max(np.abs(t * step)) < 1e-6
            F, G, Hs, L, vs = F2, G2, H2, L2, v2
            if done:
                break
        logprior = sum(-d.q[i] * math.log(sig[i]) - 0.5 * d.q[i] * LOG2PI for i in self.ig)
        val = F + logprior + 0.5 * d.dim * LOG2PI - 0.5 * (L + np.linalg.slogdet(Hs)[1])
        return val, zg, vs

    def fit(self, theta0, lower=-6.0, upper=12.0):
        cache = {"zg": None, "v": None}

        def obj(th):
            th = np.clip(th, lower, upper)
            v, zg, vs = self.laplace(th, cache["zg"], cache["v"])
            if not np.isfinite(v):
                v, zg, vs = self.laplace(th, None, None)
            if not np.isfinite(v):
                return 1e12
            cache["zg"], cache["v"] = zg, vs
            return -v

        res = optimize.minimize(obj, theta0, method="L-BFGS-B",
                                bounds=[(lower, upper)] * len(theta0),
                                options=dict(maxiter=500, ftol=1e-11, gtol=1e-6))
        self.theta = res.x
        self.loglik, self.zg, self.v = self.laplace(self.theta, cache["zg"], cache["v"])
        self.sigma, self.phi = self._unpack(self.theta)
        self.beta = self.zg[:self.d.p]
        self.n_eval = res.nfev
        return self


def fleet(S, P, M, rng, makers=3, tau=0.4, om=0.4, s_mk=0.3, eps=0.3, m=1.0, b_temp=0.25):
    """Ships x references x missions with a continuous pre-mission covariate."""
    u = rng.normal(0, tau, S)
    w = rng.normal(0, om, (S, P))
    maker_of = np.arange(P) % makers
    mk = rng.normal(0, s_mk, makers)
    rows = []
    for s in range(S):
        for p in range(P):
            for t in range(M):
                temp = rng.normal(0, 1)          # previous-mission zone temperature, standardised
                hours = rng.uniform(0.5, 1.5)
                lam = m * hours * math.exp(u[s] + w[s, p] + mk[maker_of[p]] + b_temp * temp
                                           + rng.normal(0, eps)
                                           - (tau ** 2 + om ** 2 + s_mk ** 2 + eps ** 2) / 2)
                rows.append(dict(ship=s, sp=s * P + p, maker=maker_of[p], product=p, temp=temp,
                                 hours=hours, failures=float(rng.poisson(lam))))
    d = pd.DataFrame(rows)
    for k in range(1, P):
        d[f"pr{k}"] = (d["product"] == k).astype(float)
    return d


def compare(d, P):
    fixed = ["temp"] + [f"pr{k}" for k in range(1, P)]
    random = ["maker", "ship", "sp"]
    t0 = time.time()
    c = hglmm.Fit(Design(d, fixed, random)).fit()
    tc = time.time() - t0
    f = BlockFederatedFit(Design(d, fixed, random))
    t0 = time.time()
    f.fit(c.theta)                     # same start as the optimum: tests the fixed point
    tf = time.time() - t0
    cold = BlockFederatedFit(Design(d, fixed, random)).fit(np.r_[np.full(3, math.log(0.5)), math.log(5.0)])
    vc = c.laplace(cold.theta)[0]
    vf = cold.laplace(cold.theta)[0]
    n_holders = d.ship.nunique()
    cells_per_holder = len(d) / n_holders
    return dict(
        S=n_holders, P=P, cells=len(d), failures=int(d.failures.sum()),
        d_loglik_same_theta=abs(vc - vf),
        d_loglik_fit=abs(c.loglik - cold.loglik),
        d_beta_max=float(np.max(np.abs(c.beta - cold.beta))),
        d_sigma_max=float(np.max(np.abs(np.exp(c.theta[:3]) - np.exp(cold.theta[:3])))),
        d_phi_rel=float(abs(math.exp(c.theta[3] - cold.theta[3]) - 1)),
        warm_d_loglik=abs(c.loglik - f.loglik),
        rounds=cold.rounds, messages=cold.messages, floats=cold.floats_sent,
        floats_per_message=cold.floats_sent / cold.messages,
        local_dim=1 + P, touched=len(cold.holders[0]["touched"]),
        oneshot_floats=int(n_holders * cells_per_holder * (2 + len(fixed))),
        time_central=tc, time_fed=tf)


def main():
    rng = np.random.default_rng(20261102)
    rows = [compare(fleet(20, 4, 4, rng), 4) | dict(kind="equivalence", rep=r) for r in range(100)]
    for S in (5, 10, 20, 40, 80):
        for P in (2, 4, 8, 16):
            rows.append(compare(fleet(S, P, 4, rng), P) | dict(kind="scaling", rep=0))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "federated.csv"), index=False)
    eq = T[T.kind == "equivalence"]
    summ = dict(
        n=len(eq),
        max_d_loglik_same_theta=float(eq.d_loglik_same_theta.max()),
        max_d_loglik_fit=float(eq.d_loglik_fit.max()),
        max_d_beta=float(eq.d_beta_max.max()),
        max_d_sigma=float(eq.d_sigma_max.max()),
        median_d_sigma=float(eq.d_sigma_max.median()),
        median_rounds=float(eq.rounds.median()),
        median_floats=float(eq.floats.median()),
        median_oneshot_floats=float(eq.oneshot_floats.median()))
    with open(os.path.join(OUT, "federated.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))
    print(T[T.kind == "scaling"][["S", "P", "cells", "rounds", "floats", "oneshot_floats",
                                  "floats_per_message", "d_loglik_fit", "d_beta_max"]].to_string(index=False))


if __name__ == "__main__":
    main()
