"""
Same Laplace fit as glmm.HGLMM, written for larger designs.

The last random term in `random` is eliminated inside every Newton step. Each
cell belongs to exactly one level of that term, so its block of the Hessian is
diagonal and the elimination costs nothing; what is left is a dense system in
the coefficients and the remaining effects. Put the term with most levels last
(ship x product in this paper). The objective, the prior on beta and the
optimiser are those of glmm.py, so both fits agree to the stopping tolerance.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math

import numpy as np
from scipy import optimize, sparse
from scipy.stats import chi2

from glmm import BETA_PRIOR_SD, LOG2PI, Design, _nb_terms

__all__ = ["Design", "Fit", "fit", "lrt_boundary"]

GRAD_TOL = 0.05     # largest projected gradient of an accepted fit (criterion per unit of log sd)
# sensitivity checks of the rule, off by default:
#   SPARES_POLISH=1        the kept run is continued with much tighter tolerances
#   SPARES_KEEP_FAILED=1   bootstrap refits with no accepted run are kept (best run)
POLISH = os.environ.get("SPARES_POLISH") == "1"
KEEP_FAILED = os.environ.get("SPARES_KEEP_FAILED") == "1"


class Fit:
    def __init__(self, design, family="nb"):
        d = self.d = design
        self.family = family
        k = len(d.random)
        self.k = k
        if k:
            o = d.offsets[-1]
            self.na = o                      # beta and every term but the last
            self.Jc_idx = d.idx[d.random[-1]]
            self.nc = d.q[-1]
        else:
            self.na, self.Jc_idx, self.nc = d.dim, np.zeros(len(d.y), int), 0
        self.Ja = d.J[:, :self.na].tocsc()

    def _unpack(self, theta):
        sig = np.exp(theta[:self.k])
        phi = math.exp(theta[self.k]) if self.family == "nb" else None
        return sig, phi

    def _prec(self, sig):
        d = self.d
        pr = np.zeros(d.dim)
        pr[1:d.p] = 1.0 / BETA_PRIOR_SD ** 2
        for o, n, s in zip(d.offsets, d.q, sig):
            pr[o:o + n] = 1.0 / s ** 2
        return pr

    def _blocks(self, w, prec):
        """Pieces of the negative Hessian: dense a-block, sparse a-c block, diagonal c."""
        Ja = self.Ja
        Haa = (Ja.T @ sparse.diags(w) @ Ja).toarray() + np.diag(prec[:self.na])
        if not self.nc:
            return Haa, None, None
        hcc = np.bincount(self.Jc_idx, w, self.nc) + prec[self.na:]
        Jc = sparse.csc_matrix((w, (np.arange(len(w)), self.Jc_idx)), shape=(len(w), self.nc))
        Hac = (Ja.T @ Jc).toarray()
        return Haa, Hac, hcc

    def _solve(self, Haa, Hac, hcc, g):
        if hcc is None:
            return np.linalg.solve(Haa, g)
        ga, gc = g[:self.na], g[self.na:]
        Sa = Haa - (Hac / hcc) @ Hac.T
        da = np.linalg.solve(Sa, ga - Hac @ (gc / hcc))
        dc = (gc - Hac.T @ da) / hcc
        return np.r_[da, dc]

    def _logdet(self, Haa, Hac, hcc):
        if hcc is None:
            return np.linalg.slogdet(Haa)[1]
        Sa = Haa - (Hac / hcc) @ Hac.T
        return np.sum(np.log(hcc)) + np.linalg.slogdet(Sa)[1]

    def mode(self, theta, z0=None, tol=1e-9, maxit=100):
        d = self.d
        sig, phi = self._unpack(theta)
        prec = self._prec(sig)
        z = np.zeros(d.dim) if z0 is None else z0.copy()
        if z0 is None:
            z[0] = math.log(max(d.y.sum(), 0.5) / np.exp(d.log_off).sum())
        obj = -np.inf
        for _ in range(maxit):
            ll, g, w = _nb_terms(d.y, d.J @ z, d.log_off, phi)
            obj = ll.sum() - 0.5 * np.sum(prec * z ** 2)
            grad = d.J.T @ g - prec * z
            step = self._solve(*self._blocks(w, prec), grad)
            t = 1.0
            while True:
                zn = z + t * step
                ll2, _, _ = _nb_terms(d.y, d.J @ zn, d.log_off, phi)
                obj2 = ll2.sum() - 0.5 * np.sum(prec * zn ** 2)
                if obj2 >= obj - 1e-10 or t < 1e-6:
                    break
                t *= 0.5
            z = zn
            done = abs(obj2 - obj) < tol and np.max(np.abs(t * step)) < 1e-6
            obj = obj2
            if done:
                break
        ll, g, w = _nb_terms(d.y, d.J @ z, d.log_off, phi)
        return z, self._blocks(w, prec), ll.sum()

    def laplace(self, theta, z0=None):
        z, blocks, ll = self.mode(theta, z0)
        sig, _ = self._unpack(theta)
        d = self.d
        logprior = -0.5 * np.sum(z[1:d.p] ** 2) / BETA_PRIOR_SD ** 2
        for o, n, s in zip(d.offsets, d.q, sig):
            b = z[o:o + n]
            logprior += -0.5 * np.sum(b ** 2) / s ** 2 - n * math.log(s) - 0.5 * n * LOG2PI
        val = ll + logprior + 0.5 * d.dim * LOG2PI - 0.5 * self._logdet(*blocks)
        return val, z, blocks

    def projected_gradient(self, theta, lower, ub, eps=1e-5):
        """Central-difference gradient of the criterion in theta, with the
        components that push against an active bound set to zero."""
        g = np.zeros(len(theta))
        for i in range(len(theta)):
            e = np.zeros(len(theta))
            e[i] = eps
            g[i] = (self.laplace(np.clip(theta + e, lower, ub))[0]
                    - self.laplace(np.clip(theta - e, lower, ub))[0]) / (2 * eps)
            if theta[i] <= lower + 1e-8 and g[i] < 0:
                g[i] = 0.0
            if theta[i] >= ub[i] - 1e-8 and g[i] > 0:
                g[i] = 0.0
        return g

    def fit(self, theta0=None, lower=-6.0, upper=12.0, sigma_max=None, rule=False):
        """sigma_max, if given, caps every random-effect sd (not the NB shape); it
        keeps weakly identified terms from drifting to absurd values.

        rule=True applies the convergence rule. The criterion is maximised
        from three starts: theta0, the default (sds 0.5, NB shape 5)
        and small spreads (sds e^-3, shape 1), which reaches the solutions that
        put the dispersion in the gamma multiplier. A run is accepted when the
        largest projected gradient, by central differences, is at most GRAD_TOL,
        whatever the optimiser's exit message; the accepted run with the highest
        criterion is kept, and with none accepted the fit is declared failed
        (self.accepted False). self.start_won is the index of the start kept and
        self.crit_gain its criterion minus that of the first start."""
        nth = self.k + (1 if self.family == "nb" else 0)
        default = np.r_[np.full(self.k, math.log(0.5)),
                        [math.log(5.0)] if self.family == "nb" else []]
        if theta0 is None:
            theta0 = default
        ub = np.full(nth, upper)
        if sigma_max is not None:
            ub[:self.k] = math.log(sigma_max)
        theta0 = np.minimum(theta0, ub)
        cache = {"z": None}

        def obj(th):
            th = np.clip(th, lower, ub)
            v, z, _ = self.laplace(th, cache["z"])
            if not np.isfinite(v):
                v, z, _ = self.laplace(th, None)
            if not np.isfinite(v):
                return 1e12
            cache["z"] = z
            return -v

        def run(start):
            return optimize.minimize(obj, start, method="L-BFGS-B",
                                     bounds=list(zip(np.full(nth, lower), ub)),
                                     options=dict(maxiter=500, ftol=1e-11, gtol=1e-6))

        self.accepted, self.grad_max, self.n_starts = True, 0.0, 1
        self.start_won, self.crit_gain = 0, 0.0
        z_end = None
        if nth:
            res = run(theta0)
            self.theta, self.converged, self.n_eval = res.x, bool(res.success), res.nfev
            z_end = cache["z"]
            if rule:
                small = np.minimum(np.r_[np.full(self.k, -3.0), [0.0] if self.family == "nb" else []], ub)
                tried, starts = [], []
                for start in (theta0, default, small):
                    if any(np.allclose(start, s) for s in starts):
                        continue
                    starts.append(start)
                    if tried:
                        cache["z"] = None
                        r = run(start)
                    else:
                        r = res
                    gm = float(np.max(np.abs(self.projected_gradient(r.x, lower, ub))))
                    tried.append((gm <= GRAD_TOL, -r.fun, gm, r))
                ok = [t for t in tried if t[0]]
                best = max(ok, key=lambda t: t[1]) if ok else min(tried, key=lambda t: t[2])
                self.theta_rule = best[3].x.copy()      # the run the rule keeps, before any polish
                if POLISH and ok:
                    cache["z"] = None
                    r = optimize.minimize(obj, best[3].x, method="L-BFGS-B",
                                          bounds=list(zip(np.full(nth, lower), ub)),
                                          options=dict(maxiter=5000, ftol=1e-15, gtol=1e-10))
                    if -r.fun >= best[1] - 1e-12:
                        gm = float(np.max(np.abs(self.projected_gradient(r.x, lower, ub))))
                        polished = (True, -r.fun, gm, r)
                        tried[tried.index(best)] = polished
                        best = polished
                if best[3] is not res:
                    z_end = None
                self.theta, self.converged = best[3].x, bool(best[3].success)
                self.accepted, self.grad_max, self.n_starts = bool(ok), best[2], len(tried)
                self.start_won = next(i for i, t in enumerate(tried) if t is best)
                self.crit_gain = float(best[1] - tried[0][1])
                self.n_eval = sum(t[3].nfev for t in tried)
        else:
            self.theta, self.converged, self.n_eval = np.zeros(0), True, 1
        self.loglik, self.z, self.blocks = self.laplace(self.theta, z_end)
        self.sigma, self.phi = self._unpack(self.theta)
        self.beta = self.z[:self.d.p]
        Haa, Hac, hcc = self.blocks
        Sa = Haa if hcc is None else Haa - (Hac / hcc) @ Hac.T
        self.cov_a = np.linalg.inv(Sa)        # covariance of beta and the non-eliminated effects
        self.se_beta = np.sqrt(np.diag(self.cov_a)[:self.d.p])
        return self

    def draw_latent(self, R, rng):
        """Joint draws from the Gaussian approximation at the mode, shape (R, dim)."""
        L = np.linalg.cholesky(self.cov_a + 1e-12 * np.eye(self.na))
        za = self.z[:self.na] + (L @ rng.standard_normal((self.na, R))).T
        if not self.nc:
            return za
        _, Hac, hcc = self.blocks
        # c | a is Gaussian with precision diag(hcc) and mean shifted by the a-draw
        zc = (self.z[self.na:] - ((za - self.z[:self.na]) @ Hac) / hcc
              + rng.standard_normal((R, self.nc)) / np.sqrt(hcc))
        return np.hstack([za, zc])


def fit(cells, fixed, random, family="nb", **kw):
    return Fit(Design(cells, fixed, random), family).fit(**kw)


def lrt_boundary(ll_full, ll_reduced):
    """LRT for one variance component at zero: 50:50 mixture of chi2_0 and chi2_1."""
    t = max(0.0, 2.0 * (ll_full - ll_reduced))
    return t, 0.5 * chi2.sf(t, 1) if t > 0 else 1.0


# variance-parameter uncertainty
def at_theta(f, theta):
    """Conditional mode and covariance of the coefficients and effects at a given
    theta, on the data of fit f. Returns a copy of f evaluated there."""
    g = Fit(f.d, f.family)
    g.theta = np.asarray(theta, float)
    g.loglik, g.z, g.blocks = g.laplace(g.theta, f.z)
    g.sigma, g.phi = g._unpack(g.theta)
    g.beta = g.z[:g.d.p]
    Haa, Hac, hcc = g.blocks
    Sa = Haa if hcc is None else Haa - (Hac / hcc) @ Hac.T
    g.cov_a = np.linalg.inv(Sa)
    g.se_beta = np.sqrt(np.diag(g.cov_a)[:g.d.p])
    g.converged, g.n_eval = True, 1
    return g


def simulate_counts(f, rng):
    """Counts from the fitted model: coefficients at their estimates, fresh
    random effects at the fitted sds, fresh NB multipliers."""
    d = f.d
    eta = d.X @ f.beta + d.log_off
    for i, k in enumerate(d.random):
        eta = eta + rng.normal(0, f.sigma[i], d.q[i])[d.idx[k]]
    mu = np.exp(np.minimum(eta, 30))
    if f.phi is not None:
        mu = mu * rng.gamma(f.phi, 1.0 / f.phi, len(mu))
    return rng.poisson(mu).astype(float)


def bootstrap_theta(f, B, rng, rule=False, info=None, fit_cls=None, **fit_kw):
    """Parametric bootstrap with refit: B draws of theta from the sampling
    distribution of the estimator, used as approximate posterior draws.

    With rule=True every refit follows the convergence rule of Fit.fit; a refit
    declared failed is dropped and counted in info["failed"] (info, a dict, also
    receives the restarts and the largest projected gradient kept)."""
    out = []
    make = fit_cls or (lambda d: Fit(d, f.family))
    for _ in range(B):
        d = Design.__new__(Design)
        d.__dict__.update(f.d.__dict__)
        d.y = simulate_counts(f, rng)
        g = make(d).fit(theta0=f.theta.copy(), rule=rule, **fit_kw)
        if info is not None:
            info["failed"] = info.get("failed", 0) + int(not g.accepted)
            # "restarted": another start than the estimate reached a higher criterion (by > 1e-3)
            info["restarted"] = info.get("restarted", 0) + int(g.start_won > 0 and g.crit_gain > 1e-3)
            info["grad_max"] = max(info.get("grad_max", 0.0), g.grad_max if g.accepted else 0.0)
            # for the sensitivity checks: every refit, with its status and its unpolished theta
            info.setdefault("all", []).append((bool(g.accepted), getattr(g, "theta_rule", g.theta), g.theta))
        if g.accepted or KEEP_FAILED:
            out.append(g.theta)
    return np.array(out)
