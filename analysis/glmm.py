"""
Hierarchical count model for cell aggregates, centralised and federated.

    y_c ~ NegBin(mu_c, phi)          (Poisson when phi = inf)
    log mu_c = log H_c + x_c' beta + sum_k b_k[level_k(c)]
    b_k ~ N(0, sigma_k^2 I),   k in {manufacturer, model, datacentre, source, ...}

y_c is the number of failures in cell c, H_c its exposure in device-hours. Cells
with zero failures enter the likelihood like any other cell; nothing is taken
on the log scale.

Estimation. The random effects b and the fixed effects beta are integrated by
a Laplace approximation at the joint mode, with a weakly informative
N(0, 5^2) prior on each non-intercept coefficient (close to the Laplace
analogue of REML, as in glmmTMB with REML = TRUE). The outer
problem maximises that approximate marginal likelihood over
theta = (log sigma_k, log phi). Standard errors of beta come from the joint
Hessian at the mode, i.e. they account for the uncertainty of b.

Federation. When one random-effect term (the *source*) has one level per data
holder and every cell belongs to exactly one holder, the joint Hessian is
block-arrowhead: each source's own effect u_s couples only with the global
coordinates (beta and the product effects) of its own cells. Eliminating u_s
locally by a Schur complement lets each holder send

    f_s, the gradient g_s and the Schur-complemented Hessian S_s over the
    global coordinates its cells touch, and log(-h_uu,s),

from which the coordinator reproduces the centralised Newton step and the
centralised Laplace likelihood exactly (``FederatedFit``). No cell aggregate
is sent; all messages are functions of the holder's cell aggregates only.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # small dense solves: threading only adds overhead

import math

import numpy as np
import pandas as pd
from scipy import optimize, sparse
from scipy.special import gammaln, polygamma

LOG2PI = math.log(2 * math.pi)
# Weakly informative N(0, 5^2) prior on each non-intercept coefficient (a log
# rate ratio). It is negligible for identified coefficients and keeps the mode
# finite when a subset records no failure in some covariate class.
BETA_PRIOR_SD = 5.0


def _nb_terms(y, eta, log_off, phi):
    """Log-likelihood, first and (negative) second derivative w.r.t. eta."""
    mu = np.exp(np.clip(eta + log_off, -700, 700))
    if phi is None or not np.isfinite(phi):
        ll = y * (eta + log_off) - mu - gammaln(y + 1)
        return ll, y - mu, mu
    ll = (gammaln(y + phi) - gammaln(phi) - gammaln(y + 1)
          + phi * (np.log(phi) - np.logaddexp(np.log(phi), eta + log_off))
          + y * (eta + log_off - np.logaddexp(np.log(phi), eta + log_off)))
    g = (y - mu) * phi / (phi + mu)
    w = (mu / (phi + mu)) * phi * (y + phi) / (phi + mu)      # no overflow of (phi+mu)^2
    return ll, g, w


class Design:
    """Fixed-effect matrix and random-effect index vectors for a set of cells."""

    def __init__(self, cells, fixed, random, y="failures", H="hours"):
        self.cells = cells.reset_index(drop=True)
        self.fixed = list(fixed)
        self.random = list(random)
        self.y = self.cells[y].to_numpy(float)
        self.log_off = np.log(self.cells[H].to_numpy(float))
        X = [np.ones(len(self.cells))] + [self.cells[f].to_numpy(float) for f in self.fixed]
        self.X = np.column_stack(X)
        self.levels, self.idx = {}, {}
        for k in self.random:
            codes, lev = pd.factorize(self.cells[k], sort=True)
            self.levels[k], self.idx[k] = list(lev), codes
        self.p = self.X.shape[1]
        self.q = [len(self.levels[k]) for k in self.random]
        self.offsets = np.cumsum([self.p] + self.q)[:-1]
        self.dim = self.p + sum(self.q)
        cols = [sparse.csr_matrix(self.X)]
        for k, nk in zip(self.random, self.q):
            cols.append(sparse.csr_matrix((np.ones(len(self.cells)),
                                           (np.arange(len(self.cells)), self.idx[k])),
                                          shape=(len(self.cells), nk)))
        self.J = sparse.hstack(cols).tocsr()


class HGLMM:
    """Laplace fit of the hierarchical count model (centralised)."""

    def __init__(self, design, family="nb"):
        self.d = design
        self.family = family

    def _unpack(self, theta):
        k = len(self.d.random)
        sig = np.exp(theta[:k])
        phi = math.exp(theta[k]) if self.family == "nb" else None
        return sig, phi

    def _prior_prec(self, sig):
        pr = np.zeros(self.d.dim)
        pr[1:self.d.p] = 1.0 / BETA_PRIOR_SD ** 2
        for o, n, s in zip(self.d.offsets, self.d.q, sig):
            pr[o:o + n] = 1.0 / s ** 2
        return pr

    def mode(self, theta, z0=None, tol=1e-9, maxit=100):
        d = self.d
        sig, phi = self._unpack(theta)
        prec = self._prior_prec(sig)
        z = np.zeros(d.dim) if z0 is None else z0.copy()
        if z0 is None:
            z[0] = math.log(max(d.y.sum(), 0.5) / np.exp(d.log_off).sum())
        obj_old = -np.inf
        for it in range(maxit):
            eta = d.J @ z
            ll, g, w = _nb_terms(d.y, eta, d.log_off, phi)
            obj = ll.sum() - 0.5 * np.sum(prec * z ** 2)
            grad = d.J.T @ g - prec * z
            Hn = (d.J.T @ sparse.diags(w) @ d.J).toarray() + np.diag(prec)
            step = np.linalg.solve(Hn, grad)
            t = 1.0
            while True:
                zn = z + t * step
                ll2, _, _ = _nb_terms(d.y, d.J @ zn, d.log_off, phi)
                obj2 = ll2.sum() - 0.5 * np.sum(prec * zn ** 2)
                if obj2 >= obj - 1e-10 or t < 1e-6:
                    break
                t *= 0.5
            z = zn
            if abs(obj2 - obj) < tol and np.max(np.abs(t * step)) < 1e-6:
                obj = obj2
                break
            obj = obj2
        eta = d.J @ z
        ll, g, w = _nb_terms(d.y, eta, d.log_off, phi)
        Hn = (d.J.T @ sparse.diags(w) @ d.J).toarray() + np.diag(prec)
        return z, Hn, ll.sum(), prec

    def laplace(self, theta, z0=None):
        z, Hn, ll, prec = self.mode(theta, z0)
        sig, _ = self._unpack(theta)
        logprior = -0.5 * np.sum(z[1:self.d.p] ** 2) / BETA_PRIOR_SD ** 2
        for o, n, s in zip(self.d.offsets, self.d.q, sig):
            b = z[o:o + n]
            logprior += -0.5 * np.sum(b ** 2) / s ** 2 - n * math.log(s) - 0.5 * n * LOG2PI
        sign, ldet = np.linalg.slogdet(Hn)
        val = ll + logprior + 0.5 * self.d.dim * LOG2PI - 0.5 * ldet
        return val, z, Hn

    def fit(self, theta0=None, lower=-6.0):
        k = len(self.d.random)
        nth = k + (1 if self.family == "nb" else 0)
        if theta0 is None:
            theta0 = np.r_[np.full(k, math.log(0.5)), [math.log(5.0)] if self.family == "nb" else []]
        cache = {"z": None}

        def obj(th):
            th = np.clip(th, lower, 12.0)
            v, z, _ = self.laplace(th, cache["z"])
            if not np.isfinite(v):                      # restart the inner mode cold
                v, z, _ = self.laplace(th, None)
            if not np.isfinite(v):
                return 1e12
            cache["z"] = z
            return -v

        if nth == 0:
            res = optimize.OptimizeResult(x=np.zeros(0), success=True, nfev=1)
        else:
            res = optimize.minimize(obj, theta0, method="L-BFGS-B",
                                    bounds=[(lower, 12.0)] * nth,
                                    options=dict(maxiter=500, ftol=1e-11, gtol=1e-6))
        self.theta = res.x
        self.loglik, self.z, self.H = self.laplace(self.theta, cache["z"])
        self.cov = np.linalg.inv(self.H)
        self.sigma, self.phi = self._unpack(self.theta)
        self.beta = self.z[:self.d.p]
        self.se_beta = np.sqrt(np.diag(self.cov)[:self.d.p])
        self.converged = bool(res.success)
        self.n_eval = res.nfev
        return self

    def fit_fixed(self, theta):
        """Evaluate at fixed variance parameters (no outer optimisation)."""
        self.theta = np.asarray(theta, float)
        self.loglik, self.z, self.H = self.laplace(self.theta)
        self.cov = np.linalg.inv(self.H)
        self.sigma, self.phi = self._unpack(self.theta)
        self.beta = self.z[:self.d.p]
        self.se_beta = np.sqrt(np.diag(self.cov)[:self.d.p])
        self.converged, self.n_eval = True, 1
        return self

    def effects(self, k):
        i = self.d.random.index(k)
        o, n = self.d.offsets[i], self.d.q[i]
        return dict(zip(self.d.levels[k], self.z[o:o + n]))

    def summary(self):
        out = dict(beta=dict(zip(["intercept"] + self.d.fixed, self.beta.tolist())),
                   se=dict(zip(["intercept"] + self.d.fixed, self.se_beta.tolist())),
                   sigma=dict(zip(self.d.random, self.sigma.tolist())),
                   phi=self.phi, loglik=float(self.loglik), converged=self.converged,
                   n_cells=len(self.d.y), failures=int(self.d.y.sum()))
        return out


def predictive_draws(fit, newcells, R=4000, rng=None, groups=None, param_unc=True,
                     sigma_draws=None, return_lambda=False):
    """Joint predictive draws of failure counts for new cells.

    Known levels use their conditional mode and the joint Laplace covariance;
    levels not seen in fitting are drawn from N(0, sigma_k^2), one draw per
    level shared by all its cells. The cell overdispersion (NB) is drawn per
    cell. If ``groups`` is given, draws are summed within groups.
    sigma_draws: optional (R, K) array of sigma values (variance-parameter
    uncertainty); otherwise sigma is fixed at its estimate.
    """
    rng = np.random.default_rng(rng)
    d = fit.d
    n = len(newcells)
    X = np.column_stack([np.ones(n)] + [newcells[f].to_numpy(float) for f in d.fixed])
    log_off = np.log(newcells["hours"].to_numpy(float))
    if param_unc:
        L = np.linalg.cholesky(fit.cov + 1e-12 * np.eye(d.dim))
        Z = fit.z[None, :] + (L @ rng.standard_normal((d.dim, R))).T
    else:
        Z = np.tile(fit.z, (R, 1))
    eta = Z[:, :d.p] @ X.T
    for i, k in enumerate(d.random):
        o = d.offsets[i]
        lev = {l: j for j, l in enumerate(d.levels[k])}
        vals = newcells[k].to_numpy()
        known = np.array([v in lev for v in vals])
        if known.any():
            cols = np.array([lev[v] for v in vals[known]])
            eta[:, known] += Z[:, o + cols]
        if (~known).any() and k not in getattr(fit, "no_new_draws", ()):
            new_levels, inv = np.unique(vals[~known].astype(str), return_inverse=True)
            s = fit.sigma[i] if sigma_draws is None else sigma_draws[:, i][:, None]
            draws = rng.standard_normal((R, len(new_levels))) * s
            eta[:, ~known] += draws[:, inv]
    mu = np.exp(eta + log_off[None, :])
    if fit.phi is not None:
        mu = mu * rng.gamma(fit.phi, 1.0 / fit.phi, size=mu.shape)
    if groups is None:
        lam, g = mu, None
    else:
        # given every latent draw, a group total is Poisson with the summed mean
        g, inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
        A = sparse.csr_matrix((np.ones(n), (np.arange(n), inv)), shape=(n, len(g)))
        lam = np.asarray(A.T @ mu.T).T
    lam = np.minimum(lam, 1e9)            # numerical guard only; never binding in the results
    y = rng.poisson(lam)
    if return_lambda:
        return y, lam, g
    return (y, g) if groups is not None else y


# Federated evaluation: each data holder eliminates its own source effect.
class FederatedFit:
    """Exact federated Laplace fit.

    ``source`` names the random-effect term whose levels are the data holders.
    Everything else (beta and the other random effects) is global and held by
    the coordinator. Messages are counted so the communication cost can be
    reported.
    """

    def __init__(self, design, source, family="nb"):
        self.d = design
        self.family = family
        self.source = source
        si = design.random.index(source)
        self.si = si
        self.gi = [i for i in range(len(design.random)) if i != si]
        # global coordinate map: beta then the non-source random effects
        self.gcols = list(range(design.p))
        for i in self.gi:
            o, n = design.offsets[i], design.q[i]
            self.gcols += list(range(o, o + n))
        self.gcols = np.array(self.gcols)
        Jg = design.J[:, self.gcols]
        codes = design.idx[source]
        self.holders = []
        for s in range(design.q[si]):
            m = codes == s
            Js = Jg[m]
            touched = np.unique(Js.indices)
            self.holders.append(dict(rows=np.where(m)[0], J=Js[:, touched].toarray(),
                                     touched=touched))
        self.rounds = 0
        self.floats_sent = 0
        self.messages = 0

    def _unpack(self, theta):
        k = len(self.d.random)
        sig = np.exp(theta[:k])
        phi = math.exp(theta[k]) if self.family == "nb" else None
        return sig, phi

    def local(self, h, zg_local, tau, phi, u0):
        """Holder's message for the current global coordinates."""
        y = self.d.y[h["rows"]]
        off = self.d.log_off[h["rows"]]
        base = h["J"] @ zg_local
        u = u0
        for _ in range(60):
            ll, g, w = _nb_terms(y, base + u, off, phi)
            gu = g.sum() - u / tau ** 2
            hu = w.sum() + 1.0 / tau ** 2
            du = gu / hu
            u += du
            if abs(du) < 1e-12:
                break
        ll, g, w = _nb_terms(y, base + u, off, phi)
        huu = w.sum() + 1.0 / tau ** 2
        grad = h["J"].T @ g
        Hgg = h["J"].T @ (w[:, None] * h["J"])
        hgu = h["J"].T @ w
        S = Hgg - np.outer(hgu, hgu) / huu
        f = ll.sum() - 0.5 * u ** 2 / tau ** 2 - math.log(tau) - 0.5 * LOG2PI
        m = len(h["touched"])
        self.floats_sent += 1 + m + m * (m + 1) // 2 + 1
        self.messages += 1
        return f, grad, S, math.log(huu), u

    # coordinator side
    def laplace(self, theta, zg0=None, u0=None, tol=1e-9, maxit=100):
        d = self.d
        sig, phi = self._unpack(theta)
        tau = sig[self.si]
        ng = len(self.gcols)
        prec = np.zeros(ng)
        prec[1:d.p] = 1.0 / BETA_PRIOR_SD ** 2
        pos = d.p
        for i in self.gi:
            prec[pos:pos + d.q[i]] = 1.0 / sig[i] ** 2
            pos += d.q[i]
        zg = np.zeros(ng) if zg0 is None else zg0.copy()
        if zg0 is None:
            zg[0] = math.log(max(d.y.sum(), 0.5) / np.exp(d.log_off).sum())
        us = np.zeros(len(self.holders)) if u0 is None else u0.copy()

        def evaluate(zg):
            F, G, Hs, L = 0.0, np.zeros(ng), np.zeros((ng, ng)), 0.0
            unew = us.copy()
            for s, h in enumerate(self.holders):
                f, g, S, lh, u = self.local(h, zg[h["touched"]], tau, phi, us[s])
                t = h["touched"]
                F += f
                G[t] += g
                Hs[np.ix_(t, t)] += S
                L += lh
                unew[s] = u
            self.rounds += 1
            F -= 0.5 * np.sum(prec * zg ** 2)
            G -= prec * zg
            Hs += np.diag(prec)
            return F, G, Hs, L, unew

        F, G, Hs, L, us = evaluate(zg)
        for _ in range(maxit):
            step = np.linalg.solve(Hs, G)
            t = 1.0
            while True:
                F2, G2, H2, L2, u2 = evaluate(zg + t * step)
                if F2 >= F - 1e-10 or t < 1e-6:
                    break
                t *= 0.5
            zg = zg + t * step
            done = abs(F2 - F) < tol and np.max(np.abs(t * step)) < 1e-6
            F, G, Hs, L, us = F2, G2, H2, L2, u2
            if done:
                break
        logprior = 0.0
        pos = d.p
        for i in self.gi:
            logprior += -d.q[i] * math.log(sig[i]) - 0.5 * d.q[i] * LOG2PI
            pos += d.q[i]
        # F already contains the quadratic prior terms of the global effects
        ldet = L + np.linalg.slogdet(Hs)[1]
        val = F + logprior + 0.5 * d.dim * LOG2PI - 0.5 * ldet
        return val, zg, us, Hs

    def fit(self, theta0, lower=-6.0):
        nth = len(theta0)
        cache = {"zg": None, "u": None}

        def obj(th):
            th = np.clip(th, lower, 12.0)
            v, zg, us, _ = self.laplace(th, cache["zg"], cache["u"])
            cache["zg"], cache["u"] = zg, us
            return -v

        res = optimize.minimize(obj, theta0, method="L-BFGS-B", bounds=[(lower, 12.0)] * nth,
                                options=dict(maxiter=500, ftol=1e-11, gtol=1e-6))
        self.theta = res.x
        self.loglik, self.zg, self.u, _ = self.laplace(self.theta, cache["zg"], cache["u"])
        self.sigma, self.phi = self._unpack(self.theta)
        self.beta = self.zg[:self.d.p]
        self.n_eval = res.nfev
        return self


def latent_shares(fit, cells, fixed_groups):
    """Shares of the latent (log-rate) variance of the device population.

    fixed_groups: {"temperature": ["temp_c"], "age": ["age_c", ...]}; the
    variance of each group's linear predictor is taken over the cells,
    weighted by exposure. Random terms contribute sigma_k^2. The NB cell
    overdispersion contributes trigamma(phi), the variance of log of a
    Gamma(phi, 1/phi) multiplier. Poisson sampling noise is excluded: it
    depends on exposure, not on the population.
    """
    w = cells["hours"].to_numpy(float)
    w = w / w.sum()
    comp = {}
    names = ["intercept"] + fit.d.fixed
    for gname, cols in fixed_groups.items():
        lp = sum(fit.beta[names.index(c)] * cells[c].to_numpy(float) for c in cols)
        m = np.sum(w * lp)
        comp[gname] = float(np.sum(w * (lp - m) ** 2))
    for k, s in zip(fit.d.random, fit.sigma):
        comp[k] = float(s ** 2)
    if fit.phi is not None:
        comp["cell"] = float(polygamma(1, fit.phi))
    tot = sum(comp.values())
    return {k: v / tot for k, v in comp.items()}, comp
