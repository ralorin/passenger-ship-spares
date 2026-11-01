"""
Cover delivered by a spares plan when latent heterogeneity is ignored.

Everything in this file is analytical or exact numerical quadrature. No device
data of any kind enter, so the results hold for any equipment whose mission
failure count is modelled as a mixed Poisson count with an exposure offset.

Setting. A population expects lambda failures over the mission under a
predictor that ignores latent log-rate spread of standard deviation sigma. The
ignored spread acts multiplicatively and preserves the mean:

    Y | u ~ Poisson(lambda * exp(u - sigma^2/2)),   u ~ N(0, sigma^2).

The predictor embarks q_p = min{q : F_Pois(q; lambda) >= p}.

Proposition 1 (exact).  The cover actually delivered is

    C(q_p) = E_u F_Pois(q_p; lambda e^{u - sigma^2/2}),

a one-dimensional integral. It depends on lambda and sigma SEPARATELY: on
lambda through the integer plan quantile q_p and through the conditional
Poisson, and on sigma through the mixing measure.

Proposition 2 (first-order approximation only).  If sigma is small enough for
a first-order expansion in u and lambda large enough for a normal
approximation to the count, then with kappa = sigma * sqrt(lambda)

    C(q_p) ~= Phi(z_p / sqrt(1 + kappa^2)),

and the safety margin above the mean inflates by ~ sqrt(1 + kappa^2).

This script quantifies the three things an author must not assume:

  (1) kappa alone does not determine the exact cover. For a fixed kappa we
      vary (lambda, sigma) along kappa = sigma*sqrt(lambda) and report the
      spread of the exact cover.
  (2) where the closed form is usable, as max |rule - exact| over regions of
      (kappa, lambda), and with what sign.
  (3) the error of the sqrt(1+kappa^2) margin factor against the exact
      required quantile.

    python theory_cover.py
"""
import json
import os

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm, poisson

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")
os.makedirs(OUT, exist_ok=True)

LEVELS = (90, 95, 99)
NODES = 4001   # verified stable to <1e-6 against 16001 (check_quadrature)


def exact_cover(q, lam, sigma, nodes=NODES):
    """C(q) = E_u F_Pois(q; lam e^{u - sigma^2/2}), u ~ N(0, sigma^2).

    Composite Simpson rule on u over [-10 sigma, 10 sigma], renormalised by the
    quadrature mass of the Gaussian weight. Gauss-Hermite is NOT used: the
    integrand F_Pois(q; lam e^u) is a sigmoid in u whose transition narrows as
    lam grows, and Gauss-Hermite nodes are too sparse there -- with 60 nodes it
    misplaces the cover by up to 4 percentage points at sigma >= 1. The grid
    below is verified in check_quadrature() to be stable to < 1e-6.
    """
    q = np.atleast_1d(np.asarray(q, float))
    lam = np.atleast_1d(np.asarray(lam, float))
    if sigma <= 0:
        return poisson.cdf(q, lam)
    n = int(nodes) | 1                       # odd number of points for Simpson
    u = np.linspace(-10.0 * sigma, 10.0 * sigma, n)
    w = np.ones(n)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    w = w * norm.pdf(u, scale=sigma)
    w = w / w.sum()
    out = np.zeros(np.broadcast(q, lam).shape, float)
    for ui, wi in zip(u, w):
        if wi > 0:
            out += wi * poisson.cdf(q, lam * np.exp(ui - sigma ** 2 / 2))
    return out


def plan_quantile(lam, p):
    """The plan of a predictor that ignores the spread: a Poisson quantile."""
    return poisson.ppf(p / 100.0, lam)


def exact_plan(lam, sigma, p, qmax_mult=40, nodes=2001):
    """Smallest integer q with exact cover >= p under the mixture."""
    lam = float(lam)
    hi = int(max(10, np.ceil(lam + qmax_mult * max(1.0, np.sqrt(lam)) * (1 + sigma))))
    q = np.arange(0, hi + 1)
    c = exact_cover(q, np.full_like(q, lam, dtype=float), sigma, nodes)
    idx = np.argmax(c >= p / 100.0)
    return int(q[idx]) if c[idx] >= p / 100.0 else int(hi)


def check_quadrature():
    """Convergence of the Simpson grid, and the error of Gauss-Hermite.

    Reported so that the word "exact" in Proposition 1 is backed by a number:
    the grid used is stable to `max_abs_change_grid`, while 60-node
    Gauss-Hermite -- the natural first choice for a Gaussian mixing measure --
    is off by `max_abs_error_gauss_hermite60`.
    """
    grid, gh = [], []
    for lam in (0.05, 0.5, 2.0, 10.0, 100.0, 500.0, 2000.0):
        for s in (0.05, 0.1, 0.5, 1.0, 1.5, 2.0):
            for p in LEVELS:
                q = plan_quantile(lam, p)
                ref = float(exact_cover(q, lam, s, 16001)[0])
                grid.append(abs(float(exact_cover(q, lam, s)[0]) - ref))
                z, w = hermgauss(60)
                u = np.sqrt(2.0) * s * z
                v = sum(wi * float(poisson.cdf(q, lam * np.exp(ui - s ** 2 / 2)))
                        for ui, wi in zip(u, w)) / np.sqrt(np.pi)
                gh.append(abs(v - ref))
    return dict(max_abs_change_grid=float(max(grid)),
                max_abs_error_gauss_hermite60=float(max(gh)),
                grid_nodes=NODES)


# ---------------------------------------------------------------- (1)
def kappa_insufficiency():
    """Same kappa, different (lambda, sigma): how far apart is the exact cover?

    If kappa were sufficient, every row of a block would share one cover.
    """
    lams = (0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 200.0, 1000.0)
    blocks = []
    for kap in (0.25, 0.5, 1.0, 2.0):
        rows = []
        for lam in lams:
            s = kap / np.sqrt(lam)
            for p in LEVELS:
                q = plan_quantile(lam, p)
                rows.append(dict(kappa=kap, lam=lam, sigma=float(s), p=p, q=float(q),
                                 exact=float(exact_cover(q, lam, s)[0]),
                                 rule=float(norm.cdf(norm.ppf(p / 100) / np.sqrt(1 + kap ** 2)))))
        d = pd.DataFrame(rows)
        spread = {f"p{p}": float(d[d.p == p].exact.max() - d[d.p == p].exact.min())
                  for p in LEVELS}
        blocks.append(dict(kappa=kap, spread=spread, rows=d.to_dict("records")))
    return blocks


# ---------------------------------------------------------------- (2)
def rule_domain():
    """max |rule - exact| by region of (kappa, lambda), and its sign."""
    lam = np.concatenate([np.linspace(0.05, 1, 20), np.linspace(1.25, 10, 36),
                          np.linspace(11, 100, 90), np.linspace(105, 1000, 90)])
    sig = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5])
    rows = []
    for s in sig:
        for p in LEVELS:
            z = norm.ppf(p / 100)
            q = plan_quantile(lam, p)
            ex = exact_cover(q, lam, s)
            ru = norm.cdf(z / np.sqrt(1 + (s * np.sqrt(lam)) ** 2))
            for l, e, r in zip(lam, ex, np.atleast_1d(ru)):
                rows.append(dict(lam=float(l), sigma=float(s), p=p,
                                 kappa=float(s * np.sqrt(l)), exact=float(e),
                                 rule=float(r), gap=float(r - e)))
    D = pd.DataFrame(rows)
    kb = [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, np.inf)]
    lb = [(0, 1), (1, 5), (5, 20), (20, np.inf)]
    grid = {}
    for kl, kh in kb:
        for ll, lh in lb:
            d = D[(D.kappa >= kl) & (D.kappa < kh) & (D.lam >= ll) & (D.lam < lh)]
            if len(d):
                grid[f"k[{kl},{kh}) L[{ll},{lh})"] = dict(
                    n=int(len(d)), max_abs=float(d.gap.abs().max()),
                    median_abs=float(d.gap.abs().median()),
                    lo=float(d.gap.min()), hi=float(d.gap.max()))
    return D, dict(grid=grid, max_abs_overall=float(D.gap.abs().max()),
                   signed_range=[float(D.gap.min()), float(D.gap.max())],
                   n_grid=int(len(D)))


# ---------------------------------------------------------------- (3)
def margin_factor():
    """Error of the sqrt(1+kappa^2) inflation against the exact plan."""
    rows = []
    for lam in (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0):
        for s in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
            kap = s * np.sqrt(lam)
            for p in LEVELS:
                z = norm.ppf(p / 100)
                q0 = plan_quantile(lam, p)                       # ignores the spread
                qs = exact_plan(lam, s, p)                       # exact, under the mixture
                m0, ms = q0 - lam, qs - lam                      # margins above the mean
                pred = np.sqrt(1 + kap ** 2)
                obs = ms / m0 if m0 > 0 else np.nan
                rows.append(dict(lam=lam, sigma=s, kappa=float(kap), p=p,
                                 q_ignoring=float(q0), q_exact=float(qs),
                                 margin_ignoring=float(m0), margin_exact=float(ms),
                                 factor_exact=float(obs), factor_rule=float(pred),
                                 rel_err=float((pred - obs) / obs) if m0 > 0 and obs > 0 else np.nan,
                                 units_err=float(lam + pred * m0 - qs)))
    M = pd.DataFrame(rows)
    ok = M.dropna(subset=["rel_err"])
    by = {}
    for name, m in (("lam<10", ok.lam < 10), ("10<=lam<100", (ok.lam >= 10) & (ok.lam < 100)),
                    ("lam>=100", ok.lam >= 100)):
        d = ok[m]
        by[name] = dict(n=int(len(d)), median_rel_err=float(d.rel_err.median()),
                        max_abs_rel_err=float(d.rel_err.abs().max()),
                        max_abs_units_err=float(d.units_err.abs().max()))
    return M, dict(by_lambda=by,
                   max_abs_rel_err_overall=float(ok.rel_err.abs().max()),
                   max_abs_units_err_overall=float(ok.units_err.abs().max()))


def main():
    quad = check_quadrature()
    ins = kappa_insufficiency()
    D, dom = rule_domain()
    M, mar = margin_factor()
    D.to_csv(os.path.join(OUT, "theory_rule_grid.csv"), index=False)
    M.to_csv(os.path.join(OUT, "theory_margin.csv"), index=False)
    res = dict(quadrature=quad, kappa_insufficiency=ins, rule_domain=dom, margin=mar)
    json.dump(res, open(os.path.join(OUT, "theory_cover.json"), "w"), indent=1, default=float)

    print("quadrature:", quad)
    print("\n(1) same kappa, different (lambda, sigma) -> spread of the EXACT cover")
    for b in ins:
        print(f"  kappa={b['kappa']}: spread 99% = {b['spread']['p99']:.4f}  "
              f"95% = {b['spread']['p95']:.4f}  90% = {b['spread']['p90']:.4f}")
        d = pd.DataFrame(b["rows"])
        d = d[d.p == 99]
        print("    lam:   " + "  ".join(f"{x:7.4g}" for x in d.lam))
        print("    sigma: " + "  ".join(f"{x:7.3f}" for x in d.sigma))
        print("    exact: " + "  ".join(f"{x:7.4f}" for x in d.exact))
        print(f"    rule (one value for this kappa): {d.rule.iloc[0]:.4f}")

    print("\n(2) closed form minus exact, by region")
    for k, v in dom["grid"].items():
        print(f"  {k:28s} n={v['n']:5d} max|gap|={v['max_abs']:.3f} "
              f"median={v['median_abs']:.3f} range=[{v['lo']:+.3f},{v['hi']:+.3f}]")
    print(f"  overall max|gap| = {dom['max_abs_overall']:.3f}, "
          f"signed range = [{dom['signed_range'][0]:+.3f}, {dom['signed_range'][1]:+.3f}]")

    print("\n(3) margin inflation: sqrt(1+kappa^2) against the exact plan")
    for k, v in mar["by_lambda"].items():
        print(f"  {k:14s} n={v['n']:4d} median rel.err={v['median_rel_err']:+.3f} "
              f"max|rel.err|={v['max_abs_rel_err']:.3f} max|units|={v['max_abs_units_err']:.2f}")


if __name__ == "__main__":
    main()
