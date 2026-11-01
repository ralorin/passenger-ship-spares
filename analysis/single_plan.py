"""
One population, operational range: how far each shortcut plan is from the
exact plan under a lognormal-Poisson rate.

Plans compared at level p, for a population expecting lam failures with an
ignored log-rate spread sigma:
  exact     smallest q with exact mixture cover >= p
  poisson   Poisson p-quantile (heterogeneity ignored)
  kappa     Poisson margin scaled by sqrt(1 + kappa^2), kappa = sigma sqrt(lam)
  negbin    negative binomial with the same mean and variance, the usual
            practitioner's choice (shape 1 / (exp(sigma^2) - 1))
For each plan we record the cover it delivers under the lognormal truth and
its distance in units from the exact plan.

    python single_plan.py
"""
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm, poisson

from plan_cover import Zone

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")
os.makedirs(OUT, exist_ok=True)

LAMS = np.geomspace(0.05, 50, 121)
SIGMAS = np.round(np.arange(0.1, 1.01, 0.1), 2)
LEVELS = (0.90, 0.95, 0.99)


def main():
    rows = []
    for sig in SIGMAS:
        for lam in LAMS:
            z = Zone([lam], sig, 0.0)
            cdf = z.wu @ z.G[0]
            kap = sig * math.sqrt(lam)
            phi = 1.0 / math.expm1(sig ** 2)
            for p in LEVELS:
                q_exact = int(np.argmax(cdf >= p))
                q_pois = int(poisson.ppf(p, lam))
                q_kap = int(math.ceil(lam + max(q_pois - lam, 0.0) * math.sqrt(1 + kap ** 2)))
                q_nb = int(nbinom.ppf(p, phi, phi / (phi + lam)))
                r = dict(lam=lam, sigma=sig, kappa=kap, level=p, q_exact=q_exact,
                         cover_exact=cdf[q_exact],
                         closed_form=norm.cdf(norm.ppf(p) / math.sqrt(1 + kap ** 2)))
                for name, q in (("poisson", q_pois), ("kappa", q_kap), ("negbin", q_nb)):
                    r[f"q_{name}"] = q
                    r[f"cover_{name}"] = cdf[min(q, z.smax)]
                    r[f"diff_{name}"] = q - q_exact
                rows.append(r)
    T = pd.DataFrame(rows)
    T["cf_error"] = T.closed_form - T.cover_poisson
    T.to_csv(os.path.join(OUT, "single_plan_grid.csv"), index=False)

    lam_bins = [0, 1, 5, 20, np.inf]
    sig_bins = [0, 0.35, 0.65, 1.01]
    T["lam_bin"] = pd.cut(T.lam, lam_bins, right=False).astype(str)
    T["sig_bin"] = pd.cut(T.sigma, sig_bins, right=False).astype(str)
    summ = {}
    for name in ("poisson", "kappa", "negbin"):
        g = T.groupby(["sig_bin", "lam_bin"])
        summ[name] = dict(
            share_equal=float((T[f"diff_{name}"] == 0).mean()),
            share_short=float((T[f"cover_{name}"] < T.level).mean()),
            worst_cover=float(T[f"cover_{name}"].min()),
            worst_shortfall_pp=float(100 * (T.level - T[f"cover_{name}"]).max()),
            max_units_short=int(-T[f"diff_{name}"].min()),
            max_units_over=int(T[f"diff_{name}"].max()),
            by_region=g.apply(lambda x, n=name: dict(
                min_cover=float(x[f"cover_{n}"].min()),
                share_equal=float((x[f"diff_{n}"] == 0).mean()),
                units=[int(x[f"diff_{n}"].min()), int(x[f"diff_{n}"].max())])).to_dict())
        summ[name]["by_region"] = {f"{k[0]} {k[1]}": v for k, v in summ[name]["by_region"].items()}
    at99 = T[T.level == 0.99]
    summ["closed_form_max_abs_error_pp"] = float(100 * T.cf_error.abs().max())
    summ["closed_form_max_abs_error_pp_99"] = float(100 * at99.cf_error.abs().max())
    summ["grid"] = dict(n_lam=len(LAMS), lam=[float(LAMS[0]), float(LAMS[-1])],
                        sigmas=SIGMAS.tolist(), levels=list(LEVELS), points=len(T))
    with open(os.path.join(OUT, "single_plan.json"), "w") as f:
        json.dump(summ, f, indent=1)

    for name in ("poisson", "kappa", "negbin"):
        s = summ[name]
        print(f"{name:8s} equal {s['share_equal']:.1%}  short of target {s['share_short']:.1%}"
              f"  worst cover {s['worst_cover']:.3f}  units {-s['max_units_short']}..+{s['max_units_over']}")
    print("closed form max |error| %.1f pp (99%%: %.1f pp)"
          % (summ["closed_form_max_abs_error_pp"], summ["closed_form_max_abs_error_pp_99"]))


if __name__ == "__main__":
    main()
