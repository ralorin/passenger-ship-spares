"""
Summarise the campaigns produced by design_sim.py.

Split out from design_sim.py because the delivered-cover step is the expensive
part and is best done once, on a cached grid, after all campaigns are fitted.
The latent standard deviations are binned to 0.05 before the zone-total
distribution is computed; that is an order of magnitude finer than their own
sampling error (Table~\\ref{tab:design} reports RMSEs of 0.08 and above), so the
binning changes no reported figure.

    python design_summary.py
"""
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

from multiproduct import total_pmf

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")
BIN = 0.05
NODES = 121
COVER_MAX_MEAN = 6.0   # zone totals above this are left uncomputed: the exact
                       # convolution grows with the mean and the identifiability
                       # results, which are the point of this table, do not need it
BOUND = 0.02
LEVEL = 99


@lru_cache(maxsize=100000)
def _cdf(P, total_mean, tau, omega):
    pmf, _ = total_pmf(np.full(P, total_mean / P), tau, omega, nodes=NODES)
    return tuple(np.cumsum(pmf))


def _b(x):
    return round(max(0.0, float(x)) / BIN) * BIN


def delivered(tau_hat, omega_hat, tau, omega, P, total_mean, level=LEVEL):
    c_hat = np.asarray(_cdf(P, total_mean, _b(tau_hat), _b(omega_hat)))
    q = int(np.argmax(c_hat >= level / 100.0))
    c_true = np.asarray(_cdf(P, total_mean, _b(tau), _b(omega)))
    return q, float(c_true[min(q, len(c_true) - 1)])


def main():
    R = pd.read_csv(os.path.join(OUT, "design_sim_reps.csv"))
    ok = R[R.ok].copy()
    print(f"{len(R)} campaigns, {R.ok.mean():.1%} fitted")

    rows = []
    for (S, M, P, m_exp, tau, omega), g in ok.groupby(["S", "M", "P", "m_exp", "tau", "omega"]):
        r = dict(S=int(S), M=int(M), P=int(P), m_exp=float(m_exp),
                 tau=float(tau), omega=float(omega), reps=int(len(g)),
                 failures_median=float(g.failures.median()),
                 tau_bound=float((g.tau_hat <= BOUND).mean()),
                 omega_bound=float((g.omega_hat <= BOUND).mean()),
                 tau_median=float(g.tau_hat.median()),
                 omega_median=float(g.omega_hat.median()),
                 tau_rmse=float(np.sqrt(((g.tau_hat - tau) ** 2).mean())),
                 omega_rmse=float(np.sqrt(((g.omega_hat - omega) ** 2).mean())))
        if tau != omega:
            bigger = "tau" if tau > omega else "omega"
            got = np.where(g.tau_hat > g.omega_hat, "tau", "omega")
            r["allocation_recovered"] = float((got == bigger).mean())
        else:
            r["allocation_recovered"] = float("nan")
        total_mean = float(m_exp) * int(P)
        if total_mean <= COVER_MAX_MEAN:
            q, cov = [], []
            for th, oh in zip(g.tau_hat, g.omega_hat):
                qq, cc = delivered(th, oh, tau, omega, int(P), total_mean)
                q.append(qq)
                cov.append(cc)
            r["plan99_median"] = float(np.median(q))
            r["cover_mean"] = float(np.mean(cov))
            r["cover_p10"] = float(np.quantile(cov, 0.10))
        else:
            r["plan99_median"] = r["cover_mean"] = r["cover_p10"] = float("nan")
        rows.append(r)
        print(f"  S={S} M={M} m={m_exp} tau={tau} om={omega} done", flush=True)

    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "design_sim_summary.csv"), index=False)
    json.dump(dict(reps=int(R.groupby(["S", "M", "P", "m_exp", "tau", "omega"]).size().max()),
                   eps=float(R.eps.iloc[0]), bound=BOUND, bin=BIN, nodes=NODES,
                   level=LEVEL, summary=T.to_dict("records")),
              open(os.path.join(OUT, "design_sim.json"), "w"), indent=1, default=float)

    pd.set_option("display.width", 220)
    print("\nEvery value below is simulated under assumed parameters.\n")
    print(T[["tau", "omega", "S", "M", "m_exp", "failures_median", "tau_bound",
             "omega_bound", "tau_rmse", "omega_rmse", "allocation_recovered",
             "plan99_median", "cover_mean", "cover_p10"]]
          .sort_values(["tau", "omega", "S", "M", "m_exp"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
