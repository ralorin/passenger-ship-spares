"""
Further summaries of the campaign study from the stored replicates: Monte
Carlo intervals of the rejection and cover rates, profile intervals with the
boundary-corrected critical value, and the single-mission designs.

    python campaign_extra.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import beta, chi2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")
CRIT_MIX = chi2.ppf(0.90, 1)     # 5 % point of the 50:50 mixture of chi2_0 and chi2_1


def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def main():
    R = pd.read_csv(os.path.join(OUT, "campaign_reps.csv.gz"))
    T = pd.read_csv(os.path.join(OUT, "campaign_summary.csv")).set_index("cell")
    out = {}

    rej = []
    for comp in ("tau", "omega"):
        for cell, r in T[T[comp] == 0].iterrows():
            g = R[R.cell == cell]
            k, n = int((g[f"p_{comp}"] < 0.05).sum()), len(g)
            rej.append(dict(comp=comp, cell=cell, M=r.M, k=k, n=n, rate=k / n, ci=cp(k, n)))
    E = pd.DataFrame(rej)
    top = E.loc[E.rate.idxmax()]
    out["typeI"] = dict(cells=len(E), median=float(E.rate.median()), max=float(top.rate),
                        max_ci=list(top.ci), cells_upper_below_nominal=int(sum(c[1] < 0.05 for c in E.ci)),
                        cells_lower_above_nominal=int(sum(c[0] > 0.05 for c in E.ci)),
                        omega_M1_max=float(E[(E.comp == "omega") & (E.M == 1)].rate.max()))

    # profile intervals: the stored lower end is 0 when the LRT is below the chi2_1
    # point; the boundary-consistent rule uses the mixture point instead
    ci = []
    for cell, r in T[T.reps > 0].iterrows():
        g = R[(R.cell == cell) & R.ci_tau_lo.notna()]
        if not len(g):
            continue
        for comp in ("tau", "omega"):
            true = r[comp]
            lo, hi = g[f"ci_{comp}_lo"], g[f"ci_{comp}_hi"]
            stored = (lo <= true + 1e-9) & (true <= hi)
            if true == 0:
                corrected = g[f"lrt_{comp}"] <= CRIT_MIX
            else:
                corrected = stored
            ci.append(dict(cell=cell, comp=comp, true=true, n=len(g), stored=float(stored.mean()),
                           corrected=float(corrected.mean())))
    C = pd.DataFrame(ci)
    out["profile"] = dict(cells=int(C.cell.nunique()), reps=int(C.n.iloc[0]),
                          stored_lo=float(C.stored.min()), stored_hi=float(C.stored.max()),
                          corrected_lo=float(C.corrected.min()), corrected_hi=float(C.corrected.max()),
                          zero_stored=float(C[C.true == 0].stored.mean()),
                          zero_corrected=float(C[C.true == 0].corrected.mean()),
                          crit_mix=float(CRIT_MIX))

    # single mission: interaction against mission-level variation
    one = T[(T.M == 1) & (T.P == 4) & (T.m == 1) & (T.eps == 0.3) & (~T.unbal)]
    out["M1"] = dict(power_omega_max=float(one[(one.omega > 0)].reject_omega.max()),
                     power_tau_by_S={int(S): float(g[(g.tau == 0.42)].reject_tau.iloc[0])
                                     for S, g in one.groupby("S")},
                     omega_at_bound_median=float(one.omega_at_bound.median()),
                     joint_pred_cover_min=float(one.joint_predictive_cover.min()),
                     joint_pred_cover_S10=float(one[one.S >= 10].joint_predictive_cover.min()))

    # cover of plans: Monte Carlo intervals and worst cells
    four = T[(T.M == 4) & (T.P == 4) & (T.m == 1) & (T.eps == 0.3) & (~T.unbal)]
    out["cover"] = dict(pred_mcse_max=float(T.joint_predictive_cover_mcse.max()),
                        pred_min_S10=float(T[(T.S >= 10) & (T.M >= 2)].joint_predictive_cover.min()),
                        pred_short_share_S10=float(T[(T.S >= 10) & (T.M >= 2)].joint_predictive_short.max()),
                        pred_p10_min_S10=float(T[(T.S >= 10) & (T.M >= 2)].joint_predictive_cover_p10.min()),
                        plug_S5=float(four[four.S == 5].joint_plugin_cover.mean()),
                        pred_S5=float(four[four.S == 5].joint_predictive_cover.mean()))

    # mission-level sd: what changes when eps is 0 or 0.6
    ev = T[(T.S == 20) & (T.M == 4) & (T.P == 4) & (T.m == 1) & (~T.unbal) & (T.tau == 0.42)]
    out["eps"] = {str(e): dict(pow_tau=float(g.reject_tau.iloc[0]), pow_omega=float(g.reject_omega.iloc[0]),
                              pred_cover=float(g.joint_predictive_cover.iloc[0]))
                  for e, g in ev.groupby("eps")}
    with open(os.path.join(OUT, "campaign_extra.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
