"""
Called by make_numbers_v7.py. Every number of Section IV that comes from a
bootstrap, recomputed from the campaigns rerun with the convergence rule of
hglmm.Fit.fit (SPARES_RULE=1), plus the diagnosis of the extreme plans across
two machines (analysis/campaign_diag.py, run on the HPC and on a Windows PC).
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))


def main(put, w, pct, thousands, V2):
    from campaign_boot import cells as boot_cells

    # ------------------------------------------------ Table IV campaigns --
    src = "analysis/campaign_boot.py, SPARES_RULE=1 -> data/v2/campaign_boot_rule_*"
    CB = pd.read_csv(os.path.join(V2, "campaign_boot_rule_summary.csv"))
    cells_m = pd.read_csv(os.path.join(V2, "campaign_summary.csv")).set_index("cell").m
    CB["m"] = CB.cell.map(cells_m)
    put("cbCells", str(len(CB)), "SIM", src)
    put("cbReps", str(int(CB.reps.min())), "SIM", src)
    d = CB.joint_boot_minus_pred
    put("cbGainLo", pct(d.min()), "SIM", src)
    put("cbGainHi", pct(d.max()), "SIM", src)
    null = CB[(CB.m == 3) & (CB.S == 20) & (CB.tau == 0) & (CB.omega == 0)].iloc[0]
    put("cbNullPred", pct(null.joint_predictive_cover), "SIM", src)
    put("cbNullBoot", pct(null.joint_boot_cover), "SIM", src)

    cl = pd.DataFrame(boot_cells())[["cell", "S", "M", "m", "tau", "omega"]]
    R = pd.read_csv(os.path.join(V2, "campaign_boot_rule_reps.csv.gz")).merge(cl, on="cell")
    R["in_tau"] = (R.tau_b_q10 <= R.tau) & (R.tau <= R.tau_b_q90)
    R["in_om"] = (R.omega_b_q10 <= R.omega) & (R.omega <= R.omega_b_q90)
    R["f"] = R.tau ** 2 / (R.tau ** 2 + R.omega ** 2).where(R.tau ** 2 + R.omega ** 2 > 0)
    R["in_f"] = (R.share_b_q10 <= R.f) & (R.f <= R.share_b_q90)
    t = R[R.tau > 0].groupby("cell").in_tau.mean()
    o4 = R[(R.omega > 0) & (R.M == 4)].groupby("cell").in_om.mean()
    o1 = R[(R.omega > 0) & (R.M == 1)].groupby("cell").in_om.mean()
    fs = R[(R.tau > 0) & (R.omega > 0)].groupby("cell").in_f.mean()
    put("cvTauLo", pct(t.min(), 0), "SIM", src, "80 pct bootstrap percentile interval, per cell")
    put("cvTauHi", pct(t.max(), 0), "SIM", src)
    put("cvOmLo", pct(o4.min(), 0), "SIM", src)
    put("cvOmHi", pct(o4.max(), 0), "SIM", src)
    put("cvOmOneHi", pct(o1.max(), 0), "SIM", src)
    put("cvShareLo", pct(fs.min(), 0), "SIM", src)
    put("cvShareHi", pct(fs.max(), 0), "SIM", src)

    # ----------------------------------------------- Table III campaigns --
    src = "analysis/campaign_reg.py, SPARES_RULE=1 -> data/v2/campaign_reg_rule_reps.csv.gz"
    G = pd.read_csv(os.path.join(V2, "campaign_reg_rule_reps.csv.gz"))
    plans = [("pred", "fixed"), ("boot", "bootstrap"), ("bootcap", "boot., cap 15"),
             ("penpred1", "pen., $s{=}1$, fixed"), ("pen0.5", "pen.\\ boot., $s{=}0.5$"),
             ("pen1", "pen.\\ boot., $s{=}1$"), ("pen2", "pen.\\ boot., $s{=}2$")]
    designs = [(5, 1), (10, 1)]
    rows = []
    for key, label in plans:
        cells_ = []
        for S_, M_ in designs:
            g = G[(G.S == S_) & (G.M == M_)]
            cov, un = g[f"{key}_cover"], g[f"{key}_units"]
            big = int((un > 5 * g.oracle_units).sum())
            cells_ += [pct(cov.mean()), pct(cov.quantile(0.1)), f"{un.median():.0f}",
                       f"{un.quantile(0.9):.0f}", str(big)]
        rows.append(f"{label} & " + " & ".join(cells_) + " \\\\")
    rows.append("\\midrule")
    cells_ = []
    for S_, M_ in designs:
        g = G[(G.S == S_) & (G.M == M_)]
        cells_ += [pct(g.oracle_cover.mean()), pct(g.oracle_cover.quantile(0.1)),
                   f"{g.oracle_units.median():.0f}", f"{g.oracle_units.quantile(0.9):.0f}", "--"]
    rows.append("known param. & " + " & ".join(cells_) + " \\\\")
    w("tab_reg7.tex", "\n".join(rows) + "\n")

    g5 = G[(G.S == 5) & (G.M == 1)]
    put("rgN", str(len(g5)), "SIM", src, "campaigns with S=5, M=1")
    put("rgReps", str(int(G.groupby("cell").size().min())), "SIM", src)
    big = g5.boot_units > 5 * g5.oracle_units
    put("rgBigN", str(int(big.sum())), "SIM", src, "bootstrap plan > 5x oracle")
    put("rgBigMax", thousands(g5.boot_units.max()), "SIM", src)
    for key, tag in (("pred", "Pred"), ("pen0.5", "PenHalf"), ("pen1", "PenOne"), ("pen2", "PenTwo"),
                     ("bootcap", "Cap"), ("penpred1", "PenPredOne")):
        put(f"rg{tag}Big", str(int((g5[f"{key}_units"] > 5 * g5.oracle_units).sum())), "SIM", src)
        put(f"rg{tag}Cov", pct(g5[f"{key}_cover"].mean()), "SIM", src, "S=5, M=1 mean cover")
    put("rgCapInfeas", str(int(g5.bootcap_infeasible.sum())), "SIM", src)
    put("rgTraceWide", f"{g5[big].n_wide.median():.0f}", "SIM", src, "refits with total sd > 1.5, median")
    put("rgTraceEither", pct(g5[big].over_from_either.median(), 0), "SIM", src)
    put("rgTraceHatSig", f"{np.hypot(g5[big].tau_hat, g5[big].omega_hat).median():.2f}", "SIM", src)
    put("rgTraceHatSigOther", f"{np.hypot(g5[~big].tau_hat, g5[~big].omega_hat).median():.2f}", "SIM", src)
    g54 = G[(G.S == 5) & (G.M == 4)]
    for key, tag in (("pred", "Pred"), ("boot", "Boot"), ("pen1", "PenOne")):
        put(f"rgFour{tag}Cov", pct(g54[f"{key}_cover"].mean()), "SIM", src, "S=5, M=4")
        put(f"rgFour{tag}Units", f"{g54[f'{key}_units'].median():.0f}", "SIM", src)
    put("rgFourOracleUnits", f"{g54.oracle_units.median():.0f}", "SIM", src)

    # convergence rule over all fits of the three reruns
    src = "data/v2/campaign_{boot,reg}_rule_reps.csv.gz, campaign_bconv_rule_reps.csv"
    Rc = pd.read_csv(os.path.join(V2, "campaign_bconv_rule_reps.csv"))
    pen_fail = sum(int(G[c].sum()) for c in G.columns if c.endswith("_refits_failed") and c.startswith("pen"))
    n_ref = 20 * (len(R) + len(G)) + 80 * len(Rc) + 3 * 20 * len(G)
    failed = int(R.refits_failed.sum() + G.refits_failed.sum() + Rc.refits_failed.sum()) + pen_fail
    restarted = int(R.refits_restarted.sum() + G.refits_restarted.sum())
    n_fit = len(R) + len(G) + len(Rc)
    put("rlRefits", thousands(n_ref), "SIM", src, "bootstrap refits under the rule")
    put("rlFailed", str(failed), "SIM", src, "refits declared failed and dropped")
    put("rlFailedPct", pct(failed / n_ref, 2), "SIM", src)
    put("rlRestarted", str(restarted), "SIM", src, "refits of Tables III and IV restarted")
    put("rlRestartedPct", pct(restarted / (20 * (len(R) + len(G))), 1), "SIM", src)
    put("rlFitsBad", str(int((R.fit_accepted == 0).sum() + (G.fit_accepted == 0).sum()
                             + (Rc.fit_accepted == 0).sum())), "SIM", src, "point estimates not accepted")
    put("rlFits", thousands(n_fit), "SIM", src)
    put("rlFitsBadGrad", f"{R[R.fit_accepted == 0].fit_grad.max():.2f}", "SIM", src,
        "largest projected gradient of a point estimate with no accepted run")
    gain_all = R.groupby("cell").joint_cover_boot.mean() - R.groupby("cell").joint_cover_predictive.mean()
    ok = R[R.fit_accepted == 1]
    gain_ok = ok.groupby("cell").joint_cover_boot.mean() - ok.groupby("cell").joint_cover_predictive.mean()
    put("rlFitsBadEffect", f"{100 * (gain_all - gain_ok).abs().max():.2f}", "SIM", src,
        "largest change in a cell's bootstrap cover gain (points) when they are dropped")
    assert (G.fit_accepted == 1).all() and (Rc.fit_accepted == 1).all()
    # near-equivalent maxima: the same Table III campaigns fitted from one start
    # (first run, campaign_reg_reps.csv.gz) and from three
    key = ["S", "M", "tau", "omega", "rep"]
    O = pd.read_csv(os.path.join(V2, "campaign_reg_reps.csv.gz"))
    J = O.merge(G, on=key, suffixes=("_one", "_three"))
    assert len(J) == len(G) and (J.oracle_units_one == J.oracle_units_three).all()
    alt = J[J.fit_crit_gain > 0.05]
    assert set(zip(G.S, G.M)) == {(5, 1), (10, 1), (5, 4)}
    put("rlAltOf", str(len(G)), "SIM", src, "small campaigns: Table III designs and S = 5, M = 4")
    put("rlAltN", str(len(alt)), "SIM", src, "of them, where another start gained > 0.05")
    put("rlAltOneMed", f"{alt.pred_units_one.median():.0f}", "SIM", src, "predictive units, one start")
    put("rlAltThreeMed", f"{alt.pred_units_three.median():.0f}", "SIM", src, "predictive units, three starts")
    won = pd.concat([R.fit_crit_gain, G.fit_crit_gain, Rc.fit_crit_gain])
    put("rlFitsWonPct", pct((won > 1e-3).mean(), 1), "SIM", src,
        "point estimates where a start other than the default reached a higher criterion")
    g5w = G[(G.S == 5) & (G.M == 1)].fit_crit_gain
    put("rlFitsWonFivePct", pct((g5w > 1e-3).mean(), 0), "SIM", src, "same, S = 5, M = 1")

    # design study: one start against three (analysis/campaign_starts.py)
    S = json.load(open(os.path.join(V2, "campaign_starts.json"), encoding="utf-8"))
    src = "analysis/campaign_starts.py -> data/v2/campaign_starts.json"
    put("csN", thousands(S["campaigns"]), "SIM", src)
    put("csReps", str(S["campaigns"] // S["cells"]), "SIM", src)
    put("csHigherPct", pct(S["higher"] / S["campaigns"], 1), "SIM", src)
    put("csHigherOnePct", pct(S["higher_M1"] / S["n_M1"], 0), "SIM", src)
    put("csOmChanged", str(S["omega_decision_changed"]), "SIM", src)
    put("csTauChanged", str(S["tau_decision_changed"]), "SIM", src)

    # ----------------------------------------------- B = 20 against 80 --
    src = "analysis/campaign_bconv.py, SPARES_RULE=1 -> data/v2/campaign_bconv_rule*"
    B = json.load(open(os.path.join(V2, "campaign_bconv_rule.json"), encoding="utf-8"))
    put("bcCells", {3: "three", 4: "four", 5: "five"}.get(len(B), str(len(B))), "SIM", src)
    put("bcDiffMax", f"{100 * max(abs(v['d20_80_mean']) for v in B.values()):.2f}", "SIM", src)
    put("bcMcse", f"{100 * max(v['d20_80_mcse'] for v in B.values()):.2f}", "SIM", src)
    du = (Rc.units_b80_again - Rc.units_b80).abs()
    put("bcRedrawTen", str(int((du > 10).sum())), "SIM", src, "plan changed by > 10 units")
    put("bcN", str(len(Rc)), "SIM", src)

    # ------------------------------------- the two machines, step by step --
    import glob
    src = "analysis/campaign_diag.py -> data/v2/campaign_diag_{hpc,local}_*.json"
    H, L = {}, {}
    for tag, D in (("hpc", H), ("local", L)):
        meta = None
        for path in sorted(glob.glob(os.path.join(V2, f"campaign_diag_{tag}_*.json"))):
            J = json.load(open(path, encoding="utf-8"))
            meta = J
            for c in J["cases"]:
                D[(c["cell"], c["rep"])] = c
        D["_meta"] = (meta["machine"], meta["numpy"], meta["scipy"])
    assert set(H) == set(L), "diagnosis missing on one machine"
    keys = [k for k in H if k != "_meta"]
    same_data = sum(H[k]["data_hash"] == L[k]["data_hash"] for k in keys)
    dsd = [abs(math.hypot(*np.exp(H[k]["fit"]["theta"][:2])) - math.hypot(*np.exp(L[k]["fit"]["theta"][:2])))
           for k in keys]
    dcrit = [abs(H[k]["fit"]["crit"] - L[k]["fit"]["crit"]) for k in keys]
    same_boot = [sum(a == b for a, b in zip(H[k]["refit_hashes"], L[k]["refit_hashes"])) for k in keys]
    flat = [max(abs(v["crit"] - L[k]["fit"]["crit"]) for v in L[k]["fit_starts"].values()) for k in keys]
    put("dgCases", str(len(keys)), "SIM", src)
    put("dgSameData", str(same_data), "SIM", src, "cases with identical simulated counts")
    put("dgSdDiffMax", f"{max(dsd):.3f}", "SIM", src, "largest difference in fitted total sd between machines")
    put("dgCritDiffMax", f"{max(dcrit):.1e}", "SIM", src, "largest difference in the criterion at the two estimates")
    put("dgBootSameMin", str(min(same_boot)), "SIM", src, "fewest identical bootstrap data sets of 20")
    put("dgBootSameMax", str(max(same_boot)), "SIM", src)
    put("dgBootSameMed", f"{np.median(same_boot):.0f}", "SIM", src)
    put("dgStartFlat", f"{max(flat):.0e}", "SIM", src, "largest criterion difference over restarts, point estimate")
    put("dgMachines", f"{H['_meta'][0].split('-')[0]} and {L['_meta'][0].split('-')[0]}", "SIM", src)
    # bootstrap at a fixed estimate, ten fresh streams
    ratio, lo, hi = [], [], []
    for k in keys:
        u = [r["units"] for r in H[k]["fresh_streams"]]
        ratio.append(max(u) / min(u))
        lo.append(min(u))
        hi.append(max(u))
    k19 = (1, 19)
    u19 = [r["units"] for r in H[k19]["fresh_streams"]]
    put("dgFreshK", str(len(H[keys[0]]["fresh_streams"])), "SIM", src)
    put("dgRatioMed", f"{np.median(ratio):.0f}", "SIM", src, "median max/min of the bootstrap plan over fresh streams")
    put("dgRatioMax", thousands(round(max(ratio))), "SIM", src)
    put("dgCaseLo", thousands(min(u19)), "SIM", src, "campaign (1, 19), HPC estimate")
    put("dgCaseHi", thousands(max(u19)), "SIM", src)
    put("dgCaseHpc", thousands(H[k19]["boot_units"]), "SIM", src, "Table IV stream, HPC")
    put("dgCaseLocal", thousands(L[k19]["boot_units"]), "SIM", src, "Table IV stream, PC")
    # which refits set the vector: share of draws above it from the widest refit
    top = []
    for k in keys:
        ex = np.array(H[k]["exceed_by_refit"])
        sd = np.array([r["sd_total"] for r in H[k]["refits"]])
        top.append(float(sd[np.argmax(ex)]))
    put("dgTopSdMed", f"{np.median(top):.1f}", "SIM", src, "total sd of the refit with most draws above the vector")
    # two maxima: another start reaching a higher criterion than the default start
    gains = [max(v["crit"] for v in H[k]["fit_starts"].values()) - H[k]["fit"]["crit"] for k in keys]
    two = [g for g in gains if g > 1e-3]
    put("dgTwoMax", str(len(two)), "SIM", src, "campaigns where another start found a higher maximum")
    put("dgTwoMaxLo", f"{min(two):.2f}", "SIM", src)
    put("dgTwoMaxHi", f"{max(two):.2f}", "SIM", src)
    abn = sum(sum(not r["success"] for r in H[k]["refits"]) for k in keys)
    put("dgAbnormal", str(abn), "SIM", src, "refits whose optimiser exit was not success, HPC")
    put("dgRefits", str(sum(len(H[k]["refits"]) for k in keys)), "SIM", src)
