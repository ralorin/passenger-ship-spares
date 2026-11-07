"""
Numbers and tables for the revision (exact stock vectors, bootstrap of the
variance components, new fleet comparators, street-lighting validation),
written to numbers_v3.tex and numbers_v3_manifest.csv with the provenance
tags of make_numbers.py.

    python make_numbers_v3.py
"""
import csv
import json
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, "..", "data", "v2")
TAB = os.path.join(HERE, "tables")
MACROS, MANIFEST = [], []


def put(name, value, tag, source, note=""):
    MACROS.append((name, value))
    MANIFEST.append(dict(macro=name, value=value, provenance=tag, source=source, note=note))


def pct(x, d=1):
    return f"{100 * float(x):.{d}f}"


def w(name, text):
    with open(os.path.join(TAB, name), "w", encoding="utf-8") as f:
        f.write(text)


def thou(n):
    return f"{int(n):,}".replace(",", "\\,")


def sci(x):
    m, e = f"{x:.0e}".split("e")
    return f"{m}\\times 10^{{{int(e)}}}"


def load(name):
    with open(os.path.join(V2, name), encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------- exact stock vectors --
src = "analysis/stock_exact.py -> data/v2/stock_exact*.csv"
T = pd.read_csv(os.path.join(V2, "stock_exact_part1.csv"))
S = load("stock_exact.json")
uneq = (T.means == "skewed") | (T.costs == "skewed")
put("seCases", str(len(T)), "ANALYTIC", src)
put("seProved", str(S["proved"]), "ANALYTIC", src, "cases where branch and bound finished")
put("seUnproved", str(S["cases"] - S["proved"]), "ANALYTIC", src)
un = T[~T.proved]
lb = np.ceil(un.lower_bound - 1e-9)          # costs are integers, so is the optimum
gap = (un.cost_exact - lb) / lb
put("seGapUnproved", pct(gap.max()), "ANALYTIC", src, "best found vs ceiling of Hoelder-Lagrange bound")
put("seGapUnprovedMed", pct(gap.median()), "ANALYTIC", src)
put("seGreedyWorse", str(int(((T.cost_greedy - T.cost_exact) > 1e-9).sum())), "ANALYTIC", src)
put("seGreedyOpt", str(S["greedy_optimal"]), "ANALYTIC", src, "cases where marginal analysis is optimal")
put("seGreedyGapMax", pct(S["greedy_gap_max"]), "ANALYTIC", src)
put("seSaveComMedUneq", pct(S["save_vs_common_median_uneq"], 0), "ANALYTIC", src)
put("seSaveComMax", pct(S["save_vs_common_max"], 0), "ANALYTIC", src)
put("seSaveEqrMedUneq", pct(S["save_vs_equal_risk_median_uneq"]), "ANALYTIC", src)
put("seSaveEqrMax", pct(S["save_vs_equal_risk_max"]), "ANALYTIC", src)
put("seIndepExtraMed", pct(S["indep_extra_vs_exact_median"]), "ANALYTIC", src, "f > 0")
put("seIndepExtraMax", pct(S["indep_extra_vs_exact_max"]), "ANALYTIC", src)
put("seMonoCases", str(S["mono_cases"]), "ANALYTIC", src)
mono = S["mono_min_step"]
put("seMonoMin", "0" if mono >= 0 else sci(-mono), "ANALYTIC", src,
    f"largest decrease of J along f: {mono:.2e}")
put("seSigJointMin", pct(S["sigma_joint_cover_min"]), "ANALYTIC", src)
put("seSigAggMin", pct(S["sigma_agg_cover_min"]), "ANALYTIC", src)
rows = []
labels = [("marginal", "Per-reference quantile"), ("common", "Common stock"),
          ("equal_risk", "Equal risk"), ("indep", "Independence ($f=0$)"),
          ("greedy", "Marginal analysis")]
for key, label in labels:
    if key == "marginal":
        continue
    ex = T[f"cost_{key}"] / T.cost_exact - 1
    rows.append(f"{label} & {pct(ex[uneq & (T.f > 0)].median())} & {pct(ex[~uneq & (T.f > 0)].median())} & "
                f"{pct(ex.max())} & {pct(T[f'cover_{key}'].min())} \\\\")
w("tab_opt.tex", "\n".join(rows) + "\n")

# ------------------------------------------------------------- campaign extras --
src = "analysis/campaign_extra.py -> data/v2/campaign_extra.json"
C = load("campaign_extra.json")
put("ceTypeMed", pct(C["typeI"]["median"]), "SIM", src)
put("ceTypeMax", pct(C["typeI"]["max"]), "SIM", src)
put("ceTypeMaxLo", pct(C["typeI"]["max_ci"][0]), "SIM", src, "Clopper-Pearson 95 pct")
put("ceTypeMaxHi", pct(C["typeI"]["max_ci"][1]), "SIM", src)
put("ceTypeCells", str(C["typeI"]["cells"]), "SIM", src)
put("ceTypeBelow", str(C["typeI"]["cells_upper_below_nominal"]), "SIM", src)
put("ceTypeAbove", str(C["typeI"]["cells_lower_above_nominal"]), "SIM", src)
put("ceOmOneMax", pct(C["typeI"]["omega_M1_max"]), "SIM", src)
put("ceCiLo", pct(C["profile"]["stored_lo"], 0), "SIM", src)
put("ceCiHi", pct(C["profile"]["stored_hi"], 0), "SIM", src)
put("ceCiZeroOld", pct(C["profile"]["zero_stored"]), "SIM", src)
put("ceCiZeroNew", pct(C["profile"]["zero_corrected"]), "SIM", src)
put("ceCiReps", str(C["profile"]["reps"]), "SIM", src)
put("ceCiCells", str(C["profile"]["cells"]), "SIM", src)
put("ceOnePowOm", pct(C["M1"]["power_omega_max"], 0), "SIM", src)
put("ceOneBound", pct(C["M1"]["omega_at_bound_median"], 0), "SIM", src)
put("ceOnePowTauLo", pct(C["M1"]["power_tau_by_S"]["5"], 0), "SIM", src)
put("ceOnePowTauHi", pct(C["M1"]["power_tau_by_S"]["80"], 0), "SIM", src)
put("ceOnePredMin", pct(C["M1"]["joint_pred_cover_min"]), "SIM", src)
put("cePredMcse", pct(C["cover"]["pred_mcse_max"], 2), "SIM", src)
put("cePredMinTen", pct(C["cover"]["pred_min_S10"]), "SIM", src)
for e, v in C["eps"].items():
    tag = {"0.0": "Zero", "0.3": "Base", "0.6": "Big"}[e]
    put(f"ceEps{tag}PowOm", pct(v["pow_omega"], 0), "SIM", src)
    put(f"ceEps{tag}Cover", pct(v["pred_cover"]), "SIM", src)

# ------------------------------------------------------------- federated check --
if os.path.exists(os.path.join(V2, "fed_predictive.json")):
    src = "analysis/fed_predictive.py -> data/v2/fed_predictive.json"
    F = load("fed_predictive.json")
    put("fpN", str(F["n"]), "SIM", src)
    put("fpSame", str(F["same_plan"]), "SIM", src)
    put("fpUnitsMax", str(F["units_diff_max"]), "SIM", src)
    put("fpCoverMax", pct(F["cover_diff_max"], 2), "SIM", src)

# ------------------------------------------------- bootstrap of variance --
src = "analysis/campaign_boot.py -> data/v2/campaign_boot_summary.csv"
CB = pd.read_csv(os.path.join(V2, "campaign_boot_summary.csv"))
CR = pd.read_csv(os.path.join(V2, "campaign_boot_reps.csv.gz"))
cells_m = pd.read_csv(os.path.join(V2, "campaign_summary.csv")).set_index("cell").m
CB["m"] = CB.cell.map(cells_m)
put("cbCells", str(len(CB)), "SIM", src)
put("cbReps", str(int(CB.reps.min())), "SIM", src)
put("cbB", "20", "SIM", src, "bootstrap refits per campaign, campaign_boot.B")
rows = []
for (S, M, m), g in CB.groupby(["S", "M", "m"], sort=False):
    rows.append(f"{S} & {M} & {m:g} & {pct(g.joint_predictive_cover.mean())} & {pct(g.joint_boot_cover.mean())} & "
                f"{pct(g.joint_predictive_p10.mean())} & {pct(g.joint_boot_p10.mean())} & "
                f"{g.joint_predictive_units_med.mean():.0f} & {g.joint_boot_units_med.mean():.0f} \\\\")
w("tab_boot.tex", "\n".join(rows) + "\n")
d = CB.joint_boot_minus_pred
put("cbGainLo", pct(d.min()), "SIM", src)
put("cbGainHi", pct(d.max()), "SIM", src)
big = CB[(CB.S >= 20) & (CB.M == 4) & (CB.m == 1)]
put("cbGainBig", pct(big.joint_boot_minus_pred.max()), "SIM", src, "S = 20, M = 4")
small = CB[(CB.S == 5) & (CB.M == 1)]
put("cbUnitsSmallPred", f"{small.joint_predictive_units_med.mean():.0f}", "SIM", src)
put("cbUnitsSmallBoot", f"{small.joint_boot_units_med.mean():.0f}", "SIM", src)
put("cbUnitsSmallOracle", f"{small.joint_oracle_units_med.mean():.0f}", "SIM", src)
null = CB[(CB.m == 3) & (CB.S == 20) & (CB.tau == 0) & (CB.omega == 0)].iloc[0]
put("cbNullPred", pct(null.joint_predictive_cover), "SIM", src)
put("cbNullBoot", pct(null.joint_boot_cover), "SIM", src)
put("cbNullUnits", f"{null.joint_predictive_units_med:.0f}", "SIM", src)
put("cbNullOracle", f"{null.joint_oracle_units_med:.0f}", "SIM", src)
put("cbShareWidth", f"{CB[CB.M == 1].share_width.median():.2f}", "SIM", src, "median 10-90 pct width of f, M = 1")
put("cbShareWidthBig", f"{CB[(CB.S == 20) & (CB.M == 4)].share_width.median():.2f}", "SIM", src)
put("cbMcseMax", pct(CB.joint_boot_minus_pred_mcse.max(), 2), "SIM", src)

# ------------------------------------------------------------- fleet (v4) --
src = "analysis/fleet.py, fleet_extra.py -> data/v2/fleet_extra.json"
FX = load("fleet_extra.json")
R4 = pd.read_csv(os.path.join(V2, "fleet_v4_reps.csv.gz"))
nf = FX["fleets_per_scenario"]
R4 = R4[R4.rep < nf]
put("flvFleets", str(nf), "SIM", src)
put("flvMissions", thou(round(R4[(R4.kind.isin(["known", "new"])) & (R4.method == "hier_pred")]
                              .groupby("scenario").n.sum().mean() / nf, -1)), "SIM", src, "per fleet")
sh = R4[(R4.method == "hier_pred") & (R4.scenario == "shocks_frequent")]
k_all = sh[sh.kind.isin(["known", "new"])].n.sum()
k_sh = sh[sh.kind.isin(["known_shock", "new_shock"])].n.sum()
put("flvShockShare", pct(k_sh / k_all, 1), "SIM", src, "missions with an event at 2 per ship-year")
order = [("clean", "Clean"), ("baseline", "Baseline"), ("heavy_tail", "Heavy tail"),
         ("misclass", "Misrecorded"), ("sparse", "Sparse"), ("datasheet_ok", "Datasheet right"),
         ("shocks", "Events 0.5/yr"), ("shocks_frequent", "Events 2/yr"),
         ("shocks_severe", "Events 2/yr, large")]
cols = ["mtbf_joint", "mtbf_scaled", "poisson_glm", "nb_pooled", "eb_gamma", "hier_marg", "hier_pred", "hier_shock"]
rows = []
for key, label in order:
    k = FX[key]["known"]
    cells = [f"{pct(k[m]['cover'])}/{k[m]['units']:.1f}" for m in cols]
    rows.append(label + " & " + " & ".join(cells) + " \\\\")
w("tab_fleet4.tex", "\n".join(rows) + "\n")

fitted = ["poisson_glm", "nb_pooled", "eb_gamma", "hier_plugin", "hier_pred", "hier_shock", "hier_agg"]
stress = ["shocks_frequent", "shocks_severe"]
calm = [s for s, _ in order if s not in stress]
cv = lambda scn, kind, m: FX[scn][kind][m]["cover"]
put("flvMtbfLo", pct(min(cv(s, "known", "mtbf_joint") for s in calm if s != "datasheet_ok")), "SIM", src)
put("flvMtbfHi", pct(max(cv(s, kd, "mtbf_joint") for s in calm if s != "datasheet_ok" for kd in ("known", "new"))),
    "SIM", src)
put("flvScaledLo", pct(min(cv(s, kd, "mtbf_scaled") for s in calm for kd in ("known", "new"))), "SIM", src)
put("flvScaledHi", pct(max(cv(s, kd, "mtbf_scaled") for s in calm for kd in ("known", "new"))), "SIM", src)
put("flvFitLo", pct(min(cv(s, kd, m) for s in calm for kd in ("known", "new")
                        for m in ("nb_pooled", "hier_pred", "eb_gamma"))), "SIM", src)
put("flvFitHi", pct(max(cv(s, kd, m) for s in calm for kd in ("known", "new")
                        for m in ("nb_pooled", "hier_pred", "eb_gamma"))), "SIM", src)
put("flvPredLo", pct(min(cv(s, kd, "hier_pred") for s in calm for kd in ("known", "new"))), "SIM", src)
put("flvPredHi", pct(max(cv(s, kd, "hier_pred") for s in calm for kd in ("known", "new"))), "SIM", src)
put("flvStressLo", pct(min(cv(s, kd, m) for s in stress for kd in ("known", "new") for m in fitted)), "SIM", src)
put("flvStressHi", pct(max(cv(s, kd, m) for s in stress for kd in ("known", "new") for m in fitted)), "SIM", src)
put("flvMargLo", pct(min(cv(s, kd, "hier_marg") for s in calm for kd in ("known", "new"))), "SIM", src)
put("flvMargHi", pct(max(cv(s, kd, "hier_marg") for s in calm for kd in ("known", "new"))), "SIM", src)
ci = FX["baseline"]["known"]["hier_pred"]["cover_ci"]
put("flvMtbfRight", pct(cv("datasheet_ok", "known", "mtbf_joint")), "SIM", src)
put("flvCiLo", pct(ci[0]), "SIM", src)
put("flvCiHi", pct(ci[1]), "SIM", src)
b = FX["baseline"]


def trip(x, d=2, scale=1.0):
    return [f"{scale * v:+.{d}f}" for v in x]


p = b["pair_known_nb_pooled"]
a, lo, hi = trip(p["cover"], 2, 100)
put("flvNbCov", f"${a}$", "SIM", src); put("flvNbCovLo", f"${lo}$", "SIM", src); put("flvNbCovHi", f"${hi}$", "SIM", src)
a, lo, hi = trip(p["units"], 2)
put("flvNbUn", f"${a}$", "SIM", src); put("flvNbUnLo", f"${lo}$", "SIM", src); put("flvNbUnHi", f"${hi}$", "SIM", src)
a, lo, hi = trip(p["log_score"], 3)
put("flvNbLs", f"${a}$", "SIM", src); put("flvNbLsLo", f"${lo}$", "SIM", src); put("flvNbLsHi", f"${hi}$", "SIM", src)
put("flvNbFewer", pct(p["fewer_units"], 0), "SIM", src)
put("flvNbFewerLo", pct(min(FX[s]["pair_known_nb_pooled"]["fewer_units"] for s, _ in order), 0), "SIM", src)
put("flvNbFewerHi", pct(max(FX[s]["pair_known_nb_pooled"]["fewer_units"] for s, _ in order), 0), "SIM", src)
pn = b["pair_new_nb_pooled"]
put("flvNbNewUn", f"${pn['units'][0]:+.2f}$", "SIM", src)
put("flvNbNewCov", f"${100 * pn['cover'][0]:+.2f}$", "SIM", src)
ps = b["pair_known_mtbf_scaled"]
put("flvScCov", f"${100 * ps['cover'][0]:+.2f}$", "SIM", src)
put("flvScUn", f"${ps['units'][0]:+.2f}$", "SIM", src)
put("flvUnitsPred", f"{b['known']['hier_pred']['units']:.1f}", "SIM", src)
put("flvUnitsNb", f"{b['known']['nb_pooled']['units']:.1f}", "SIM", src)
put("flvUnitsMtbf", f"{b['known']['mtbf_joint']['units']:.1f}", "SIM", src)
put("flvUnitsScaled", f"{b['known']['mtbf_scaled']['units']:.1f}", "SIM", src)
for r in ("1", "100"):
    t = "One" if r == "1" else "Hundred"
    for m, tag in (("hier_pred", "Pred"), ("nb_pooled", "Nb"), ("mtbf_scaled", "Sc"), ("mtbf_joint", "Mtbf")):
        put(f"flvCost{tag}{t}", f"{b['known'][m]['total_cost'][r]:.1f}", "SIM", src)
put("flvAggUnits", f"{b['known']['hier_agg']['units']:.1f}", "SIM", src)
put("flvAggCover", pct(b["known"]["hier_agg"]["cover"]), "SIM", src)
put("flvPredCover", pct(b["known"]["hier_pred"]["cover"]), "SIM", src)
for scn, tag in (("baseline", "Base"), ("shocks_frequent", "Freq"), ("shocks_severe", "Sev")):
    o = FX[scn]["known_shock"]
    put(f"flvSh{tag}N", thou(o["hier_pred"]["missions"]), "SIM", src)
    put(f"flvSh{tag}Pred", pct(o["hier_pred"]["cover"], 0), "SIM", src)
    put(f"flvSh{tag}Mis", pct(o["hier_shock"]["cover"], 0), "SIM", src)
    put(f"flvSh{tag}Short", f"{o['hier_pred']['short']:.1f}", "SIM", src, "mean units short per event mission")
    put(f"flvSh{tag}ShortCost", f"{o['hier_pred']['short_cost']:.1f}", "SIM", src)
allsh = [FX[s][kd][m]["cover"] for s, _ in order for kd in ("known_shock", "new_shock")
         if kd in FX[s] and s != "sparse" for m in fitted]
put("flvShLo", pct(min(allsh), 0), "SIM", src)
put("flvShHi", pct(max(allsh), 0), "SIM", src)
put("flvCapFleets", str(FX["caps"]["sigma_hier"]["fleets"]), "SIM", src)
put("flvCapHits", str(FX["caps"]["sigma_hier"]["fleets_at_cap"] + FX["caps"]["sigma_shock"]["fleets_at_cap"]
                      + FX["caps"]["sigma_agg"]["fleets_at_cap"]), "SIM", src)

# ------------------------------------------------------ street lighting --
src = "analysis/ni_validation.py -> data/v2/ni_validation_*.json"
FL = load("ni_validation_flow.json")
EV = load("ni_validation_evidence.json")
DD = load("ni_validation_dedup.json")
IV = load("ni_validation_inventory.json")
BT = load("ni_validation_boot.json")
flow = dict(FL["flow"])
put("niRowsAll", thou(flow["after removing repeated DATA_UID across files"]), "PUBLICDATA", src)
put("niRowsType", thou(flow["fault type Light Out or All lights out in street"]), "PUBLICDATA", src)
put("niRowsAsset", thou(flow["operational area recorded"]), "PUBLICDATA", src, "lighting point, area recorded")
put("niRowsComment", thou(flow["closure comment does not point away from the lantern"]), "PUBLICDATA", src)
ex = FL["exclusions_by_comment"]
put("niExCable", thou(ex["cable"]), "PUBLICDATA", src)
put("niExSupply", thou(ex["NIE (supply)"]), "PUBLICDATA", src)
put("niExNoFault", thou(ex["not out / no fault / all ok / operational"] + ex["working on attendance"]),
    "PUBLICDATA", src)
put("niRowsMerged", thou(flow["after merging repeats on a street within 7 days"]), "PUBLICDATA", src)
put("niMergeShare", pct(1 - flow["after merging repeats on a street within 7 days"]
                        / flow["closure comment does not point away from the lantern"], 0), "PUBLICDATA", src)
put("niRowsPure", thou(flow["Light Out on a street with one LED family (any month)"]), "PUBLICDATA", src)
put("niSkipped", thou(sum(load("ni_prep_2025.json")["lines_skipped"].values())), "PUBLICDATA", src)
fw = FL["fit_window_reports"]
put("niCommentShare", pct(1 - FL["fit_window_empty_comment"] / fw, 0), "PUBLICDATA", src)
kw = FL["fit_window_keyword_counts"]
put("niPartCount", str(kw["driver"] + kw["lantern / luminaire"] + kw["lamp / LED unit"] + kw["photocell / PECU"]),
    "PUBLICDATA", src, "reports naming a part")
put("niDupCount", str(kw["duplicate / see other card"]), "PUBLICDATA", src)
put("niOpen", str(fw - FL["fit_window_status"]["Completed"] - FL["fit_window_status"]["Record Closed"]),
    "PUBLICDATA", src, "fit reports not closed")

for key, tag in (("no merging", "Zero"), ("window 14", "Fourteen"), ("window 7", "Seven"),
                 ("window 7, streets <= 10 points only", "Small")):
    v = DD[key]
    put(f"niDd{tag}Fail", thou(v["fit_failures"]), "PUBLICDATA", src)
    put(f"niDd{tag}Inter", f"{v['sigma_area_family']:.2f}", "PUBLICDATA", src)
    put(f"niDd{tag}Phi", f"{v['phi']:.1f}", "PUBLICDATA", src)
    put(f"niDd{tag}HierUnits", f"{v['within']['hier_pred|known']['units']:.0f}", "PUBLICDATA", src)
    put(f"niDd{tag}HierExc", str(192 - v["within"]["hier_pred|known"]["covered"]), "PUBLICDATA", src)
    put(f"niDd{tag}NbUnits", f"{v['within']['nb_pooled|known']['units']:.0f}", "PUBLICDATA", src)
    put(f"niDd{tag}NbExc", str(192 - v["within"]["nb_pooled|known"]["covered"]), "PUBLICDATA", src)
inter = [v["sigma_area_family"] for v in DD.values()]
put("niDdInterLo", f"{min(inter):.2f}", "PUBLICDATA", src)
put("niDdInterHi", f"{max(inter):.2f}", "PUBLICDATA", src)

sn = IV["snapshots"]
put("niStreets", thou(sn["pure_led_streets_2025"]), "PUBLICDATA", src)
put("niStreetsSame", pct(sn["identical"] / sn["pure_led_streets_2025"], 0), "PUBLICDATA", src)
put("niLedGrowth", pct(sn["led_growth_per_year"]), "PUBLICDATA", src)
st = IV["stable_subset"]
put("niStableArea", f"{st['sigma'][0]:.2f}", "PUBLICDATA", src)
put("niStableInter", f"{st['sigma'][1]:.2f}", "PUBLICDATA", src)
put("niStableHierUnits", f"{st['within']['hier_pred|known']['units']:.0f}", "PUBLICDATA", src)
put("niStableHierExc", str(192 - st["within"]["hier_pred|known"]["covered"]), "PUBLICDATA", src)
put("niStableNbExc", str(192 - st["within"]["nb_pooled|known"]["covered"]), "PUBLICDATA", src)
r2 = IV["ramp at twice the growth"]
put("niRampInter", f"{r2['sigma'][1]:.2f}", "PUBLICDATA", src)
put("niRampHierUnits", f"{r2['within']['hier_pred|known']['units']:.0f}", "PUBLICDATA", src)
ly = IV["fit Apr 2024-Mar 2025 only"]
put("niLastArea", f"{ly['sigma'][0]:.2f}", "PUBLICDATA", src)
put("niLastInter", f"{ly['sigma'][1]:.2f}", "PUBLICDATA", src)

cf = BT["components_full"]["boot"]
put("niBootInterLo", f"{cf['area_family']['p2_5']:.2f}", "PUBLICDATA", src)
put("niBootInterHi", f"{cf['area_family']['p97_5']:.2f}", "PUBLICDATA", src)
put("niBootAreaZero", pct(cf["area"]["near_zero"], 0), "PUBLICDATA", src)
wh = BT["within_hier"]
put("niWithinArea", f"{wh['hat'][0]:.2f}", "PUBLICDATA", src)
put("niShareHi", f"{wh['share_area']['p97_5']:.2f}", "PUBLICDATA", src)
put("niShareMed", f"{wh['share_area']['median']:.2f}", "PUBLICDATA", src)
put("niCapFits", str(EV["sd_cap"]["fits"]), "PUBLICDATA", src)
put("niCapMax", f"{EV['sd_cap']['max_sd']:.2f}", "PUBLICDATA", src)

W, Fw = EV["within"], EV["forward"]
Wb = BT["within_plans"]
cov = dict(W["cover"])
cov.update({k: v for k, v in Wb["cover"].items() if k.startswith("hier_boot")})
ls = dict(W["log_scores"])
ls.update({k: v for k, v in Wb["log_scores"].items() if k.startswith("hier_boot")})
rows = []
for key, label in (("poisson", "Poisson"), ("poisson_jnt", "Poisson, joint"), ("nb_pooled", "Pooled NB"),
                   ("eb_gamma", "Gamma EB"), ("hier_plugin", "Hier., plug-in"),
                   ("hier_pred", "Hier., predictive"), ("hier_boot", "Hier., $\\theta$-integr."),
                   ("hier_shock", "Hier., + month")):
    k, n = cov[f"{key}|known"], cov[f"{key}|new"]
    l = ls.get(f"{key}|known")
    lstr = f"${l['log_score']:.2f}$" if l else "--"
    fk = Fw["cover"].get(f"{key}|known")
    fstr = f"{fk['exceed']} & {fk['units']:.0f}" if fk else "-- & --"
    rows.append(f"{label} & {k['exceed']} & {k['units']:.0f} & {lstr} & {n['exceed']} & {n['units']:.0f} & {fstr} \\\\")
w("tab_ni3.tex", "\n".join(rows) + "\n")
put("niBootNewUnits", f"{Wb['cover']['hier_boot|new']['units']:.0f}", "PUBLICDATA", src)
put("niBootPredNewUnits", f"{Wb['cover']['hier_pred|new']['units']:.0f}", "PUBLICDATA", src, "same run as hier_boot")
put("niWithinN", str(cov["hier_pred|known"]["n"]), "PUBLICDATA", src)
put("niForwardN", str(Fw["cover"]["hier_pred|known"]["n"]), "PUBLICDATA", src)
put("niLowZero", pct(cov["hier_pred|known"]["one_sided_lower95"]), "PUBLICDATA", src, "one-sided 95 pct, 0/192")
put("niLowZeroFwd", pct(Fw["cover"]["hier_pred|known"]["one_sided_lower95"]), "PUBLICDATA", src)
put("niCpNbLo", pct(cov["nb_pooled|known"]["cp95"][0]), "PUBLICDATA", src)
put("niCpNbHi", pct(cov["nb_pooled|known"]["cp95"][1]), "PUBLICDATA", src)
put("niClNbLo", pct(cov["nb_pooled|known"]["area_cluster95"][0]), "PUBLICDATA", src)
put("niNbKnownExc", str(cov["nb_pooled|known"]["exceed"]), "PUBLICDATA", src)
put("niNbNewExc", str(cov["nb_pooled|new"]["exceed"]), "PUBLICDATA", src)
put("niNbAreasBelow", str(cov["nb_pooled|known"]["areas_below_99"]), "PUBLICDATA", src)
put("niPoisAreasBelow", str(cov["poisson|known"]["areas_below_99"]), "PUBLICDATA", src)
pr = W["paired"]["hier_pred-nb_pooled|known"]
put("niPairUnits", f"{-pr['d_units']:.0f}", "PUBLICDATA", src, "pooled NB minus hier_pred, known")
put("niPairUnitsLo", f"{-pr['d_units95'][1]:.0f}", "PUBLICDATA", src)
put("niPairUnitsHi", f"{-pr['d_units95'][0]:.0f}", "PUBLICDATA", src)
put("niPairCoverHi", pct(pr["d_cover95"][1]), "PUBLICDATA", src)
pn = W["paired"]["hier_pred-nb_pooled|new"]
put("niPairNewUnits", f"{pn['d_units']:.0f}", "PUBLICDATA", src)
put("niPairNewLo", f"{pn['d_units95'][0]:.0f}", "PUBLICDATA", src)
put("niPairNewHi", f"{pn['d_units95'][1]:.0f}", "PUBLICDATA", src)


def lci(key, block):
    d = block["log_score_diffs"][key]
    return f"{d['d']:+.3f}", f"{d['ci95'][0]:+.3f}", f"{d['ci95'][1]:+.3f}"


for key, tag in (("hier_pred-nb_pooled|known", "LsNbKnown"), ("hier_pred-nb_pooled|new", "LsNbNew"),
                 ("hier_pred-poisson|known", "LsPoisKnown")):
    a, lo, hi = lci(key, W)
    put(f"ni{tag}", f"${a}$", "PUBLICDATA", src)
    put(f"ni{tag}Lo", f"${lo}$", "PUBLICDATA", src)
    put(f"ni{tag}Hi", f"${hi}$", "PUBLICDATA", src)
a, lo, hi = lci("hier_pred-nb_pooled|new", Fw)
put("niLsFwdNew", f"${a}$", "PUBLICDATA", src)
put("niLsFwdNewLo", f"${lo}$", "PUBLICDATA", src)
put("niLsFwdNewHi", f"${hi}$", "PUBLICDATA", src)
put("niTailHier", pct(W["log_scores"]["hier_pred|known"]["exceed_q99"], 1), "PUBLICDATA", src)
put("niTailNb", pct(W["log_scores"]["nb_pooled|known"]["exceed_q99"], 1), "PUBLICDATA", src)
cst = W["cost_units_plus_ratio_times_shortage"]
for r in ("1", "100"):
    for m, tag in (("hier_pred|known", "HierK"), ("nb_pooled|known", "NbK"), ("poisson|known", "PoisK"),
                   ("hier_pred|new", "HierN"), ("nb_pooled|new", "NbN")):
        put(f"niCost{tag}R{'One' if r == '1' else 'Hundred'}", f"{cst[r][m]:.0f}", "PUBLICDATA", src)
cv = W["cover_units_curve"]
put("niCurveHierCover", pct(cv["hier_pred|known"]["0.95"]["cover"]), "PUBLICDATA", src)
put("niCurveHierUnits", f"{cv['hier_pred|known']['0.95']['units']:.0f}", "PUBLICDATA", src)
put("niCurveNbCover", pct(cv["nb_pooled|known"]["0.995"]["cover"]), "PUBLICDATA", src)
put("niCurveNbUnits", f"{cv['nb_pooled|known']['0.995']['units']:.0f}", "PUBLICDATA", src)

with open(os.path.join(HERE, "numbers_v3.tex"), "w", encoding="utf-8") as f:
    f.write("% Generated by make_numbers_v3.py -- do not edit by hand.\n")
    for k, v in MACROS:
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
with open(os.path.join(HERE, "numbers_v3_manifest.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=["macro", "value", "provenance", "source", "note"])
    wr.writeheader()
    wr.writerows(MANIFEST)
print(len(MACROS), "macros")
