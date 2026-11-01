"""
Generate numbers.tex and the LaTeX tables of the manuscript from the result
files, so that no number is typed by hand and every one carries a provenance
tag.

Provenance tags, printed in numbers_manifest.csv next to every macro:
  ANALYTIC   exact calculation or numerical quadrature, no data
  SIM        simulation under assumed parameters
  REALDATA   measured, from the source named in the manifest
There is no fourth category: nothing in this article is measured on shipboard
equipment.

    python make_numbers.py
"""
import csv
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
TAB = os.path.join(HERE, "tables")
os.makedirs(TAB, exist_ok=True)

MACROS, MANIFEST = [], []


def put(name, value, tag, source, note=""):
    MACROS.append((name, value))
    MANIFEST.append(dict(macro=name, value=value, provenance=tag,
                         source=source, note=note))


def pct(x, d=1):
    return f"{100 * float(x):.{d}f}"


def num(x, d=2):
    return f"{float(x):.{d}f}"


# ---------------------------------------------------------------- theory
T = json.load(open(os.path.join(DATA, "theory_cover.json")))
src = "analysis/theory_cover.py -> data/theory_cover.json"

q = T["quadrature"]
put("quadGrid", f"{q['max_abs_change_grid']:.0e}".replace("e-", "\\times 10^{-") + "}",
    "ANALYTIC", src, "stability of the Simpson grid")
put("quadNodes", str(int(q["grid_nodes"])), "ANALYTIC", src)
put("quadGH", pct(q["max_abs_error_gauss_hermite60"], 1), "ANALYTIC", src,
    "error of 60-node Gauss-Hermite, percentage points")

ins = {b["kappa"]: b for b in T["kappa_insufficiency"]}
for k, tagname in ((0.25, "Qtr"), (0.5, "Half"), (1.0, "One"), (2.0, "Two")):
    b = ins[k]
    put(f"spreadK{tagname}Nn", pct(b["spread"]["p99"], 1), "ANALYTIC", src,
        f"spread of exact 99 pct cover at kappa={k}")
    put(f"spreadK{tagname}Nt", pct(b["spread"]["p90"], 1), "ANALYTIC", src,
        f"spread of exact 90 pct cover at kappa={k}")
d2 = pd.DataFrame(ins[2.0]["rows"])
d2 = d2[d2.p == 99]
put("inspTwoLo", pct(d2.exact.min(), 1), "ANALYTIC", src)
put("inspTwoHi", pct(d2.exact.max(), 1), "ANALYTIC", src)
put("inspTwoRule", pct(d2.rule.iloc[0], 1), "ANALYTIC", src)

dom = T["rule_domain"]
put("domMaxGap", pct(dom["max_abs_overall"], 1), "ANALYTIC", src,
    "largest |closed form - exact| on the grid")
put("domOverstate", pct(dom["signed_range"][1], 1), "ANALYTIC", src,
    "largest amount by which the closed form OVERstates cover")
g = dom["grid"]
put("domBest", pct(g["k[0,0.25) L[20,inf)"]["max_abs"], 1), "ANALYTIC", src,
    "kappa<0.25, lambda>=20")
put("domSmallLam", pct(max(g[k]["max_abs"] for k in g if "L[0,1)" in k), 1),
    "ANALYTIC", src, "worst case for lambda<1")
put("domBigKappa", pct(max(g[k]["max_abs"] for k in g if "k[2.0,inf)" in k), 1),
    "ANALYTIC", src, "worst case for kappa>=2")
put("domModerate", pct(max(g[k]["max_abs"] for k in g
                           if ("k[0,0.25)" in k or "k[0.25,0.5)" in k) and "L[5,20)" in k), 1),
    "ANALYTIC", src, "kappa<0.5, 5<=lambda<20")

mar = T["margin"]
put("marMaxRel", pct(mar["max_abs_rel_err_overall"], 0), "ANALYTIC", src,
    "largest relative error of the sqrt(1+kappa^2) margin factor")
put("marMaxUnits", f"{mar['max_abs_units_err_overall']:.0f}", "ANALYTIC", src,
    "largest error of that factor in spare units")
put("marMedMid", pct(abs(mar["by_lambda"]["10<=lam<100"]["median_rel_err"]), 0),
    "ANALYTIC", src, "median relative error, 10<=lambda<100")

# ---------------------------------------------------------------- multiproduct
M = json.load(open(os.path.join(DATA, "multiproduct.json")))
src = "analysis/multiproduct.py -> data/multiproduct.json"
P = pd.DataFrame(M["plans"])
X = pd.DataFrame(M["cross"])
put("mpSigma", num(M["sigma"], 1), "ANALYTIC", src, "assumed total latent sd")
put("mpMean", f"{M['total_mean']:.0f}", "ANALYTIC", src, "assumed expected failures")
put("mpTrunc", f"{M['max_truncation']:.0e}".replace("e-", "\\times 10^{-") + "}",
    "ANALYTIC", src)
WORD = {1: "One", 2: "Two", 4: "Four", 8: "Eight"}
for Pn in (1, 2, 4, 8):
    a = int(P[(P.P == Pn) & (P.structure == "A: ship-wide only")].q99.iloc[0])
    b = int(P[(P.P == Pn) & (P.structure == "B: interaction only")].q99.iloc[0])
    m = int(P[(P.P == Pn) & (P.structure == "mixed: equal split")].q99.iloc[0])
    put(f"mpShip{WORD[Pn]}", str(a), "ANALYTIC", src)
    put(f"mpInter{WORD[Pn]}", str(b), "ANALYTIC", src)
    put(f"mpMixed{WORD[Pn]}", str(m), "ANALYTIC", src)
r = X[(X.P == 8) & (X.planned_under == "B: interaction only") &
      (X.truth == "A: ship-wide only")]
put("mpWrongCover", pct(r.delivered.iloc[0], 1), "ANALYTIC", src,
    "plan assuming interaction only, truth ship-wide, 8 products")
r2 = X[(X.P == 8) & (X.planned_under == "A: ship-wide only") &
       (X.truth == "B: interaction only")]
put("mpConsCover", pct(r2.delivered.iloc[0], 3), "ANALYTIC", src)

# ---------------------------------------------------------------- real data
A = json.load(open(os.path.join(DATA, "real_data_audit.json")))
src = "analysis/real_data_audit.py -> data/real_data_audit.json"
c = A["capacitor"]
put("capUnits", str(c["units"]), "REALDATA", src + " (NASA PCoE capacitor)")
put("capLevels", "/".join(str(x) for x in c["stress_levels"]), "REALDATA", src)
put("capHours", f"{c['test_hours']:,.0f}".replace(",", "\\,"), "REALDATA", src)
put("capCrit", num(c["failure_criterion_esr_ratio"], 1), "REALDATA", src)
put("capAnyCross", str(c["units_with_any_crossing"]), "REALDATA", src)
put("capSustained", str(c["units_with_sustained_crossing"]), "REALDATA", src)
put("capEndsAbove", str(c["units_ending_above_criterion"]), "REALDATA", src)
put("capMaxFinal", num(c["max_final_ratio"], 2), "REALDATA", src)
bys = {int(r["stress_V"]): r for r in c["by_stress"]}
VWORD = {10: "Ten", 12: "Twelve", 14: "Fourteen"}
for v in c["stress_levels"]:
    put(f"capFinal{VWORD[v]}", num(bys[v]["final_ratio_mean"], 2), "REALDATA", src)
put("capSd", num(c["within_stress_final_ratio_sd"], 3), "REALDATA", src)
put("nSources", str(len(A["sources"])), "REALDATA", src)
put("nSuitable", str(sum(1 for s in A["sources"] if s["meets_all_requirements"])),
    "REALDATA", src, "sources meeting all four requirements")

# ---------------------------------------------------------------- design sim
DS_PATH = os.path.join(DATA, "design_sim.json")
if os.path.exists(DS_PATH):
    D = json.load(open(DS_PATH))
    src = "analysis/design_sim.py + design_summary.py -> data/design_sim.json"
    S = pd.DataFrame(D["summary"])
    put("dsReps", str(D["reps"]), "SIM", src)
    put("dsEps", num(D["eps"], 1), "SIM", src, "assumed mission-to-mission sd")
    put("dsBound", num(D["bound"], 2), "SIM", src, "boundary threshold")

    def cells(**kw):
        m = pd.Series(True, index=S.index)
        for k, v in kw.items():
            m &= np.isclose(S[k], v)
        return S[m]

    # can a campaign tell the two allocations apart?
    one = cells(M=1)["allocation_recovered"].dropna()
    put("dsAllocOneLo", pct(one.min(), 0), "SIM", src, "M=1, worst case")
    put("dsAllocOneHi", pct(one.max(), 0), "SIM", src, "M=1, best case")
    four = cells(S=10, M=4, m_exp=1.0)["allocation_recovered"].dropna()
    put("dsAllocTenLo", pct(four.min(), 0), "SIM", src)
    put("dsAllocTenHi", pct(four.max(), 0), "SIM", src)
    tw = cells(S=20, M=4, m_exp=1.0)["allocation_recovered"].dropna()
    put("dsAllocTwentyLo", pct(tw.min(), 0), "SIM", src)

    # a spurious term: the truth is zero yet the estimate leaves the boundary
    sp_ship = 1.0 - S[np.isclose(S.tau, 0.0)].tau_bound
    put("dsSpurShipLo", pct(sp_ship.min(), 0), "SIM", src,
        "tau=0 yet a ship-wide term is estimated")
    put("dsSpurShipHi", pct(sp_ship.max(), 0), "SIM", src)
    sp_in = 1.0 - S[np.isclose(S.omega, 0.0)].omega_bound
    put("dsSpurInterLo", pct(sp_in.min(), 0), "SIM", src,
        "omega=0 yet an interaction term is estimated")
    put("dsSpurInterHi", pct(sp_in.max(), 0), "SIM", src)
    big = cells(S=40, M=4, m_exp=1.0)
    put("dsSpurShipForty", pct(1.0 - big[np.isclose(big.tau, 0.0)].tau_bound.iloc[0], 0),
        "SIM", src)
    put("dsSpurInterForty", pct(1.0 - big[np.isclose(big.omega, 0.0)].omega_bound.iloc[0], 0),
        "SIM", src)

    # precision of the two components, equal-split truth
    eq = cells(tau=0.42, omega=0.42)
    r = eq[(eq.S == 10) & (eq.M == 4) & np.isclose(eq.m_exp, 1.0)].iloc[0]
    put("dsTenFail", f"{r.failures_median:.0f}", "SIM", src)
    put("dsTenTauRmse", num(r.tau_rmse, 2), "SIM", src)
    put("dsTenOmRmse", num(r.omega_rmse, 2), "SIM", src)
    put("dsTenCover", pct(r.cover_mean, 1), "SIM", src)
    put("dsTenCoverPTen", pct(r.cover_p10, 1), "SIM", src)
    r1 = eq[(eq.S == 10) & (eq.M == 1)].iloc[0]
    put("dsOneFail", f"{r1.failures_median:.0f}", "SIM", src)
    put("dsOneOmBound", pct(r1.omega_bound, 0), "SIM", src)
    r4 = eq[(eq.S == 40) & (eq.M == 4)].iloc[0]
    put("dsFortyTauRmse", num(r4.tau_rmse, 2), "SIM", src)
    put("dsFortyCover", pct(r4.cover_mean, 1), "SIM", src)
    put("dsFortyCoverPTen", pct(r4.cover_p10, 1), "SIM", src)
    r2 = eq[(eq.S == 20) & (eq.M == 4)].iloc[0]
    put("dsTwentyCover", pct(r2.cover_mean, 1), "SIM", src)
    put("dsTwentyCoverPTen", pct(r2.cover_p10, 1), "SIM", src)
    # worst tenth under a ship-wide truth with one mission
    sw = cells(tau=0.6, omega=0.0, S=10, M=1).iloc[0]
    put("dsShipOneCoverPTen", pct(sw.cover_p10, 1), "SIM", src)

# ---------------------------------------------------------------- grid extent
# The figures in Table I are maxima observed on a finite grid, not analytical
# bounds; its extent has to be reported with them.
G_ = pd.read_csv(os.path.join(DATA, "theory_rule_grid.csv"))
src = "analysis/theory_cover.py -> data/theory_rule_grid.csv"
put("gridPts", f"{len(G_):,}".replace(",", "\\,"), "ANALYTIC", src)
put("gridLam", str(int(G_.lam.nunique())), "ANALYTIC", src)
put("gridSig", str(int(G_.sigma.nunique())), "ANALYTIC", src)
put("gridLamLo", num(G_.lam.min(), 2), "ANALYTIC", src)
put("gridLamHi", f"{G_.lam.max():.0f}", "ANALYTIC", src)
put("gridSigLo", num(G_.sigma.min(), 2), "ANALYTIC", src)
put("gridSigHi", num(G_.sigma.max(), 1), "ANALYTIC", src)

# ---------------------------------------------------------------- stock vector
SV = os.path.join(DATA, "stock_vector.json")
if os.path.exists(SV):
    V = json.load(open(SV))
    src = "analysis/stock_vector.py -> data/stock_vector.json"
    R_ = pd.DataFrame(V["rows"])
    q99 = R_[R_.level == 99]

    def sv(P, structure, col):
        r = q99[(q99.P == P) & (q99.structure == structure)]
        return r[col].iloc[0]

    for Pn in (1, 2, 4, 8):
        put(f"svJoint{WORD[Pn]}", str(int(sv(Pn, "ship-wide", "s_joint"))),
            "ANALYTIC", src, "per-reference stock, joint criterion, ship-wide")
        put(f"svJointTot{WORD[Pn]}", str(int(sv(Pn, "ship-wide", "total_joint"))),
            "ANALYTIC", src)
        put(f"svMarg{WORD[Pn]}", str(int(sv(Pn, "ship-wide", "s_marginal"))),
            "ANALYTIC", src, "per-reference marginal quantile")
        put(f"svMargTot{WORD[Pn]}", str(int(sv(Pn, "ship-wide", "total_marginal"))),
            "ANALYTIC", src)
    # joint cover actually delivered by per-reference marginal quantiles
    put("svNaiveShip", pct(sv(8, "ship-wide", "joint_cover_of_marginal_plan"), 1),
        "ANALYTIC", src, "P=8, ship-wide")
    put("svNaiveInter", pct(sv(8, "interaction", "joint_cover_of_marginal_plan"), 1),
        "ANALYTIC", src, "P=8, interaction")
    put("svNaiveShipFour", pct(sv(4, "ship-wide", "joint_cover_of_marginal_plan"), 1),
        "ANALYTIC", src)
    # the reversal: joint criterion needs no more stock under a shared deviation
    put("svJointShipTwo", str(int(sv(2, "ship-wide", "s_joint"))), "ANALYTIC", src)
    put("svJointInterTwo", str(int(sv(2, "interaction", "s_joint"))), "ANALYTIC", src)
    put("svTotJointEight", str(int(sv(8, "ship-wide", "total_joint"))), "ANALYTIC", src)
    put("svTotJointInterEight", str(int(sv(8, "interaction", "total_joint"))),
        "ANALYTIC", src)

# ---------------------------------------------------------------- tables
def w(name, text):
    open(os.path.join(TAB, name), "w", encoding="utf-8").write(text)


# T: kappa is not sufficient
lines = []
for k in (0.25, 0.5, 1.0, 2.0):
    b = ins[k]
    d = pd.DataFrame(b["rows"])
    d = d[d.p == 99].sort_values("lam")
    sel = d[d.lam.isin([0.5, 2.0, 10.0, 200.0])]
    cells = " & ".join(pct(x, 1) for x in sel.exact)
    lines.append(f"{k:g} & {pct(d.rule.iloc[0],1)} & {cells} & "
                 f"{pct(b['spread']['p99'],1)} \\\\")
w("tab_kappa.tex", "\n".join(lines) + "\n")

# T: domain of the closed form
order = [("$\\kappa<0.25$", "k[0,0.25)"), ("$0.25\\le\\kappa<0.5$", "k[0.25,0.5)"),
         ("$0.5\\le\\kappa<1$", "k[0.5,1.0)"), ("$1\\le\\kappa<2$", "k[1.0,2.0)"),
         ("$\\kappa\\ge2$", "k[2.0,inf)")]
lam_order = ["L[0,1)", "L[1,5)", "L[5,20)", "L[20,inf)"]
lines = []
for lab, kk in order:
    cells = []
    for ll in lam_order:
        key = f"{kk} {ll}"
        cells.append(pct(g[key]["max_abs"], 1) if key in g else "--")
    lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
w("tab_domain.tex", "\n".join(lines) + "\n")

# T: multi-product plans
lines = []
for Pn in (1, 2, 4, 8):
    a = int(P[(P.P == Pn) & (P.structure == "A: ship-wide only")].q99.iloc[0])
    m = int(P[(P.P == Pn) & (P.structure == "mixed: equal split")].q99.iloc[0])
    b = int(P[(P.P == Pn) & (P.structure == "B: interaction only")].q99.iloc[0])
    cw = X[(X.P == Pn) & (X.planned_under == "B: interaction only") &
           (X.truth == "A: ship-wide only")].delivered.iloc[0]
    lines.append(f"{Pn} & {a} & {m} & {b} & {pct(cw,1)} \\\\")
w("tab_multiproduct.tex", "\n".join(lines) + "\n")

# T: evidence audit
lines = []
for s in A["sources"]:
    miss = "none" if not s["lacks"] else f"{len(s['lacks'])} of 4"
    verdict = s["verdict"].split(":")[0]
    lines.append(f"{s['name']} & {s['environment']} & {miss} & {verdict} \\\\")
w("tab_audit.tex", "\n".join(lines) + "\n")

# T: capacitor context
lines = []
for v in c["stress_levels"]:
    r = bys[v]
    lines.append(f"{v} & {int(r['units'])} & {num(r['max_ratio_mean'],2)} & "
                 f"{num(r['final_ratio_mean'],2)} & {int(r['any_crossing'])} & "
                 f"{int(r['sustained'])} & {int(r['ends_above'])} \\\\")
w("tab_capacitor.tex", "\n".join(lines) + "\n")

# T: design simulation
if os.path.exists(DS_PATH):
    lines = []
    for _, r in S.sort_values(["tau", "omega", "S", "M"]).iterrows():
        alloc = "--" if not np.isfinite(r.allocation_recovered) else pct(r.allocation_recovered, 0)
        lines.append(f"{num(r.tau,2)} & {num(r.omega,2)} & {r.S:.0f} & {r.M:.0f} & "
                     f"{r.m_exp:.0f} & {r.failures_median:.0f} & {pct(r.tau_bound,0)} & "
                     f"{pct(r.omega_bound,0)} & {num(r.tau_rmse,2)} & {num(r.omega_rmse,2)} & "
                     f"{alloc} & {pct(r.cover_mean,1)} & {pct(r.cover_p10,1)} \\\\")
    w("tab_design.tex", "\n".join(lines) + "\n")


# T: per-reference stock vector against the aggregate plan
if os.path.exists(SV):
    lines = []
    for Pn in (2, 4, 8):
        for st in ("ship-wide", "interaction"):
            r = q99[(q99.P == Pn) & (q99.structure == st)].iloc[0]
            agg = int(P[(P.P == Pn) & (P.structure ==
                        ("A: ship-wide only" if st == "ship-wide"
                         else "B: interaction only"))].q99.iloc[0])
            lines.append(f"{Pn} & {st} & {int(r.s_marginal)} & "
                         f"{pct(r.joint_cover_of_marginal_plan,1)} & "
                         f"{int(r.s_joint)} & {int(r.total_joint)} & {agg} \\\\")
    w("tab_stock.tex", "\n".join(lines) + "\n")

# ---------------------------------------------------------------- emit
with open(os.path.join(HERE, "numbers.tex"), "w", encoding="utf-8") as f:
    f.write("% Generated by make_numbers.py -- do not edit by hand.\n")
    for n, v in MACROS:
        f.write(f"\\newcommand{{\\{n}}}{{{v}}}\n")
with open(os.path.join(HERE, "numbers_manifest.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=["macro", "value", "provenance", "source", "note"])
    wr.writeheader()
    wr.writerows(MANIFEST)

by_tag = pd.Series([m["provenance"] for m in MANIFEST]).value_counts()
print(f"{len(MACROS)} macros written to numbers.tex")
print(by_tag.to_string())
print("tables:", sorted(os.listdir(TAB)))
