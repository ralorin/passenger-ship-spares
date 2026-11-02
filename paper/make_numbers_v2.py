"""
Numbers and tables for the second round of experiments (data/v2), written to
numbers_v2.tex and numbers_v2_manifest.csv with the same provenance rule as
make_numbers.py, plus one more tag:
  PUBLICDATA  measured, from the public street-lighting records (E7)

    python make_numbers_v2.py
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


def sci(x):
    m, e = f"{x:.0e}".split("e")
    return f"{m}\\times 10^{{{int(e)}}}"


def thou(n):
    """Integer with a thin space as thousands separator."""
    return f"{int(n):,}".replace(",", "\\,")


# ------------------------------------------------- one population (E6) --
src = "analysis/single_plan.py -> data/v2/single_plan_grid.csv"
G = pd.read_csv(os.path.join(V2, "single_plan_grid.csv"))
put("spLamN", str(G.lam.nunique()), "ANALYTIC", src)
put("spLamHi", f"{G.lam.max():g}", "ANALYTIC", src)
put("spLamLo", f"{G.lam.min():g}", "ANALYTIC", src)
put("spSigLo", f"{G.sigma.min():g}", "ANALYTIC", src)
put("spSigHi", f"{G.sigma.max():g}", "ANALYTIC", src)
put("spPts", f"{len(G):,}".replace(",", "\\,"), "ANALYTIC", src)
names = dict(poisson="Poisson", kappa="Margin factor", negbin="Neg. binomial")
lines = []
for key, label in names.items():
    G[f"short_{key}"] = (G.level - G[f"cover_{key}"]).clip(lower=0)
    cells = [label]
    for lev in (0.90, 0.95, 0.99):
        g = G[np.isclose(G.level, lev)]
        cells += [pct(g[f"short_{key}"].max(), 1),
                  f"${int(g[f'diff_{key}'].min())}$/$+{int(g[f'diff_{key}'].max())}$"]
    lines.append(" & ".join(cells) + " \\\\")
    g99 = G[np.isclose(G.level, 0.99)]
    tag = dict(poisson="Pois", kappa="Kap", negbin="Nb")[key]
    put(f"sp{tag}Short", pct(g99[f"short_{key}"].max(), 1), "ANALYTIC", src, "worst shortfall at 99 pct, pp")
    put(f"sp{tag}Units", str(int(-g99[f"diff_{key}"].min())), "ANALYTIC", src, "most units short at 99 pct")
    put(f"sp{tag}Equal", pct((g99[f"diff_{key}"] == 0).mean(), 0), "ANALYTIC", src, "share equal to exact plan")
w("tab_single.tex", "\n".join(lines) + "\n")
g9 = G[G.level < 0.98]
put("spNbOver", str(int(g9.diff_negbin.max())), "ANALYTIC", src, "largest NB excess at 90-95 pct")
cf = G[np.isclose(G.level, 0.99)]
put("spCfErr", pct(cf.cf_error.abs().max(), 1), "ANALYTIC", src, "closed form vs exact cover of the Poisson plan")
op = cf[(cf.lam <= 20) & (cf.sigma <= 0.6)]
put("spNbShortOp", pct(op.short_negbin.max(), 2), "ANALYTIC", src, "lambda<=20, sigma<=0.6")
put("spKapShortOp", pct(op.short_kappa.max(), 1), "ANALYTIC", src, "lambda<=20, sigma<=0.6")
put("spPoisShortOp", pct(op.short_poisson.max(), 1), "ANALYTIC", src, "lambda<=20, sigma<=0.6")

# ------------------------------------------------- stock vectors (E5) ----
src = "analysis/robust_plan.py -> data/v2/robust_part1.csv, robust_part2.csv"
R1 = pd.read_csv(os.path.join(V2, "robust_part1.csv"))
uneq = R1[(R1.means == "skewed") | (R1.costs == "skewed")]
eq = R1[(R1.means == "equal") & (R1.costs == "equal")]
put("rvSaveMed", pct(uneq.saving_vs_common.median(), 0), "ANALYTIC", src, "median cost saving, unequal means or costs")
put("rvSaveMax", pct(R1.saving_vs_common.max(), 0), "ANALYTIC", src)
put("rvSaveEq", pct(eq.saving_vs_common.median(), 0), "ANALYTIC", src)
put("rvMargMin", pct(R1.cover_marginal.min(), 1), "ANALYTIC", src, "joint cover of per-reference quantiles, worst")
ex = R1[(R1.P == 8) & (R1.total == 8) & (R1.means == "equal") & (R1.costs == "equal") & (R1.f == 1.0)].iloc[0]
put("rvExOpt", str(int(ex.units_opt)), "ANALYTIC", src, "P=8, 8 expected, ship-wide")
put("rvExCom", str(int(ex.units_common)), "ANALYTIC", src)
R2 = pd.read_csv(os.path.join(V2, "robust_part2.csv"))
lines = []
for P in (4, 8):
    for L in (8.0, 20.0):
        k = R2[(R2.P == P) & (R2.total == L)]
        base = k[k.plan == "known"].set_index("f")
        cells = [str(P), f"{L:g}"]
        for plan in ("conservative", "averaged"):
            g = k[k.plan == plan].set_index("f")
            cells += [pct(g.joint_cover.min(), 1), f"+{int((g.joint_units - base.joint_units).max())}",
                      pct(g.agg_cover.min(), 1), f"+{int((g.agg_units - base.agg_units).max())}"]
        lines.append(" & ".join(cells) + " \\\\")
w("tab_robust.tex", "\n".join(lines) + "\n")
for plan, tag in (("conservative", "Cons"), ("averaged", "Avg")):
    g = R2[R2.plan == plan]
    put(f"rv{tag}JointMin", pct(g.joint_cover.min(), 1), "ANALYTIC", src)
    put(f"rv{tag}AggMin", pct(g.agg_cover.min(), 1), "ANALYTIC", src)

# ------------------------------------------------- federated (E4) --------
src = "analysis/federated.py -> data/v2/federated.csv"
F = pd.read_csv(os.path.join(V2, "federated.csv"))
e = F[F.kind == "equivalence"]
put("fedN", str(len(e)), "SIM", src)
put("fedSameTheta", sci(e.d_loglik_same_theta.max()), "SIM", src, "max |Laplace difference| at equal theta")
put("fedSigMed", f"{e.d_sigma_max.median():.3f}", "SIM", src)
put("fedBetaMax", f"{e.d_beta_max.max():.3f}", "SIM", src)
put("fedDiverge", str(int((e.d_loglik_fit > 1e-3).sum())), "SIM", src, "fits whose optima differ by >1e-3 in loglik")
put("fedRounds", f"{e.rounds.median():.0f}", "SIM", src)
put("fedFloats", f"{e.floats.median():,.0f}".replace(",", "\\,"), "SIM", src)
put("fedOneShot", f"{e.oneshot_floats.median():,.0f}".replace(",", "\\,"), "SIM", src)
put("fedRatio", f"{(e.floats / e.oneshot_floats).median():.0f}", "SIM", src)

# ------------------------------------------------- campaigns (E1) --------
src = "analysis/campaign.py -> data/v2/campaign_summary.csv, campaign_reps.csv.gz"
T = pd.read_csv(os.path.join(V2, "campaign_summary.csv"))
Rp = pd.read_csv(os.path.join(V2, "campaign_reps.csv.gz"))
put("dsCells", str(len(T)), "SIM", src)
put("cdReps", f"{int(T.reps.max()):,}".replace(",", "\\,"), "SIM", src)
put("dsCampaigns", f"{len(Rp):,}".replace(",", "\\,"), "SIM", src)
put("dsFailed", str(int(T.failed.sum())), "SIM", src)
core = T[(T.P == 4) & (T.m == 1.0) & (T.eps == 0.3) & (~T.unbal)]
t1 = pd.concat([core[core.tau == 0].reject_tau, core[(core.omega == 0) & (core.M > 1)].reject_omega])
put("dsTypeOneMax", pct(t1.max(), 1), "SIM", src, "rejection at 5 pct when the component is zero")
put("dsTypeOneMed", pct(t1.median(), 1), "SIM", src)
put("dsMcseRej", pct(math.sqrt(0.05 * 0.95 / 1000), 1), "SIM", src, "MCSE of a 5 pct rate, 1000 reps")
ci = T.dropna(subset=["ci_tau_cover"])
cc = pd.concat([ci.ci_tau_cover, ci.ci_omega_cover])
put("dsCiLo", pct(cc.min(), 1), "SIM", src)
put("dsCiHi", pct(cc.max(), 1), "SIM", src)
put("dsCiReps", str(int(ci.ci_tau_n.max())), "SIM", src)
oo = core[core.tau != core.omega]
om1 = oo[(oo.M == 1) & (oo.tau == 0)]
put("dsOrderOneLo", pct(om1.order_ok.min(), 0), "SIM", src, "M=1, interaction-only truth")
put("dsOrderOneHi", pct(om1.order_ok.max(), 0), "SIM", src, "M=1, interaction-only truth")
put("dsOrderFour", pct(oo[(oo.M >= 4) & (oo.S >= 10)].order_ok.min(), 0), "SIM", src)
eqs = core[(core.tau == 0.42) & (core.omega == 0.42)]
lines = []
for S in sorted(eqs.S.unique()):
    g = eqs[eqs.S == S].set_index("M")
    cells = [str(int(S))] + [pct(g.loc[M, "reject_tau"], 0) for M in (1, 2, 4, 8)] \
        + [pct(g.loc[M, "reject_omega"], 0) for M in (1, 2, 4, 8)]
    lines.append(" & ".join(cells) + " \\\\")
w("tab_power.tex", "\n".join(lines) + "\n")
put("dsPowOmTwentyFour", pct(eqs[(eqs.S == 20) & (eqs.M == 4)].reject_omega.iloc[0], 0), "SIM", src)
put("dsPowOmFortyEight", pct(eqs[(eqs.S == 40) & (eqs.M == 8)].reject_omega.iloc[0], 0), "SIM", src)
put("dsPowOmTenFour", pct(eqs[(eqs.S == 10) & (eqs.M == 4)].reject_omega.iloc[0], 0), "SIM", src)

meth = ["oracle", "plugin", "predictive", "conservative", "poisson"]
Rp = Rp.merge(T[["cell", "S", "M", "P", "m", "eps", "unbal"]], on="cell")
lines = []
m4 = core[core.M == 4]
r4 = Rp[Rp.cell.isin(m4.cell)]
for S in sorted(m4.S.unique()):
    g = m4[m4.S == S]
    rr = r4[r4.S == S]
    cells = [str(int(S))] + [pct(g[f"joint_{m}_cover"].mean(), 1) for m in meth[1:]] \
        + [pct(g["joint_predictive_cover_p10"].mean(), 1),
           f"{rr.joint_units_predictive.median():.0f}", f"{rr.joint_units_oracle.median():.0f}"]
    lines.append(" & ".join(cells) + " \\\\")
w("tab_campaign_cover.tex", "\n".join(lines) + "\n")
put("dsPredCoverTen", pct(m4[m4.S == 10].joint_predictive_cover.mean(), 1), "SIM", src)
put("dsPlugCoverFive", pct(m4[m4.S == 5].joint_plugin_cover.mean(), 1), "SIM", src)
put("dsPoisLo", pct(m4.joint_poisson_cover.min(), 1), "SIM", src)
put("dsPoisHi", pct(m4.joint_poisson_cover.max(), 1), "SIM", src)
o = T[(T.S == 20) & (T.M == 4)]
put("dsPoisMThree", pct(T[(T.M == 4) & (T.m == 3.0)].joint_poisson_cover.mean(), 1), "SIM", src)
put("dsPoisPEight", pct(o[o.P == 8].joint_poisson_cover.mean(), 1), "SIM", src)
put("dsPredMin", pct(T[(T.M >= 2) & (T.S >= 10)].joint_predictive_cover.min(), 1), "SIM", src,
    "worst cell, S>=10, M>=2")
put("dsPredMThree", pct(T[(T.M == 4) & (T.m == 3.0)].joint_predictive_cover.mean(), 1), "SIM", src)
r5 = Rp[Rp.cell.isin(m4[m4.S == 5].cell)]
put("dsUnitsFiveMean", f"{r5.joint_units_predictive.mean():.0f}", "SIM", src)
put("dsUnitsFiveMed", f"{r5.joint_units_predictive.median():.0f}", "SIM", src)
put("dsUnitsOracle", f"{r5.joint_units_oracle.median():.0f}", "SIM", src)

# ------------------------------------------------- fleet (E2, E3) --------
FP = json.load(open(os.path.join(HERE, "..", "analysis", "fleet_params.json"), encoding="utf-8"))
put("flShips", str(sum(c["ships"] for c in FP["classes"].values())), "SIM", "analysis/fleet_params.json")
FS = os.path.join(V2, "fleet_summary.csv")
if os.path.exists(FS):
    src = "analysis/fleet.py -> data/v2/fleet_summary.csv"
    S = pd.read_csv(FS).set_index(["scenario", "method", "kind"]).sort_index()
    R = pd.read_csv(os.path.join(V2, "fleet_reps.csv.gz"))
    R["upm"] = R.units / R.n
    order = ["mtbf", "mtbf_joint", "nb_pooled", "eb_gamma", "hier_marg", "hier_plugin", "hier_pred", "hier_shock"]
    scen = dict(clean="Clean", baseline="Baseline", heavy_tail="Heavy tail", shocks="Supply events",
                misclass="Misrecorded", sparse="Sparse", datasheet_ok="Datasheet right")
    stat = ["nb_pooled", "eb_gamma", "hier_plugin", "hier_pred", "hier_shock"]
    misled = [s for s in scen if s != "datasheet_ok"]

    def cell(sc, m, col, fmt):
        return " / ".join(fmt(S.loc[(sc, m, k), col]) for k in ("known", "new"))

    lines = [" & ".join([lab] + [cell(sc, m, "cover", pct) for m in order]) + " \\\\"
             for sc, lab in scen.items()]
    w("tab_fleet_cover.tex", "\n".join(lines) + "\n")
    lines = [" & ".join([lab] + [cell(sc, m, "units", lambda x: f"{x:.1f}") for m in order]) + " \\\\"
             for sc, lab in scen.items()]
    w("tab_fleet_units.tex", "\n".join(lines) + "\n")

    def span(methods, scens, col="cover"):
        v = S.loc[(scens, methods, slice(None)), col]
        return v.min(), v.max()

    for m, tag in (("mtbf", "Mtbf"), ("mtbf_joint", "MtbfJoint"), ("hier_marg", "Marg")):
        lo, hi = span([m], misled)
        put(f"fl{tag}Lo", pct(lo), "SIM", src, "cover range over scenarios with optimistic datasheets")
        put(f"fl{tag}Hi", pct(hi), "SIM", src)
    lo, hi = span(stat, list(scen))
    put("flStatLo", pct(lo), "SIM", src, "cover range of the fitted joint plans, all scenarios")
    put("flStatHi", pct(hi), "SIM", src)
    for m, tag in (("mtbf", "Mtbf"), ("mtbf_joint", "MtbfJoint")):
        put(f"flRight{tag}", pct(min(S.loc[("datasheet_ok", m, k), "cover"] for k in ("known", "new"))),
            "SIM", src, "datasheets right, lower of known/new")
    for sc, st in (("baseline", "Base"), ("sparse", "Sparse"), ("datasheet_ok", "Right"),
                   ("shocks", "Shock"), ("heavy_tail", "Heavy")):
        for m, tag in (("mtbf", "Mtbf"), ("mtbf_joint", "MtbfJoint"), ("nb_pooled", "Nb"),
                       ("eb_gamma", "Eb"), ("hier_plugin", "Plug"), ("hier_pred", "Pred"),
                       ("hier_shock", "Shock"), ("hier_marg", "Marg")):
            for kind, kt in (("known", "Known"), ("new", "New")):
                put(f"fl{st}{tag}{kt}", pct(S.loc[(sc, m, kind), "cover"]), "SIM", src)
                put(f"fl{st}{tag}{kt}Units", f"{S.loc[(sc, m, kind), 'units']:.1f}", "SIM", src)
    b = S.loc["baseline"]
    put("flRatio", f"{b.loc[('hier_pred', 'known'), 'units'] / b.loc[('mtbf', 'known'), 'units']:.1f}",
        "SIM", src, "units of the predictive plan over the datasheet plan, baseline, known ships")
    for m, tag in (("nb_pooled", "Nb"), ("eb_gamma", "Eb"), ("hier_pred", "Pred")):
        put(f"flLs{tag}", f"{b.loc[(m, 'known'), 'log_score']:.3f}", "SIM", src, "mean log score, baseline")
    put("flPitMax", f"{S.pit_max_dev.max():.3f}", "SIM", src, "largest deviation of a PIT decile from 0.1")
    put("flMcseMax", f"{100 * S.loc[(slice(None), stat, slice(None)), 'cover_mcse'].max():.2f}", "SIM", src,
        "largest Monte Carlo standard error of a fitted plan's cover, points")
    med = R.groupby(["scenario", "method", "kind"]).upm.transform("median")
    big = R[(R.upm > 3 * med) & R.method.isin(stat + ["hier_marg"])]
    put("flBigFleets", str(big[["scenario", "rep"]].drop_duplicates().shape[0]), "SIM", src,
        "fleets in which some fitted plan exceeds three times its median units")
    put("flBigHeavy", str(big[big.scenario == "heavy_tail"].rep.nunique()), "SIM", src)
    put("flFleets", str(int(S.fleets.max())), "SIM", src)
    put("flMissions", thou(S.loc[("baseline", "mtbf"), "missions"].sum()), "SIM", src)

# ------------------------------------------------- street lighting (E7) --
NI = os.path.join(V2, "ni_results.json")
if os.path.exists(NI):
    src = "analysis/ni_streetlights.py -> data/v2/ni_results.json, ni_results.csv"
    J = json.load(open(NI))
    C = J["components"]
    sp = J["splits"]
    put("niTrainFail", thou(J["train_failures"]), "PUBLICDATA", src)
    put("niWithinTrain", thou(sp["within"]["train_failures"]), "PUBLICDATA", src)
    put("niWithinTest", thou(sp["within"]["test_failures"]), "PUBLICDATA", src)
    put("niForwardTest", thou(sp["forward"]["test_failures"]), "PUBLICDATA", src)
    put("niAreas", str(J["areas"]), "PUBLICDATA", src)
    put("niFamilies", str(len(J["families"])), "PUBLICDATA", src)
    for k, tag in (("area", "Area"), ("area_family", "Inter"), ("area_month", "Month")):
        put(f"niSig{tag}", f"{C['sigma'][k]:.2f}", "PUBLICDATA", src)
    for k, tag in (("area", "Area"), ("af", "Inter"), ("am", "Month")):
        lo, hi = C[f"ci_{k}"]
        put(f"niCi{tag}", f"{lo:.2f}--{hi:.2f}", "PUBLICDATA", src)
        p_ = C[f"p_{k}"]
        put(f"niP{tag}", "<10^{-20}" if p_ < 1e-20 else f"{p_:.2f}" if p_ >= 0.1 else f"{p_:.3f}",
            "PUBLICDATA", src)
    put("niPhi", f"{C['phi']:.1f}", "PUBLICDATA", src)
    fe = C["family_effects"]
    put("niFamLo", f"{math.exp(min(fe.values())):.2f}", "PUBLICDATA", src, "smallest family rate ratio")
    put("niFamHi", f"{math.exp(max(max(fe.values()), 0)):.2f}", "PUBLICDATA", src, "largest family rate ratio")
    Rn = pd.read_csv(os.path.join(V2, "ni_results.csv"))
    order = ["poisson", "poisson_jnt", "nb_pooled", "eb_gamma", "hier_marg", "hier_plugin", "hier_pred", "hier_shock"]
    label = dict(poisson="Poisson", poisson_jnt="Poisson, joint", nb_pooled="NB pooled",
                 eb_gamma="Gamma EB", hier_marg="Hier., marginal", hier_plugin="Hier., plug-in",
                 hier_pred="Hier., predictive", hier_shock="Hier., + month")
    lines = []
    for m in order:
        g = Rn[Rn.method == m].set_index(["split", "kind"])
        ls = g.loc[("within", "known"), "log_score"]
        lines.append(" & ".join([label[m],
            pct(g.loc[("within", "known"), "cover"], 1), f"{g.loc[('within', 'known'), 'units']:.0f}",
            "--" if pd.isna(ls) else f"${ls:.2f}$",
            pct(g.loc[("within", "new"), "cover"], 1), f"{g.loc[('within', 'new'), 'units']:.0f}",
            pct(g.loc[("forward", "known"), "cover"], 1), f"{g.loc[('forward', 'known'), 'units']:.0f}"]) + " \\\\")
        for split, kind in (("within", "known"), ("within", "new"), ("forward", "known")):
            tag = "ni" + m.title().replace("_", "") + split.title() + kind.title()
            put(tag + "Cover", pct(g.loc[(split, kind), "cover"], 1), "PUBLICDATA", src)
            put(tag + "Units", f"{g.loc[(split, kind), 'units']:.0f}", "PUBLICDATA", src)
    w("tab_ni.tex", "\n".join(lines) + "\n")
    put("niWithinMonths", str(int(Rn[(Rn.split == "within") & (Rn.method == "poisson")].n.max())), "PUBLICDATA", src)
    put("niForwardMonths", str(int(Rn[(Rn.split == "forward") & (Rn.method == "poisson")].n.max())), "PUBLICDATA", src)

# ------------------------------------------------- emit -------------------
with open(os.path.join(HERE, "numbers_v2.tex"), "w", encoding="utf-8") as f:
    f.write("% Generated by make_numbers_v2.py -- do not edit by hand.\n")
    for n, v in MACROS:
        f.write(f"\\newcommand{{\\{n}}}{{{v}}}\n")
with open(os.path.join(HERE, "numbers_v2_manifest.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=["macro", "value", "provenance", "source", "note"])
    wr.writeheader()
    wr.writerows(MANIFEST)
print(f"{len(MACROS)} macros;", pd.Series([m["provenance"] for m in MANIFEST]).value_counts().to_dict())
