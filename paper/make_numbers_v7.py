"""
Numbers and tables of the revision of 2026-11-07. Runs make_numbers_v6.py
first, then writes numbers_v7.tex (loaded last, so \\renewcommand corrects an
earlier value), the tables it names, and numbers_v7_manifest.csv.

  - Table I with the f = 0 searches certified by dynamic programming
    (analysis/stock_f0_dp.py) and the gap measured against the lower bound
  - the bridge contrast in units (both plans are unit-minimal)
  - certification counts of the fleet plans
  - the bootstrap campaigns rerun with the convergence rule
    (SPARES_RULE=1: data/v2/campaign_*_rule*) and the diagnosis of the
    extreme plans (analysis/campaign_diag.py)

    python make_numbers_v7.py
"""
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, "..", "data", "v2")
TAB = os.path.join(HERE, "tables")
OLD = set()
MACROS, MANIFEST, SEEN = [], [], set()


def run_previous():
    subprocess.run([sys.executable, os.path.join(HERE, "make_numbers_v6.py")], check=True, cwd=HERE)
    for name in ("numbers.tex", "numbers_v2.tex", "numbers_v3.tex", "numbers_v4.tex",
                 "numbers_v5.tex", "numbers_v6.tex"):
        for line in open(os.path.join(HERE, name), encoding="utf-8"):
            for cmd in ("\\newcommand{", "\\renewcommand{"):
                if line.startswith(cmd):
                    OLD.add(line[len(cmd):].split("}")[0])


def put(name, value, kind, src, note=""):
    assert name not in SEEN, name
    SEEN.add(name)
    cmd = "\\renewcommand" if "\\" + name in OLD else "\\newcommand"
    MACROS.append(f"{cmd}{{\\{name}}}{{{value}}}")
    MANIFEST.append((name, value, kind, src, note))


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def w(name, text):
    with open(os.path.join(TAB, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def thousands(n):
    return f"{int(n):,}".replace(",", "\\,")


def table_one():
    src = "analysis/stock_exact.py + analysis/stock_f0_dp.py -> data/v2/stock_exact_part1.csv, stock_f0_dp.json"
    T = pd.read_csv(os.path.join(V2, "stock_exact_part1.csv"))
    put("seCertBB", str(int(T.proved.astype(bool).sum())), "ANALYTIC", src, "certified by branch and bound")
    put("seUnprovedBB", str(int((~T.proved.astype(bool)).sum())), "ANALYTIC", src)
    D = json.load(open(os.path.join(V2, "stock_f0_dp.json"), encoding="utf-8"))
    dp = {}
    for d in D:
        assert d["certified"]
        m = ((T.P == d["P"]) & (T.total == d["total"]) & (T.means == d["means"])
             & (T.costs == d["costs"]) & (T.f == 0))
        assert m.sum() == 1
        T.loc[m, "cost_exact"] = d["dp_cost"]
        T.loc[m, "cost_indep"] = d["dp_cost"]      # the independence plan is the optimum at f = 0
        T.loc[m, "proved"] = True
        dp[(d["means"], d["costs"])] = d
    put("seDpBetter", str(int(sum(d["best_found"] - d["dp_cost"] > 0 for d in D))), "ANALYTIC", src)
    put("seDpMarginMin", f"{min(d['log_margin_below'] for d in D):.1e}", "ANALYTIC", src,
        "log(level) - best log cover one unit of cost below the optimum")
    uneq = (T.means == "skewed") | (T.costs == "skewed")
    cert = T.proved.astype(bool)
    sel = cert & (T.f > 0)
    rows = []
    for key, label in (("common", "Common stock"), ("equal_risk", "Equal risk"),
                       ("indep", "Independence ($f=0$)"), ("greedy", "Marginal analysis")):
        ex = T[f"cost_{key}"] / T.cost_exact - 1
        rows.append(f"{label} & {pct(ex[uneq & sel].median())} & {pct(ex[~uneq & sel].median())} & "
                    f"{pct(ex[cert].max())} & {pct(T[f'cover_{key}'].min())} \\\\")
    lab = {("equal", "skewed"): "costs unequal", ("skewed", "equal"): "means unequal",
           ("skewed", "skewed"): "both unequal"}
    rows.append("\\midrule")
    rows.append("\\multicolumn{5}{@{}l}{\\emph{Searches not finished, $P=8$, 20 failures}} \\\\")
    rows.append("& $f=0$ & $f=0.5$ & $f=1$ & max.\\ gap (\\%) \\\\")
    un = T[~cert].copy()
    lbs = np.ceil(un.lower_bound - 1e-9)
    gaps = (un.cost_exact - lbs) / lbs
    for (m, c), g in un.groupby(["means", "costs"]):
        g = g.sort_values("f")
        lb = np.ceil(g.lower_bound - 1e-9)
        cells = [f"{int(dp[(m, c)]['dp_cost'])} (DP)"] + [f"{int(b)}/{int(l)}" for b, l in zip(g.cost_exact, lb)]
        gap = ((g.cost_exact - lb) / lb).max()
        rows.append(f"\\quad {lab[(m, c)]} & " + " & ".join(cells) + f" & {pct(gap)} \\\\")
    w("tab_opt7.tex", "\n".join(rows) + "\n")
    put("seCert", str(int(cert.sum())), "ANALYTIC", src, "branch and bound plus dynamic programming at f = 0")
    put("seUnproved", str(int((~cert).sum())), "ANALYTIC", src)
    put("seCertPos", str(int(sel.sum())), "ANALYTIC", src, "certified cases with f > 0")
    put("seGapLo", pct(gaps.min()), "ANALYTIC", src, "100 (best - ceil(lower bound)) / ceil(lower bound)")
    put("seGapHi", pct(gaps.max()), "ANALYTIC", src)
    ex_ind = T.cost_indep / T.cost_exact - 1
    put("seIndCertMed", pct(ex_ind[sel].median()), "ANALYTIC", src, "certified cases with f > 0")
    put("seIndCertMax", pct(ex_ind[sel].max()), "ANALYTIC", src)
    exg = T.cost_greedy / T.cost_exact - 1
    put("seGreedyCertOpt", str(int((exg[cert].abs() < 1e-9).sum())), "ANALYTIC", src, "certified cases")
    put("seGreedyCertMax", pct(exg[cert].max()), "ANALYTIC", src)


def bridge():
    src = "analysis/bridge.py -> data/v2/bridge.json"
    B = json.load(open(os.path.join(V2, "bridge.json"), encoding="utf-8"))
    big = [r for r in B["rows"] if r["sigma"] == 0.6 and r["horizon"] == 56 and r["f"] == 1.0][0]
    du = big["ind_units"] - big["dep_units"]
    put("brBigSave", str(du), "EXACT", src, "units saved by the dependent optimum")
    put("brBigUnitPct", pct(du / big["ind_units"]), "EXACT", src, "share of the units of the independence plan")


def fleet_cert():
    src = "analysis/fleet_dep.py -> data/v2/raw/fleet_dep (counts kept in data/v2/fleet_dep_cert.json)"
    keep = os.path.join(V2, "fleet_dep_cert.json")
    files = sorted(glob.glob(os.path.join(V2, "raw", "fleet_dep", "*.csv.gz")))
    if files:                       # the raw files are not in the repository; their counts are
        D = pd.concat([pd.read_csv(f) for f in files])
        C = dict(jp=int((D.joint_cert == 0).sum() + (D.perm_cert == 0).sum()),
                 ev=int(sum((D[c] == 0).sum() for c in D.columns if c.startswith("ev_") and c.endswith("_cert"))),
                 gap=int(max((D.joint_units - D.joint_lb)[D.joint_cert == 0].max(),
                             (D.perm_units - D.perm_lb)[D.perm_cert == 0].max())),
                 n=int(len(D)))
        json.dump(C, open(keep, "w", encoding="utf-8"), indent=1)
    C = json.load(open(keep, encoding="utf-8"))
    jp, ev, gap, n = C["jp"], C["ev"], C["gap"], C["n"]
    put("fdNotCertJP", str(jp), "SIM", src, "joint and fixed-marginal plans not certified")
    put("fdNotCertEv", str(ev), "SIM", src, "incident plans not certified")
    put("fdNotCertGap", str(int(gap)), "SIM", src, "best vector minus lower bound, units")
    put("fdPlansPerType", thousands(round(n, -3)), "SIM", src, "ship-missions planned per plan type")


def sci(x, nd=1):
    m, e = f"{x:.{nd}e}".split("e")
    return f"{m}\\times 10^{{{int(e)}}}"


def cover_accuracy():
    src = "analysis/cover_accuracy.py -> data/v2/cover_accuracy.json"
    S = json.load(open(os.path.join(V2, "cover_accuracy.json"), encoding="utf-8"))["summary"]
    assert S["rechecked"] == S["optima"] == S["same"] and S["below_fine"] == 0
    put("caN", str(S["optima"]), "EXACT", src, "certified optima of Tables I, II and V")
    put("caPerSd", str(S["base"][0]), "EXACT", src, "lattice points per sd, base")
    put("caFinePerSd", str(S["fine"][0]), "EXACT", src)
    put("caFineWidth", f"{S['fine'][1]:.0f}", "EXACT", src)
    put("caErr", sci(S["err_max"]), "EXACT", src, "largest change of J on the finer lattice")
    put("caMargin", sci(S["margin_min"]), "EXACT", src, "smallest J - 0.99 of a certified optimum")
    put("caDelta", sci(S["delta"], 0).replace("1\\times ", ""), "EXACT", src, "lowered level 0.99 - delta")


def write():
    with open(os.path.join(HERE, "numbers_v7.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("% generated by make_numbers_v7.py; do not edit\n" + "\n".join(MACROS) + "\n")
    with open(os.path.join(HERE, "numbers_v7_manifest.csv"), "w", encoding="utf-8", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["macro", "value", "kind", "source", "note"])
        cw.writerows(MANIFEST)


if __name__ == "__main__":
    run_previous()
    table_one()
    bridge()
    fleet_cert()
    cover_accuracy()
    if os.path.exists(os.path.join(V2, "campaign_reg_rule_reps.csv.gz")):
        sys.path.insert(0, HERE)
        import make_numbers_v7_boot
        make_numbers_v7_boot.main(put, w, pct, thousands, V2)
    write()
    print(len(MACROS), "macros")
