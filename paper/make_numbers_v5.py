"""
Certified and interrupted cases of the grid, the per-reference plans of the
large ship, the fixed-marginal contrast and the incident plans of the fleet
(make_numbers_v5_fleet.py), regularised small campaigns
(make_numbers_v5_reg.py) and the seed repetitions of the street-lighting plans.
Writes numbers_v5.tex (loaded after numbers_v4.tex; a \\renewcommand
overrides an earlier value), the tables it names and numbers_v5_manifest.csv.

    python make_numbers_v5.py
"""
import csv
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, "..", "data", "v2")
TAB = os.path.join(HERE, "tables")
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
OLD = set()
for name in ("numbers.tex", "numbers_v2.tex", "numbers_v3.tex", "numbers_v4.tex"):
    for line in open(os.path.join(HERE, name), encoding="utf-8"):
        if line.startswith("\\newcommand{"):
            OLD.add(line.split("{")[1].rstrip("}"))
MACROS, MANIFEST = [], []
SEEN = set()


def put(name, value, kind, src, note=""):
    assert name not in SEEN, name
    SEEN.add(name)
    cmd = "\\renewcommand" if "\\" + name in OLD else "\\newcommand"
    MACROS.append(f"{cmd}{{\\{name}}}{{{value}}}")
    MANIFEST.append((name, value, kind, src, note))


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def sgn(x, nd=2):
    return f"${x:+.{nd}f}$"


def load(name):
    with open(os.path.join(V2, name), encoding="utf-8") as f:
        return json.load(f)


def w(name, text):
    with open(os.path.join(TAB, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def have(name):
    return os.path.exists(os.path.join(V2, name))


rng_boot = np.random.default_rng(20261105)


def cluster_ci(values_by_cluster, B=4000):
    """Mean over clusters and a percentile interval from resampling clusters."""
    x = np.asarray(values_by_cluster, float)
    idx = rng_boot.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


# certified grid cases
src = "analysis/stock_exact.py -> data/v2/stock_exact_part1.csv"
T = pd.read_csv(os.path.join(V2, "stock_exact_part1.csv"))
uneq = (T.means == "skewed") | (T.costs == "skewed")
cert = T.proved.astype(bool)
rows = []
for key, label in (("common", "Common stock"), ("equal_risk", "Equal risk"),
                   ("indep", "Independence ($f=0$)"), ("greedy", "Marginal analysis")):
    ex = T[f"cost_{key}"] / T.cost_exact - 1
    sel = cert & (T.f > 0)
    rows.append(f"{label} & {pct(ex[uneq & sel].median())} & {pct(ex[~uneq & sel].median())} & "
                f"{pct(ex[cert].max())} & {pct(T[f'cover_{key}'].min())} \\\\")
un = T[~cert].copy()
lab = {("equal", "skewed"): "costs unequal", ("skewed", "equal"): "means unequal",
       ("skewed", "skewed"): "both unequal"}
rows.append("\\midrule")
rows.append("\\multicolumn{5}{@{}l}{\\emph{Interrupted searches, $P=8$, 20 failures: best cost / lower bound}} \\\\")
rows.append("& $f=0$ & $f=0.5$ & $f=1$ & max.\\ gap (\\%) \\\\")
for (m, c), g in un.groupby(["means", "costs"]):
    g = g.sort_values("f")
    lb = np.ceil(g.lower_bound - 1e-9)
    cells = [f"{int(b)}/{int(l)}" for b, l in zip(g.cost_exact, lb)]
    gap = ((g.cost_exact - lb) / g.cost_exact).max()
    rows.append(f"\\quad {lab[(m, c)]} & " + " & ".join(cells) + f" & {pct(gap)} \\\\")
w("tab_opt5.tex", "\n".join(rows) + "\n")
lb = np.ceil(un.lower_bound - 1e-9)
gap = (un.cost_exact - lb) / un.cost_exact
put("seCert", str(int(cert.sum())), "ANALYTIC", src)
put("seGapLo", pct(gap.min()), "ANALYTIC", src, "(best - ceil(lower bound)) / best, interrupted cases")
put("seGapHi", pct(gap.max()), "ANALYTIC", src)
ex_ind = T.cost_indep / T.cost_exact - 1
sel = cert & (T.f > 0)
put("seIndCertMed", pct(ex_ind[sel].median()), "ANALYTIC", src, "certified cases with f > 0")
put("seIndCertMax", pct(ex_ind[sel].max()), "ANALYTIC", src)
exg = T.cost_greedy / T.cost_exact - 1
put("seGreedyCertOpt", str(int((exg[cert].abs() < 1e-9).sum())), "ANALYTIC", src, "certified cases")
put("seGreedyCertMax", pct(exg[cert].max()), "ANALYTIC", src)

# the large ship
src = "analysis/ship_cases.py -> data/v2/ship_cases.json"
S = load("ship_cases.json")
C = S["cases"]
short = {"cabin_led_driver": "Cabin LED driver", "cabin_usb_psu": "Cabin USB supply",
         "fancoil_ec_drive": "Fan-coil drive", "public_led_driver": "Public-area driver",
         "galley_smps": "Galley supply", "exterior_led_driver": "Exterior driver",
         "technical_vfd": "Technical drive"}
rows = []
for j, p in enumerate(S["products"]):
    N = f"{S['installed'][j]:,}".replace(",", "\\,")
    lam = 7 * S["failures_per_day"][j]
    rows.append(f"{short[p]} & {N} & {S['cost'][j]:g} & {lam:.2f} & "
                + " & ".join(str(c["robust"][j]) for c in C) + " \\\\")
rows.append("\\midrule")


def line(label, vals):
    rows.append(f"\\multicolumn{{4}}{{@{{}}l}}{{{label}}} & " + " & ".join(vals) + " \\\\")


line("$\\Lambda$, expected failures", [f"{c['total']:.2f}" for c in C])
line("Units of robust plan", [str(c["robust_units"]) for c in C])
line("Cost of robust plan", [f"{c['robust_cost']:.1f}" for c in C])
line("Cost of optimum at $f=1$", [f"{c['know_f1_cost']:.1f}" for c in C])
line("Premium (\\%)", [pct(c["premium_vs_f1"]) for c in C])
line("$J$, hours $\\times1.25$", [pct(c["robust_cover_exposure125"]) for c in C])
line("$J$ given an event", [pct(c["robust_cover_given_event"]) for c in C])
line("$J$, events twice a year", [pct(c["robust_cover_with_events"]) for c in C])
line("Cost, event in law (\\%)", [f"$+{pct(c['event_plan_cost'] / c['robust_cost'] - 1, 0)}$" for c in C])
w("tab_ship5.tex", "\n".join(rows) + "\n")
put("shKnowCostLo", f"{min(c['know_f1_cost'] for c in C):.1f}", "ANALYTIC", src)

# power, from the generated table
src = "paper/tables/tab_power.tex (make_numbers_v2.py, campaign study)"
PW = [[int(x) for x in l.replace("\\\\", "").split("&")] for l in open(os.path.join(TAB, "tab_power.tex")) if "&" in l]
PW = np.array(PW)          # columns: S, tau M=1,2,4,8, omega M=1,2,4,8
for j, M in ((2, "Two"), (3, "Four"), (4, "Eight")):
    put(f"pwTau{M}Lo", str(PW[:, j].min()), "SIM", src)
    put(f"pwTau{M}Hi", str(PW[:, j].max()), "SIM", src)
    put(f"pwOm{M}Lo", str(PW[:, j + 4].min()), "SIM", src)
    put(f"pwOm{M}Hi", str(PW[:, j + 4].max()), "SIM", src)

# street lighting seeds
src = "analysis/ni_seeds.py -> data/v2/ni_seeds.json"
NI = load("ni_seeds.json")
pn = NI["hier_pred|new"]["units"]
put("niSeedN", str(len(pn)), "PUBLICDATA", src)
put("niSeedPredNewMean", f"{np.mean(pn):.0f}", "PUBLICDATA", src, "predictive plan, new area, units per area-month")
put("niSeedPredNewLo", f"{min(pn):.0f}", "PUBLICDATA", src)
put("niSeedPredNewHi", f"{max(pn):.0f}", "PUBLICDATA", src)
dk = NI["diff|known"]
put("niSeedKnownLo", f"{-max(dk):.0f}", "PUBLICDATA", src, "pooled minus predictive, known areas")
put("niSeedKnownHi", f"{-min(dk):.0f}", "PUBLICDATA", src)
lo_, hi_ = f"{-max(dk):.0f}", f"{-min(dk):.0f}"
put("niSeedKnownRange", lo_ if lo_ == hi_ else f"{lo_}--{hi_}", "PUBLICDATA", src)
dn = NI["diff|new"]
put("niSeedNewLo", f"{min(dn):.0f}", "PUBLICDATA", src, "predictive minus pooled, new areas")
put("niSeedNewHi", f"{max(dn):.0f}", "PUBLICDATA", src)
ex = NI["hier_pred|new"]["exceed"] + NI["hier_pred|known"]["exceed"]
put("niSeedPredExcMax", str(max(ex)), "PUBLICDATA", src)
en = NI["nb_pooled|new"]["exceed"]
put("niSeedNbNewExcLo", str(min(en)), "PUBLICDATA", src)
put("niSeedNbNewExcHi", str(max(en)), "PUBLICDATA", src)

# fleet: fixed-marginal contrast
if have("fleet_dep.json"):
    exec(open(os.path.join(HERE, "make_numbers_v5_fleet.py"), encoding="utf-8").read())

# regularised campaigns
if have("campaign_reg_reps.csv.gz"):
    exec(open(os.path.join(HERE, "make_numbers_v5_reg.py"), encoding="utf-8").read())

# write out
with open(os.path.join(HERE, "numbers_v5.tex"), "w", encoding="utf-8", newline="\n") as f:
    f.write("% generated by make_numbers_v5.py; do not edit\n" + "\n".join(MACROS) + "\n")
with open(os.path.join(HERE, "numbers_v5_manifest.csv"), "w", encoding="utf-8", newline="") as f:
    cw = csv.writer(f)
    cw.writerow(["macro", "value", "kind", "source", "note"])
    cw.writerows(MANIFEST)
print(len(MACROS), "macros")
