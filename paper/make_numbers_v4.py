"""
Numbers added in the revision of 2026-11-04: shipboard cases, calibration and
convergence of the bootstrap, and the audit of the street-lighting inputs.
Writes numbers_v4.tex (loaded after numbers_v3.tex, so a \\renewcommand here
corrects an earlier value), tables/tab_ship.tex, tables/tab_ni4.tex and
numbers_v4_manifest.csv.

    python make_numbers_v4.py
"""
import csv
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, "..", "data", "v2")
TAB = os.path.join(HERE, "tables")
OLD = set()
for name in ("numbers.tex", "numbers_v2.tex", "numbers_v3.tex"):
    for line in open(os.path.join(HERE, name), encoding="utf-8"):
        if line.startswith("\\newcommand{"):
            OLD.add(line.split("{")[1].rstrip("}"))
MACROS, MANIFEST = [], []


def put(name, value, kind, src, note=""):
    cmd = "\\renewcommand" if "\\" + name in OLD else "\\newcommand"
    MACROS.append(f"{cmd}{{\\{name}}}{{{value}}}")
    MANIFEST.append((name, value, kind, src, note))


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def load(name):
    with open(os.path.join(V2, name), encoding="utf-8") as f:
        return json.load(f)


def w(name, text):
    with open(os.path.join(TAB, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


# ------------------------------------------------------------ shipboard cases --
src = "analysis/ship_cases.py -> data/v2/ship_cases.json"
S = load("ship_cases.json")
N = dict(zip(S["products"], S["installed"]))
for key, tag in (("cabin_led_driver", "Cabin"), ("cabin_usb_psu", "Usb"), ("fancoil_ec_drive", "Fan"),
                 ("public_led_driver", "Public"), ("galley_smps", "Galley"),
                 ("exterior_led_driver", "Ext"), ("technical_vfd", "Vfd")):
    put(f"sh{tag}", f"{N[key]:,}".replace(",", "\\,"), "ANALYTIC", src, "installed units, large class")
C = S["cases"]
assert all(c["proved"] for c in C)
put("shLamWeek", f"{C[0]['total']:.2f}", "ANALYTIC", src, "7 days")
rows = []
for c in C:
    lab = f"{c['days']} d, {c['sigma']:.1f}"
    rows.append(f"{lab} & {c['total']:.2f} & {pct(c['marg_cover_f0'])} & {c['robust_units']} & "
                f"{pct(c['premium_vs_f1'])} & & {pct(c['robust_cover_exposure125'])} & "
                f"{pct(c['robust_cover_given_event'])} & {pct(c['robust_cover_with_events'])} \\\\")
w("tab_ship.tex", "\n".join(rows) + "\n")


def rng(key, f=pct):
    v = [c[key] for c in C]
    return f(min(v)), f(max(v))


put("shMargLo", rng("marg_cover_f0")[0], "ANALYTIC", src)
put("shMargHi", rng("marg_cover_f0")[1], "ANALYTIC", src)
put("shPremMax", rng("premium_vs_f1")[1], "ANALYTIC", src, "robust vs optimum at f = 1")
put("shExpLo", rng("robust_cover_exposure125")[0], "ANALYTIC", src)
put("shExpHi", rng("robust_cover_exposure125")[1], "ANALYTIC", src)
put("shEvLo", rng("robust_cover_given_event")[0], "ANALYTIC", src)
put("shEvHi", rng("robust_cover_given_event")[1], "ANALYTIC", src)
put("shEvAllLo", rng("robust_cover_with_events")[0], "ANALYTIC", src)
put("shEvAllHi", rng("robust_cover_with_events")[1], "ANALYTIC", src)
ev = [c["event_plan_cost"] / c["robust_cost"] - 1 for c in C]
big = [c["big_event_plan_cost"] / c["robust_cost"] - 1 for c in C]
put("shEvCostLo", pct(min(ev), 0), "ANALYTIC", src)
put("shEvCostHi", pct(max(ev), 0), "ANALYTIC", src)
put("shBigCostLo", pct(min(big), 0), "ANALYTIC", src)
put("shBigCostHi", pct(max(big), 0), "ANALYTIC", src)

# ------------------------------------------------- bootstrap calibration --
src = "data/v2/campaign_boot_reps.csv.gz (analysis/campaign_boot.py)"
import sys
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
from campaign_boot import cells as boot_cells          # noqa: E402

cl = pd.DataFrame(boot_cells())[["cell", "S", "M", "m", "tau", "omega"]]
R = pd.read_csv(os.path.join(V2, "campaign_boot_reps.csv.gz")).merge(cl, on="cell")
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

# ------------------------------------------------- bootstrap convergence --
src = "analysis/campaign_bconv.py -> data/v2/campaign_bconv.json"
B = load("campaign_bconv.json")
put("bcCells", {3: "three", 4: "four", 5: "five"}.get(len(B), str(len(B))), "SIM", src)
put("bcDiffMax", f"{100 * max(abs(v['d20_80_mean']) for v in B.values()):.2f}", "SIM", src)
put("bcMcse", f"{100 * max(v['d20_80_mcse'] for v in B.values()):.2f}", "SIM", src)
put("bcRedraw", f"{100 * np.median([v['redraw_absmed'] for v in B.values()]):.2f}", "SIM", src)
put("bcRedrawMax", f"{100 * max(v['redraw_absmax'] for v in B.values()):.1f}", "SIM", src)
put("bcUnitsMax", str(max(int(v["redraw_units_absmax"]) for v in B.values())), "SIM", src)
Rc = pd.read_csv(os.path.join(V2, "campaign_bconv_reps.csv"))
du = (Rc.units_b80_again - Rc.units_b80).abs()
put("bcRedrawTen", str(int((du > 10).sum())), "SIM", "data/v2/campaign_bconv_reps.csv", "plan changed by > 10 units")
put("bcN", str(len(Rc)), "SIM", "data/v2/campaign_bconv_reps.csv")
put("bcRedrawUnitsMed", f"{du.median():.0f}", "SIM", "data/v2/campaign_bconv_reps.csv")
src = "data/v2/campaign_boot_reps.csv.gz"
small = R[(R.S == 5) & (R.M == 1)]
big = small.joint_units_boot > 5 * small.joint_units_oracle
put("cbBigN", str(int(big.sum())), "SIM", src, "S=5, M=1: bootstrap plan above 5x the oracle units")
put("cbBigTot", str(len(small)), "SIM", src)
mx = int(small.joint_units_boot.max())
e = len(str(mx)) - 1
put("cbBigMax", f"{mx / 10 ** e:.0f}\\times 10^{{{e}}}", "SIM", src, str(mx))
put("cbBigSd", f"{small[big][['tau_b_q90', 'omega_b_q90']].max(axis=1).median():.1f}", "SIM", src,
    "median of the larger 90th bootstrap percentile of the sds in those campaigns")

# ---------------------------------------------------- street lighting audit --
src = "analysis/ni_audit.py -> data/v2/ni_audit.json"
A = load("ni_audit.json")
put("niSkipped", str(A["malformed"]), "PUBLICDATA", src, "records with a wrong field count")
put("niNearShare", pct(A["area_nearest_share"], 0), "PUBLICDATA", src)
put("niHoursPerReport", f"{A['hours_per_report'] / 1e6:.2f}", "PUBLICDATA", src)
fam = A["hours_per_report_family"].values()
put("niHoursFamLo", f"{min(fam) / 1e6:.2f}", "PUBLICDATA", src)
put("niHoursFamHi", f"{max(fam) / 1e6:.2f}", "PUBLICDATA", src)
mtbf_ext = json.load(open(os.path.join(HERE, "..", "analysis", "fleet_params.json"),
                          encoding="utf-8"))["products"]["exterior_led_driver"]["mtbf_h"]
put("niDsRatio", f"{mtbf_ext / A['hours_per_report']:.0f}", "PUBLICDATA", src,
    "datasheet MTBF of the exterior driver over hours per report")

lines = open(os.path.join(TAB, "tab_ni3.tex"), encoding="utf-8").read().splitlines()
keep = []
for ln in lines:
    if ln.split("&")[0].strip() in ("Poisson", "Hier., + month", "Gamma EB", "Hier., plug-in"):
        continue
    keep.append(ln.replace("$\\theta$-integr.", "bootstrap"))
w("tab_ni4.tex", "\n".join(keep) + "\n")
cells_ = {ln.split("&")[0].strip(): [x.strip() for x in ln.split("&")] for ln in keep}
put("niTabPredNew", cells_["Hier., predictive"][5], "PUBLICDATA", "tables/tab_ni3.tex")
put("niTabNbNew", cells_["Pooled NB"][5], "PUBLICDATA", "tables/tab_ni3.tex")

with open(os.path.join(HERE, "numbers_v4.tex"), "w", encoding="utf-8", newline="\n") as f:
    f.write("% generated by make_numbers_v4.py\n" + "\n".join(MACROS) + "\n")
with open(os.path.join(HERE, "numbers_v4_manifest.csv"), "w", encoding="utf-8", newline="") as f:
    cw = csv.writer(f)
    cw.writerow(["macro", "value", "kind", "source", "note"])
    cw.writerows(MANIFEST)
print(len(MACROS), "macros")
