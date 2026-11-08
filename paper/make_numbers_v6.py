"""
The bridge experiment (analysis/bridge.py), the per-configuration premiums of
the large ship, the profile intervals inverted at the mixture point, and the
street-lighting table with new areas in both retrospective tests. Runs
make_numbers_v5.py first; numbers_v6.tex is loaded after numbers_v5.tex.

    python make_numbers_v6.py
"""
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, "..", "data", "v2")
TAB = os.path.join(HERE, "tables")

subprocess.run([sys.executable, os.path.join(HERE, "make_numbers_v5.py")], check=True, cwd=HERE)

OLD = set()
for name in ("numbers.tex", "numbers_v2.tex", "numbers_v3.tex", "numbers_v4.tex", "numbers_v5.tex"):
    for line in open(os.path.join(HERE, name), encoding="utf-8"):
        if line.startswith(("\\newcommand{", "\\renewcommand{")):
            OLD.add(line.split("{")[1].rstrip("}"))
MACROS, MANIFEST, SEEN = [], [], set()


def put(name, value, kind, src, note=""):
    assert name not in SEEN, name
    SEEN.add(name)
    cmd = "\\renewcommand" if "\\" + name in OLD else "\\newcommand"
    MACROS.append(f"{cmd}{{\\{name}}}{{{value}}}")
    MANIFEST.append((name, value, kind, src, note))


def load(name):
    return json.load(open(os.path.join(V2, name), encoding="utf-8"))


def w(name, text):
    with open(os.path.join(TAB, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def sg(x, d=2):
    v = round(x, d)
    if v == 0:
        return f"${0:.{d}f}$"
    return f"${v:+.{d}f}$"


# large ship
src = "analysis/ship_cases.py -> data/v2/ship_cases.json"
S = load("ship_cases.json")
C = {(c["days"], c["sigma"]): c for c in S["cases"]}
for (d, sgm), tag in (((7, 0.6), "SevenMid"), ((14, 0.6), "Fourteen"), ((28, 0.6), "TwentyEight"),
                      ((7, 0.3), "SevenLow"), ((7, 0.9), "SevenHigh")):
    put(f"shPrem{tag}", f"{100 * C[(d, sgm)]['premium_vs_f1']:.1f}", "ANALYTIC", src,
        f"robust over optimum at f = 1, {d} d, sigma {sgm}")
put("shUnitsTwo", str(C[(14, 0.6)]["robust_units"] - C[(7, 0.6)]["robust_units"]), "ANALYTIC", src)
put("shUnitsFour", str(C[(28, 0.6)]["robust_units"] - C[(7, 0.6)]["robust_units"]), "ANALYTIC", src)
diff = max(sum(a != b for a, b in zip(c["robust"], c["know_f1"])) for c in S["cases"])
put("shDiffRefs", str(diff), "ANALYTIC", src, "references where robust and f = 1 optimum differ")
dmax = max(max(abs(a - b) for a, b in zip(c["robust"], c["know_f1"])) for c in S["cases"])
assert dmax == 1

# bridge
src = "analysis/bridge.py -> data/v2/bridge.json"
B = load("bridge.json")
R = {(r["sigma"], r["horizon"], r["f"]): r for r in B["rows"]}
assert all(r["proved"] for r in B["rows"])
put("brSeeds", str(B["rows"][0]["seeds"]), "SIM", src)
put("brDraws", f"{B['rows'][0]['R']:,}".replace(",", "\\,"), "SIM", src)
put("brCert", f"{100 * min(r['cert'] for r in B['rows']):.0f}", "SIM", src)
rows = []
for sgm in B["sigmas"]:
    rows.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{$\\sigma={sgm}$}}}} \\\\")
    for H in (7, 14, 28, 56):
        r5, r1 = R[(sgm, H, 0.5)], R[(sgm, H, 1.0)]
        ex5 = r5["dep_units"] - r5["ind_units"]
        ex1 = r1["dep_units"] - r1["ind_units"]
        hw = (r1["du_ci"][1] - r1["du_ci"][0]) / 2
        hwj = 100 * (r1["dJ_ci"][1] - r1["dJ_ci"][0]) / 2
        rows.append(f"\\quad {H} & {r1['Lambda']:.1f} & {r1['ind_units']} & {sg(ex5, 0)} & {sg(ex1, 0)} & "
                    f"{sg(r5['du'])} & {sg(r1['du'])}\\,$\\pm${hw:.2f} & {sg(100 * r1['dJ'])}\\,$\\pm${hwj:.2f} \\\\")
w("tab_bridge.tex", "\n".join(rows) + "\n")
f0 = [R[k] for k in R if k[2] == 0.0]
put("brNoiseU", f"{max(max(abs(x) for x in r['du_ci']) for r in f0):.2f}", "SIM", src,
    "f = 0: largest |end| of the 95 pct interval of the unit contrast")
put("brNoiseJ", f"{100 * max(max(abs(x) for x in r['dJ_ci']) for r in f0):.2f}", "SIM", src)
put("brResid", f"{100 * max(r['resid_max'] for r in B['rows']):.2f}", "SIM", src,
    "max over seeds and cases of permuted-draw cover minus product cover at the chosen vector")
sel = [r for r in B["rows"] if r["f"] > 0]
put("brJLo", f"{100 * min(min(r['Jj'], r['Jp']) for r in B['rows']):.2f}", "SIM", src, "mean exact cover of draw plans")
put("brJHi", f"{100 * max(max(r['Jj'], r['Jp']) for r in B['rows']):.2f}", "SIM", src)
fl = R[(0.42, 7, 1.0)]
put("brFleetDu", sg(fl["du"]), "SIM", src, "sigma 0.42, 7 d, f = 1")
put("brFleetDJ", sg(100 * fl["dJ"]), "SIM", src)
big = R[(0.6, 56, 1.0)]
put("brBigDep", str(big["dep_units"]), "SIM", src)
put("brBigInd", str(big["ind_units"]), "SIM", src)
put("brBigDu", sg(big["du"]), "SIM", src)
put("brBigCostPct", f"{100 * (big['ind_cost'] / big['dep_cost'] - 1):.1f}", "SIM", src,
    "cost of independence over dependent optimum, relative costs")
put("brBigJInd", f"{100 * big['J_ind']:.2f}", "SIM", src)

# profile CIs
src = "analysis/campaign_extra.py -> data/v2/campaign_extra.json"
P = load("campaign_extra.json")["profile"]
put("ceCiCorrLo", f"{100 * P['corrected_lo']:.0f}", "SIM", src, "zero inside when LRT < 2.71")
put("ceCiCorrHi", f"{100 * P['corrected_hi']:.0f}", "SIM", src)
put("ceCiZeroCorr", f"{100 * P['zero_corrected']:.1f}", "SIM", src)

# street lighting
src = "analysis/ni_validation.py -> data/v2/ni_validation_evidence.json, ni_validation_boot.json"
E = load("ni_validation_evidence.json")
BT = load("ni_validation_boot.json")["within_plans"]["cover"]
W, F = E["within"]["cover"], E["forward"]["cover"]


def cell(C_, key):
    v = C_.get(key)
    return "-- & --" if v is None else f"{v['exceed']} & {v['units']:.0f}"


rows = []
for key, label in (("poisson_jnt", "Poisson, joint"), ("nb_pooled", "Pooled NB"),
                   ("hier_pred", "Hier., predictive"), ("hier_boot", "Hier., bootstrap")):
    Wc = BT if key == "hier_boot" else W
    rows.append(f"{label} & {cell(Wc, key + '|known')} & {cell(Wc, key + '|new')} & "
                f"{cell(F, key + '|known')} & {cell(F, key + '|new')} \\\\")
w("tab_ni6.tex", "\n".join(rows) + "\n")
put("niFwdNewN", str(F["hier_pred|new"]["n"]), "PUBLICDATA", src)
put("niFwdPredNew", f"{F['hier_pred|new']['units']:.0f}", "PUBLICDATA", src)
put("niFwdNbNew", f"{F['nb_pooled|new']['units']:.0f}", "PUBLICDATA", src)
put("niLsWithinN", str(E["within"]["log_scores"]["hier_pred|known"]["n"]), "PUBLICDATA", src,
    "family-months per group, temporal split")
put("niLsFwdN", str(E["forward"]["log_scores"]["hier_pred|new"]["n"]), "PUBLICDATA", src,
    "family-months per group, later period")
d = E["forward"]["log_score_diffs"]["hier_pred-nb_pooled|known"]
put("niLsFwdKnown", sg(d["d"], 3), "PUBLICDATA", src)
put("niLsFwdKnownLo", sg(d["ci95"][0], 3), "PUBLICDATA", src)
put("niLsFwdKnownHi", sg(d["ci95"][1], 3), "PUBLICDATA", src)
put("niBootRunPredNew", f"{BT['hier_pred|new']['units']:.0f}", "PUBLICDATA", src, "bootstrap run, its own stream")

with open(os.path.join(HERE, "numbers_v6.tex"), "w", encoding="utf-8", newline="\n") as f:
    f.write("% generated by make_numbers_v6.py\n" + "\n".join(MACROS) + "\n")
with open(os.path.join(HERE, "numbers_v6_manifest.csv"), "w", encoding="utf-8", newline="") as f:
    cw = csv.writer(f)
    cw.writerow(["macro", "value", "kind", "source", "note"])
    cw.writerows(MANIFEST)
print(len(MACROS), "macros")
