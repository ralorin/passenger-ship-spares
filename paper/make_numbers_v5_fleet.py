# Executed by make_numbers_v5.py (shares its namespace): the fleet study of
# analysis/fleet_dep.py, summarised by analysis/fleet_dep_summary.py.
src = "analysis/fleet_dep.py -> data/v2/fleet_dep.json"
F = load("fleet_dep.json")
ORDER = [("clean", "Clean"), ("baseline", "Baseline"), ("heavy_tail", "Heavy tail"),
         ("misclass", "Misrecorded"), ("sparse", "Sparse"), ("datasheet_ok", "Datasheet right"),
         ("shocks", "Events 0.5/yr"), ("shocks_frequent", "Events 2/yr"),
         ("shocks_severe", "Events 2/yr, large")]


def jmark(c):
    """Cover in %, italic below the 99 % target, starred when the 95 % interval
    over fleets lies wholly below it."""
    s = pct(c[0])
    if c[0] < 0.99:
        s = f"\\textit{{{s}}}" + ("$^\\ast$" if c[2] < 0.99 else "")
    return s


def cu(p):
    return f"{jmark(p['cover'])}/{p['units'][0]:.1f}"


def sg(x, nd=2):
    """Signed, without a sign on a value that rounds to zero."""
    s = f"{x:+.{nd}f}"
    return f"{0:.{nd}f}" if float(s) == 0 else s


def dci(d, scale=100, nd=2):
    est, lo, hi = (scale * x for x in d)
    return f"${sg(est, nd)}$ & \\scriptsize$[{sg(lo, nd)},{sg(hi, nd)}]$"


rows = []
for key, label in ORDER:
    if key not in F:
        continue
    k, n = F[key]["known"], F[key]["new"]
    pk, pn = k["plans"], n["plans"]
    rows.append(f"{label} & {jmark(pk['mtbf']['cover'])} & {cu(pk['scaled'])} & {cu(pk['pooled'])} & "
                f"{cu(pk['joint'])} & {cu(pk['perm'])} & {dci(k['joint_minus_perm']['cover'])} & "
                f"${sg(k['joint_minus_perm']['units'][0])}$ & {cu(pn['joint'])} & "
                f"{dci(n['joint_minus_perm']['cover'])} & ${sg(n['joint_minus_perm']['units'][0])}$ \\\\")
w("tab_fleet5.tex", "\n".join(rows) + "\n")

put("fdFleets", str(min(F["fleets"].values())), "SIM", src, "fleets per scenario (minimum)")
put("fdFleetsMax", str(max(F["fleets"].values())), "SIM", src)
nk = [F[s]["known"]["n"] / F["fleets"][s] for s, _ in ORDER if s in F]
put("fdMissions", f"{np.mean([F['missions'][s] / F['fleets'][s] for s, _ in ORDER if s in F]):.0f}", "SIM", src,
    "predicted ship-missions per fleet")


def rng_over(fn, keys=None, fmt=lambda x: pct(x)):
    vals = [fn(F[s]) for s, _ in ORDER if s in F and (keys is None or s in keys)]
    return fmt(min(vals)), fmt(max(vals))


calm = ("clean", "baseline", "heavy_tail", "misclass", "sparse", "datasheet_ok", "shocks")
# fixed-marginal contrast
for kind, tag in (("known", "K"), ("new", "N")):
    lo, hi = rng_over(lambda e: e[kind]["joint_minus_perm"]["cover"][0], fmt=lambda x: sg(100 * x))
    put(f"fdDc{tag}Lo", f"${lo}$", "SIM", src, f"joint minus fixed-marginal cover, pp, {kind}")
    put(f"fdDc{tag}Hi", f"${hi}$", "SIM", src)
    lo, hi = rng_over(lambda e: e[kind]["joint_minus_perm"]["units"][0], fmt=lambda x: sg(x))
    put(f"fdDu{tag}Lo", f"${lo}$", "SIM", src, f"joint minus fixed-marginal units per mission, {kind}")
    put(f"fdDu{tag}Hi", f"${hi}$", "SIM", src)
    lo, hi = rng_over(lambda e: e[kind]["joint_minus_perm"]["cover"][1], fmt=lambda x: sg(100 * x))
    put(f"fdDcCi{tag}Lo", f"${lo}$", "SIM", src, "lowest lower CI limit over scenarios")
    lo, hi = rng_over(lambda e: e[kind]["joint_minus_perm"]["cover"][2], fmt=lambda x: sg(100 * x))
    put(f"fdDcCi{tag}Hi", f"${hi}$", "SIM", src, "highest upper CI limit over scenarios")
    lo, hi = rng_over(lambda e: e[kind]["same_joint_perm"])
    put(f"fdSame{tag}Lo", lo, "SIM", src, "share of missions with equal total units")
    put(f"fdSame{tag}Hi", hi, "SIM", src)
    lo, hi = rng_over(lambda e: e[kind]["units_joint_lt_perm"])
    put(f"fdFewer{tag}Hi", hi, "SIM", src, "share of missions where the joint plan has fewer units")
    lo, hi = rng_over(lambda e: e[kind]["units_joint_gt_perm"])
    put(f"fdMore{tag}Hi", hi, "SIM", src, "share of missions where the joint plan has more units")
b = F["baseline"]["known"]
put("fdBaseJoint", pct(b["plans"]["joint"]["cover"][0]), "SIM", src)
put("fdBaseJointLo", pct(b["plans"]["joint"]["cover"][1]), "SIM", src)
put("fdBaseJointHi", pct(b["plans"]["joint"]["cover"][2]), "SIM", src)
# learning the rate
lo, hi = rng_over(lambda e: e["known"]["plans"]["mtbf"]["cover"][0], keys=[s for s in calm if s != "datasheet_ok"])
put("fdMtbfLo", lo, "SIM", src)
put("fdMtbfHi", hi, "SIM", src)
put("fdMtbfRight", pct(F["datasheet_ok"]["known"]["plans"]["mtbf"]["cover"][0]), "SIM", src)
lo, hi = rng_over(lambda e: e["known"]["plans"]["scaled"]["cover"][0], keys=calm)
put("fdScaledLo", lo, "SIM", src)
put("fdScaledHi", hi, "SIM", src)
for kind, tag in (("known", "K"), ("new", "N")):
    lo, hi = rng_over(lambda e: min(e[kind]["plans"]["pooled"]["cover"][0], e[kind]["plans"]["joint"]["cover"][0],
                                    e[kind]["plans"]["perm"]["cover"][0]), keys=calm)
    put(f"fdFit{tag}Lo", lo, "SIM", src, "lowest of pooled, joint, fixed-marginal, calm scenarios")
    lo, hi = rng_over(lambda e: max(e[kind]["plans"]["pooled"]["cover"][0], e[kind]["plans"]["joint"]["cover"][0],
                                    e[kind]["plans"]["perm"]["cover"][0]), keys=calm)
    put(f"fdFit{tag}Hi", hi, "SIM", src)
jp = b["joint_minus_pooled"]
put("fdJpCov", sgn(100 * jp["cover"][0]), "SIM", src, "baseline known, joint minus pooled NB, pp")
put("fdJpCovLo", sgn(100 * jp["cover"][1]), "SIM", src)
put("fdJpCovHi", sgn(100 * jp["cover"][2]), "SIM", src)
put("fdJpUn", sgn(jp["units"][0]), "SIM", src)
put("fdJpUnLo", sgn(jp["units"][1]), "SIM", src)
put("fdJpUnHi", sgn(jp["units"][2]), "SIM", src)
ps = b["pooled_minus_scaled"]
put("fdPsCov", sgn(100 * ps["cover"][0]), "SIM", src, "baseline known, pooled minus scaled, pp")
put("fdPsUn", sgn(ps["units"][0]), "SIM", src)
jn = F["baseline"]["new"]["joint_minus_pooled"]
put("fdJpNewCov", sgn(100 * jn["cover"][0]), "SIM", src)
put("fdJpNewUn", sgn(jn["units"][0]), "SIM", src)
put("fdMargLo", rng_over(lambda e: e["known"]["plans"]["marg"]["cover"][0], keys=calm)[0], "SIM", src)
put("fdMargHi", rng_over(lambda e: e["known"]["plans"]["marg"]["cover"][0], keys=calm)[1], "SIM", src)
# certification, heuristic gap, residual dependence
allk = [F[s][k] for s, _ in ORDER if s in F for k in ("known", "new")]
put("fdCert", pct(min(min(e["cert_joint"], e["cert_perm"]) for e in allk), 2), "SIM", src, "share of plans certified")
put("fdHeurGap", f"{max(e['heur_gap_joint'] for e in allk):.2f}", "SIM", src, "heuristic minus certified units, max over groups")
put("fdHeurWorse", pct(max(e["heur_worse_joint"] for e in allk), 0), "SIM", src)
put("fdResid", f"{max(e['perm_resid_max_abs'] for e in allk) * 100:.2f}", "SIM", src,
    "max |J_perm - prod of marginals| at the plan, pp")
put("fdResidMean", f"{max(e['perm_resid_mean_abs'] for e in allk) * 100:.3f}", "SIM", src)
put("fdPermSeedSame", pct(min(e["perm_seed_same"] for e in allk), 0), "SIM", src)
put("fdPermJoint", pct(min(e["perm_Jjoint_mean"] for e in allk), 2), "SIM", src,
    "mean cover of the fixed-marginal plan under the joint draws, min over groups")
put("fdJointProd", pct(min(e["joint_Jprod_mean"] for e in allk), 2), "SIM", src,
    "mean product-of-marginals cover of the joint plan, min over groups")
# Monte Carlo selection
st = [F[s]["known"]["stab"] for s, _ in ORDER if s in F and "stab" in F[s]["known"]]
if st:
    put("fdStabFleets", str(min(x["fleets"] for x in st)), "SIM", src)
    put("fdStabSameSeed", pct(min(x["same_seed"] for x in st), 0), "SIM", src, "identical vector with a new seed")
    put("fdStabSameBig", pct(min(x["same_big"] for x in st), 0), "SIM", src, "identical vector at 24000 draws")
    put("fdStabAbs", f"{max(x['absdiff_big'] for x in st):.2f}", "SIM", src)
    put("fdStabFresh", pct(min(x["Jfresh_joint"] for x in st), 2), "SIM", src,
        "cover of the 6000-draw plan under 24000 fresh draws, mean, min over scenarios")
    vals = [100 * x[c]["cover"][0] for x in st for c in ("contrast_r6000", "contrast_seed2", "contrast_r24000")]
    put("fdStabDcLo", f"${min(vals):+.2f}$", "SIM", src)
    put("fdStabDcHi", f"${max(vals):+.2f}$", "SIM", src)
    vals = [x[c]["units"][0] for x in st for c in ("contrast_r6000", "contrast_seed2", "contrast_r24000")]
    put("fdStabDuLo", f"${min(vals):+.2f}$", "SIM", src)
    put("fdStabDuHi", f"${max(vals):+.2f}$", "SIM", src)

# incident plans
EVR = [("joint", "no event term"), ("ev_known", "known, idealised"), ("ev_rate_half", "rate $\\times0.5$"),
       ("ev_rate_double", "rate $\\times2$"), ("ev_size_half", "size $\\times0.5$"), ("ev_size_double", "size $\\times2$")]
rows = []
for key, label in (("shocks_frequent", "2 a year, 0.05\\,\\%"), ("shocks_severe", "2 a year, 0.2\\,\\%")):
    if key not in F or "events" not in F[key]["known"]:
        continue
    E = F[key]["known"]["events"]
    rows.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{Events {label}: {E['event']['n']} of {E['all']['n']} "
                f"missions}}}} \\\\")
    base_units = E["all"]["plans"]["joint"]["units"][0]
    for k, lab in EVR:
        a, e, ne = E["all"]["plans"][k], E["event"]["plans"][k], E["no_event"]["plans"][k]
        rows.append(f"\\quad {lab} & {pct(a['cover'][0])} & {pct(e['cover'][0], 0)} & {pct(ne['cover'][0])} & "
                    f"{a['units'][0]:.1f} & ${a['units'][0] - base_units:+.1f}$ & {e['short'][0]:.1f} \\\\")
w("tab_ev.tex", "\n".join(rows) + "\n")
for key, tag in (("shocks_frequent", "Freq"), ("shocks_severe", "Sev"), ("baseline", "Base"), ("shocks", "Half")):
    if key not in F or "events" not in F[key]["known"]:
        continue
    E = F[key]["known"]["events"]
    put(f"fdEv{tag}N", str(E["event"]["n"]), "SIM", src, "event missions, known ships")
    put(f"fdEv{tag}All", str(E["all"]["n"]), "SIM", src)
    for k, t2 in (("joint", "Omit"), ("ev_known", "Known")):
        put(f"fdEv{tag}{t2}Ev", pct(E["event"]["plans"][k]["cover"][0], 0), "SIM", src)
        ci = E["event"]["plans"][k]["cover"]
        put(f"fdEv{tag}{t2}EvLo", pct(ci[1], 0), "SIM", src)
        put(f"fdEv{tag}{t2}EvHi", pct(ci[2], 0), "SIM", src)
        put(f"fdEv{tag}{t2}All", pct(E["all"]["plans"][k]["cover"][0]), "SIM", src)
        put(f"fdEv{tag}{t2}AllLo", pct(E["all"]["plans"][k]["cover"][1]), "SIM", src)
        put(f"fdEv{tag}{t2}AllHi", pct(E["all"]["plans"][k]["cover"][2]), "SIM", src)
        put(f"fdEv{tag}{t2}Units", f"{E['all']['plans'][k]['units'][0]:.1f}", "SIM", src)
        put(f"fdEv{tag}{t2}Short", f"{E['event']['plans'][k]['short'][0]:.1f}", "SIM", src)
    put(f"fdEv{tag}Extra", pct(E["all"]["plans"]["ev_known"]["units"][0] / E["all"]["plans"]["joint"]["units"][0] - 1, 0),
        "SIM", src, "extra units of the known-event plan, %")
    put(f"fdEv{tag}ExtraCost", pct(E["all"]["plans"]["ev_known"]["cost"] / E["all"]["plans"]["joint"]["cost"] - 1, 0),
        "SIM", src, "extra cost of the known-event plan, %")
    if "new" in F[key] and "events" in F[key]["new"]:
        En = F[key]["new"]["events"]
        put(f"fdEv{tag}NewN", str(En["event"]["n"]) if "event" in En else "0", "SIM", src)
        if "event" in En:
            put(f"fdEv{tag}NewKnownEv", pct(En["event"]["plans"]["ev_known"]["cover"][0], 0), "SIM", src)
            put(f"fdEv{tag}NewOmitEv", pct(En["event"]["plans"]["joint"]["cover"][0], 0), "SIM", src)
        put(f"fdEv{tag}NewKnownAll", pct(En["all"]["plans"]["ev_known"]["cover"][0]), "SIM", src)
put("fdAbsDcMax", f"{max(abs(100 * F[s][k]['joint_minus_perm']['cover'][0]) for s, _ in ORDER if s in F for k in ('known', 'new')):.2f}",
    "SIM", src, "largest |joint - fixed-marginal| cover difference over scenarios and ship kinds, pp")
put("fdAbsDuMax", f"{max(max(abs(x) for x in F[s][k]['joint_minus_perm']['units']) for s, _ in ORDER if s in F for k in ('known', 'new')):.2f}",
    "SIM", src)
