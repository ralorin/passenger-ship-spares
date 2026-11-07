# Executed by make_numbers_v5.py (shares its namespace): small campaigns with
# fixed, bootstrapped, capped and regularised dispersion (analysis/campaign_reg.py).
src = "analysis/campaign_reg.py -> data/v2/campaign_reg_reps.csv.gz"
G = pd.read_csv(os.path.join(V2, "campaign_reg_reps.csv.gz"))
PLANS_REG = [("pred", "fixed"), ("boot", "bootstrap"), ("bootcap", "boot., cap 15"),
             ("penpred1", "pen., $s{=}1$, fixed"), ("pen0.5", "pen.\\ boot., $s{=}0.5$"),
             ("pen1", "pen.\\ boot., $s{=}1$"), ("pen2", "pen.\\ boot., $s{=}2$")]
DESIGNS = [(5, 1), (10, 1), (5, 4)]
rows = []
for key, label in PLANS_REG:
    cells_ = []
    for S_, M_ in DESIGNS[:2]:
        g = G[(G.S == S_) & (G.M == M_)]
        cov, un = g[f"{key}_cover"], g[f"{key}_units"]
        big = int((un > 5 * g.oracle_units).sum())
        cells_ += [pct(cov.mean()), pct(cov.quantile(0.1)), f"{un.median():.0f}", f"{un.quantile(0.9):.0f}", str(big)]
    rows.append(f"{label} & " + " & ".join(cells_) + " \\\\")
rows.append("\\midrule")
cells_ = []
for S_, M_ in DESIGNS[:2]:
    g = G[(G.S == S_) & (G.M == M_)]
    cells_ += [pct(g.oracle_cover.mean()), pct(g.oracle_cover.quantile(0.1)),
               f"{g.oracle_units.median():.0f}", f"{g.oracle_units.quantile(0.9):.0f}", "--"]
rows.append("known param. & " + " & ".join(cells_) + " \\\\")
w("tab_reg.tex", "\n".join(rows) + "\n")

g5 = G[(G.S == 5) & (G.M == 1)]
n5 = len(g5)
put("rgN", str(n5), "SIM", src, "campaigns with S=5, M=1")
put("rgReps", str(int(G.groupby("cell").size().min())), "SIM", src)
big = g5.boot_units > 5 * g5.oracle_units
put("rgBigN", str(int(big.sum())), "SIM", src, "bootstrap plan > 5x oracle, reproduces Table IV")
put("rgBigMax", f"{g5.boot_units.max():,}".replace(",", "\\,"), "SIM", src)
for key, tag in (("pred", "Pred"), ("pen0.5", "PenHalf"), ("pen1", "PenOne"), ("pen2", "PenTwo"),
                 ("bootcap", "Cap"), ("penpred1", "PenPredOne")):
    put(f"rg{tag}Big", str(int((g5[f"{key}_units"] > 5 * g5.oracle_units).sum())), "SIM", src)
    put(f"rg{tag}Cov", pct(g5[f"{key}_cover"].mean()), "SIM", src, "S=5, M=1 mean cover")
    put(f"rg{tag}Pten", pct(g5[f"{key}_cover"].quantile(0.1)), "SIM", src)
put("rgCapInfeas", str(int(g5.bootcap_infeasible.sum())), "SIM", src, "cap 15 per reference, target missed under the bootstrap draws")
put("rgCapInfeasAll", str(int(G.bootcap_infeasible.sum())), "SIM", src)
put("rgAll", str(len(G)), "SIM", src)
# trace of the extreme plans
put("rgTraceSigMed", f"{g5[big].bsig_q90.median():.1f}", "SIM", src, "median over extreme campaigns of the 90th pct of refitted total sd")
put("rgTraceWide", f"{g5[big].n_wide.median():.0f}", "SIM", src, "refits with total sd > 1.5, median")
put("rgTraceFlat", f"{g5[big].n_flat.median():.0f}", "SIM", src, "refits with NB shape < 1, median")
put("rgTraceEither", pct(g5[big].over_from_either.median(), 0), "SIM", src,
    "share of draws above the oracle vector that come from those refits")
put("rgTraceHatSig", f"{np.hypot(g5[big].tau_hat, g5[big].omega_hat).median():.2f}", "SIM", src,
    "median total sd of the point estimate in extreme campaigns")
put("rgTraceHatSigOther", f"{np.hypot(g5[~big].tau_hat, g5[~big].omega_hat).median():.2f}", "SIM", src)
put("rgTracePredBig", str(int((g5[big].pred_units > 5 * g5[big].oracle_units).sum())), "SIM", src)
# S = 5, M = 4 in text
g54 = G[(G.S == 5) & (G.M == 4)]
for key, tag in (("pred", "Pred"), ("boot", "Boot"), ("pen1", "PenOne")):
    put(f"rgFour{tag}Cov", pct(g54[f"{key}_cover"].mean()), "SIM", src, "S=5, M=4")
    put(f"rgFour{tag}Units", f"{g54[f'{key}_units'].median():.0f}", "SIM", src)
put("rgFourOracleUnits", f"{g54.oracle_units.median():.0f}", "SIM", src)
