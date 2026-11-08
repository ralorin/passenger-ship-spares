"""
Checks on the Northern Ireland street-lighting analysis (ni_streetlights.py).

  flow       counts before and after every selection step, report status,
             closure comments and what they say about the replaced part
  dedup      sensitivity to the window that merges repeat reports on a street
  inventory  the two inventory snapshots compared street by street, a stable
             subset, exposure ramps for the LED conversion, a fit on the last
             year only
  evidence   cover per area-month with binomial, area-cluster and month-block
             intervals, paired differences, log scores, tail checks, a
             stock/shortage cost and cover-units curves; also whether any
             leave-one-area-out fit reaches the sd cap
  boot       parametric bootstrap of the variance parameters with refit, and
             plans from a predictive that integrates over them

Each part writes data/v2/ni_validation_<part>.json; --collect merges them into
data/v2/ni_validation.json.

    python ni_validation.py --part flow|dedup|inventory|evidence|boot [--jobs 7]
    python ni_validation.py --collect
"""
import argparse
import json
import math
import os
import re
from multiprocessing import Pool
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, poisson

import hglmm
import ni_streetlights as NS
import ni_streetlights_prep as P
from plan_cover import greedy_joint, plan_joint_draws_robust, plan_marginal_draws

OUT = NS.OUT
SEED = 20261103
LEVELS = (0.95, 0.98, 0.99, 0.995)
WITHIN = (("2023-04", "2024-03"), ("2024-04", "2025-03"))
FORWARD = (NS.TRAIN, ("2026-04", "2026-10"))
SNAP = (pd.Timestamp("2025-04-24"), pd.Timestamp("2026-11-01"))
NB_BOOT = 2000
PREDICTIVE = ["nb_pooled", "eb_gamma", "hier_plugin", "hier_pred", "hier_shock"]


def dump(part, obj):
    with open(os.path.join(OUT, f"ni_validation_{part}.json"), "w") as f:
        json.dump(obj, f, indent=1, default=float)


def select(cells, period):
    """ni_streetlights.load on a cell table already in memory."""
    c = cells[(cells.tech == "LED") & (cells.N_pure > 0)
              & (cells.month >= period[0]) & (cells.month <= period[1])].copy()
    c["y"] = c.lightout_pure.astype(float)
    c["hours_plan"] = c.hours_pure
    c["moy"] = c.month.str[5:]
    c["af"] = c.area + "|" + c.family
    c["am"] = c.area + "|" + c.month
    return c.reset_index(drop=True)


def levels_of(cells):
    full = select(cells, NS.TRAIN)
    return {c: sorted(full[c].unique()) for c in ("family", "moy")}


def predict(fit, model, new, Z, rng):
    """Model.predict with the number of draws taken from Z."""
    d = fit.d
    new = model.prepare(new)
    R = Z.shape[0]
    X = np.column_stack([np.ones(len(new))] + [new[c].to_numpy(float) for c in d.fixed])
    eta = Z[:, :d.p] @ X.T + np.log(new.hours_plan.to_numpy(float))[None, :]
    for i, k in enumerate(d.random):
        o = d.offsets[i]
        lev = {l: j for j, l in enumerate(d.levels[k])}
        vals = new[k].to_numpy()
        known = np.array([v in lev for v in vals])
        if known.any():
            eta[:, known] += Z[:, o + np.array([lev[v] for v in vals[known]])]
        if (~known).any():
            nl, inv = np.unique(vals[~known].astype(str), return_inverse=True)
            eta[:, ~known] += (rng.standard_normal((R, len(nl))) * fit.sigma[i])[:, inv]
    mu = np.exp(np.minimum(eta, 30))
    if fit.phi is not None:
        mu *= rng.gamma(fit.phi, 1.0 / fit.phi, mu.shape)
    return rng.poisson(mu)


# evaluation
def score(Y, y, rng):
    """Per family: log score from draws (as fleet.log_score), randomised PIT,
    and whether y exceeds the marginal 95 and 99 % quantiles."""
    R = Y.shape[0]
    ls = np.log((np.sum(Y == y[None, :], axis=0) + 0.5) / (R + 1.0))
    lo, hi = np.mean(Y < y[None, :], axis=0), np.mean(Y <= y[None, :], axis=0)
    pit = lo + rng.random(len(y)) * (hi - lo)
    q95 = plan_marginal_draws(Y, 0.95)
    q99 = plan_marginal_draws(Y, 0.99)
    return ls, pit, y > q95, y > q99


def evaluate(models, test, kind, rng, curve=True, extra=None):
    """One row per method, level and area-month, and one per method and family-month."""
    Z = dict(nb_pooled=models["nb_pooled"].draws(rng, plugin=False),
             hier_plugin=models["hier"].draws(rng, plugin=True),
             hier_pred=models["hier"].draws(rng, plugin=False),
             hier_shock=models["hier_shock"].draws(rng, plugin=False))
    src = dict(nb_pooled="nb_pooled", hier_plugin="hier", hier_pred="hier", hier_shock="hier_shock")
    levels = LEVELS if curve else (0.99,)
    am_rows, fm_rows = [], []
    for (area, month), g in test.groupby(["area", "month"]):
        g = g.sort_values("family")
        y = g.y.to_numpy().astype(int)

        def record(method, plans, Y=None):
            for lev, s in plans.items():
                s = np.asarray(s)
                am_rows.append(dict(method=method, kind=kind, area=area, month=month, level=lev,
                                    covered=int(np.all(y <= s)), units=float(s.sum()),
                                    short=float(np.maximum(y - s, 0).sum()), y=int(y.sum())))
            if Y is not None:
                ls, pit, e95, e99 = score(Y, y, rng)
                for j, fam in enumerate(g.family):
                    fm_rows.append(dict(method=method, kind=kind, area=area, month=month, family=fam,
                                        y=int(y[j]), ls=float(ls[j]), pit=float(pit[j]),
                                        exc95=bool(e95[j]), exc99=bool(e99[j])))

        mp = models["poisson"]
        Xb = mp.prepare(g)
        X = np.column_stack([np.ones(len(Xb))] + [Xb[c].to_numpy(float) for c in mp.fit.d.fixed])
        lam = np.exp(X @ mp.fit.beta) * g.hours_plan.to_numpy()
        start = {lev: np.array([int(poisson.ppf(lev, l)) for l in lam]) for lev in levels}
        Yp = rng.poisson(lam, (NS.R_DRAWS, len(lam)))
        record("poisson", {0.99: start[0.99]}, Yp)
        record("poisson_jnt", {lev: greedy_joint(lambda s: float(np.prod(poisson.cdf(s, lam))),
                                                 start[lev], lev) for lev in levels})
        Y = models["eb"].predict(area, g.family.tolist(), g.hours_plan.to_numpy(), NS.R_DRAWS, rng)
        record("eb_gamma", {lev: plan_joint_draws_robust(Y, lev) for lev in levels}, Y)
        for m, key in src.items():
            Y = models[key].predict(g, Z[m], rng)
            record(m, {lev: plan_joint_draws_robust(Y, lev) for lev in levels}, Y)
            if m == "hier_pred":
                record("hier_marg", {0.99: plan_marginal_draws(Y, 0.99)})
        if extra is not None:
            for m, fn in extra.items():
                Y = fn(g, rng)
                record(m, {lev: plan_joint_draws_robust(Y, lev) for lev in levels}, Y)
    return am_rows, fm_rows


def sigmas(models):
    return {k: [float(x) for x in models[k].fit.sigma] + [float(models[k].fit.phi or np.nan)]
            for k in ("hier", "hier_shock")}


def _loo(args):
    cells, split, i, area, levels, curve = args
    (tr, te) = split
    inv_te = "2026" if te[0] >= "2026" else "2025"
    train = select(cells["2025"], tr)
    test = select(cells[inv_te], te)
    test = test[test.family.isin(train.family.unique()) & (test.area == area)]
    rng = np.random.default_rng([SEED, i, int(te[0][:4])])
    models = NS.fit_all(train[train.area != area], levels)
    a, f = evaluate(models, test, "new", rng, curve=curve)
    return a, f, dict(area=area, split=te[0], sigmas=sigmas(models))


def run_split(cells, split, levels, rng, jobs, new=True, curve=True):
    tr, te = split
    inv_te = "2026" if te[0] >= "2026" else "2025"
    train = select(cells["2025"], tr)
    test = select(cells[inv_te], te)
    test = test[test.family.isin(train.family.unique())]
    models = NS.fit_all(train, levels)
    am, fm = evaluate(models, test, "known", rng, curve=curve)
    caps = [dict(area="all", split=te[0], sigmas=sigmas(models))]
    if new:
        with Pool(jobs) as pool:
            for a, f, c in pool.map(_loo, [(cells, split, i, ar, levels, curve)
                                           for i, ar in enumerate(sorted(train.area.unique()))]):
                am += a
                fm += f
                caps.append(c)
    return pd.DataFrame(am), pd.DataFrame(fm), caps, models


def summarise(am, level=0.99):
    a = am[am.level == level]
    return {f"{m}|{k}": dict(n=len(g), covered=int(g.covered.sum()), cover=float(g.covered.mean()),
                             units=float(g.units.mean()))
            for (m, k), g in a.groupby(["method", "kind"])}


def part_flow(_jobs):
    raw_lines, parsed = {}, {}
    frames = []
    for f in sorted(P.glob.glob(os.path.join(P.RAW, "LightingFaults*.csv"))):
        d = pd.read_csv(f, encoding="latin-1", low_memory=False, on_bad_lines="skip")
        with open(f, encoding="latin-1") as fh:
            raw_lines[os.path.basename(f)] = sum(1 for _ in fh) - 1
        parsed[os.path.basename(f)] = len(d)
        frames.append(d)
    F0 = pd.concat(frames, ignore_index=True)
    F = P.load_faults()
    flow = [("text lines in the fault files (excluding headers)", sum(raw_lines.values())),
            ("rows parsed (lines skipped as malformed: %d)" % sum(P.SKIPPED.values()), len(F0)),
            ("after removing repeated DATA_UID across files", len(F))]
    status_all = F.STATUS.value_counts().to_dict()
    f = F[F.FAULT_TYPE.isin(P.KEEP)]
    flow.append(("fault type Light Out or All lights out in street", len(f)))
    f = f[f.ASSET_TYPE == "Lighting Point"]
    flow.append(("asset type Lighting Point", len(f)))
    f = f[f.OPERATIONAL_AREA != "Unable to determine"]
    flow.append(("operational area recorded", len(f)))
    f = f[f.date.notna()]
    flow.append(("valid date", len(f)))
    status_kept = f.STATUS.value_counts().to_dict()
    comments = f.CLEARED_COMMENTS.fillna("").str.strip()
    empty_share = float((comments == "").mean())

    pats = [("cable", r"cable"), ("NIE (supply)", r"NIE"), ("working on attendance", r"working on attendance"),
            ("not out / no fault / all ok / operational", r"not out|no fault|all ok|operational")]
    rest = f.copy()
    excl = {}
    for name, pat in pats:
        hit = rest.CLEARED_COMMENTS.fillna("").str.contains(pat, case=False, regex=True)
        excl[name] = int(hit.sum())
        rest = rest[~hit]
    flow.append(("closure comment does not point away from the lantern", len(rest)))

    cells, report, info = P.build("2025", F=F)
    Fk, M = info["faults"], info["matched"]
    flow.append(("after merging repeats on a street within 7 days", len(Fk)))
    flow.append(("  of which Light Out", int((Fk.kind == "light_out").sum())))
    led = ~M.family.isin(["SON", "SOX", "other"])
    ids_matched = M.DATA_UID.unique()
    flow.append(("street found in the April 2025 inventory", len(ids_matched)))
    pure_led_lo = M[M.pure & led & (M.kind == "light_out")]
    flow.append(("Light Out on a street with one LED family (any month)", len(pure_led_lo)))
    windows = {"fit Apr 2023-Mar 2025": NS.TRAIN, "within-period fit": WITHIN[0],
               "within-period test": WITHIN[1]}
    counts = {}
    for name, (a, b) in windows.items():
        counts[name] = int(((pure_led_lo.month >= a) & (pure_led_lo.month <= b)).sum())
    cells26, _, info26 = P.build("2026", F=F)
    M26 = info26["matched"]
    lo26 = M26[M26.pure & ~M26.family.isin(["SON", "SOX", "other"]) & (M26.kind == "light_out")]
    counts["forward test Apr-Oct 2026 (Nov 2026 inventory)"] = int(
        ((lo26.month >= FORWARD[1][0]) & (lo26.month <= FORWARD[1][1])).sum())
    # cross-check against the cell tables the analysis reads
    check = dict(fit=float(select(cells, NS.TRAIN).y.sum()),
                 within_test=float(select(cells, WITHIN[1]).y.sum()),
                 forward=float(select(cells26, FORWARD[1]).y.sum()))

    # what kept reports say at closure
    kept = pure_led_lo[(pure_led_lo.month >= NS.TRAIN[0]) & (pure_led_lo.month <= NS.TRAIN[1])]
    txt = kept.CLEARED_COMMENTS.fillna("").str.strip().str.upper()
    keys = {"driver": r"DRIVER|GEAR", "lantern / luminaire": r"LANTERN|LUMINAIRE|HEAD",
            "photocell / PECU": r"PHOTO|PECU|CELL", "lamp / LED unit": r"LAMP|BULB|\bLED\b",
            "fuse": r"FUSE", "replaced / repaired": r"REPLAC|REPAIR|RENEW|FIXED",
            "duplicate / see other card": r"DUPLICATE|SEE CARD|REPORTED UNDER|RAISED UNDER|REF W",
            "sign lantern": r"SIGN", "third party / not adopted": r"THIRD PARTY|NOT ADOPTED|PRIVATE"}
    kw = {k: int(txt.str.contains(p, regex=True).sum()) for k, p in keys.items()}
    status_final = kept.STATUS.value_counts().to_dict()
    top = txt[txt != ""].value_counts().head(25).to_dict()

    out =dict(flow=flow, exclusions_by_comment=excl, status_all=status_all,
               status_after_type_filters=status_kept, empty_comment_share_after_type_filters=empty_share,
               light_out_pure_led_by_window=counts, cell_table_check=check,
               fit_window_reports=len(kept), fit_window_empty_comment=int((txt == "").sum()),
               fit_window_status=status_final, fit_window_keyword_counts=kw,
               fit_window_top_comments=top,
               all_out_in_fit_window_pure_led=int(((M.pure & led & (M.kind == "all_out"))
                                                   & (M.month >= NS.TRAIN[0])
                                                   & (M.month <= NS.TRAIN[1])).sum()),
               area_assignment=dict(direct=report["area_direct_share"],
                                    nearest=report["area_nearest_share"]))
    dump("flow", out)
    print(json.dumps(out, indent=1, default=float)[:6000])


def part_dedup(jobs, only=None):
    F = P.load_faults()
    A25 = P.load_inventory("2025")
    # repeat_days=0 still merges reports less than a day apart; -1 merges nothing
    variants = [("no merging", dict(repeat_days=-1)),
                ("window 0 (same day)", dict(repeat_days=0)), ("window 1", dict(repeat_days=1)),
                ("window 3", dict(repeat_days=3)), ("window 7", dict(repeat_days=7)),
                ("window 14", dict(repeat_days=14)),
                ("window 7, streets <= 10 points only", dict(repeat_days=7, merge_max_points=10)),
                ("window 7, streets <= 30 points only", dict(repeat_days=7, merge_max_points=30))]
    path = os.path.join(OUT, "ni_validation_dedup.json")
    res = json.load(open(path)) if only and os.path.exists(path) else {}
    if only:
        variants = [v for v in variants if v[0] == only]
    for name, kw in variants:
        c25, rep, _ = P.build("2025", F=F, A=A25.copy(), **kw)
        cells = {"2025": c25}
        levels = levels_of(c25)
        full = select(c25, NS.TRAIN)
        comp = NS.Model(full, levels, ["area", "af", "am"]).fit
        rng = np.random.default_rng(SEED)
        loo = kw.get("repeat_days") in (-1, 0, 14) and "merge_max_points" not in kw
        am, fm, caps, _ = run_split(cells, WITHIN, levels, rng, jobs, new=loo, curve=False)
        s = summarise(am)
        res[name] = dict(dropped_repeats=rep["dropped"]["repeats"], fit_failures=float(full.y.sum()),
                         sigma_area=float(comp.sigma[0]), sigma_area_family=float(comp.sigma[1]),
                         sigma_area_month=float(comp.sigma[2]), phi=float(comp.phi),
                         within={k: v for k, v in s.items()
                                 if k.split("|")[0] in ("poisson", "nb_pooled", "hier_pred", "eb_gamma")})
        print(name, json.dumps(res[name], default=float), flush=True)
    dump("dedup", res)


# inventory
def street_table(A):
    g = A.groupby(["street", "town"]).agg(n=("family", "size"), nfam=("family", "nunique"),
                                          family=("family", "first"))
    g["led_pure"] = (g.nfam == 1) & ~g.family.isin(["SON", "SOX", "other"])
    return g


def cells_on(info, keep, months):
    """Pure-street cell table restricted to the streets in `keep`."""
    A, M = info["inventory"], info["matched"]
    ka = pd.MultiIndex.from_frame(A[["street", "town"]]).isin(keep)
    km = pd.MultiIndex.from_frame(M[["street", "town"]]).isin(keep)
    A, M = A[ka & A.pure], M[km & M.pure & (M.kind == "light_out")]
    N = A.groupby(["area", "family"]).size().rename("N_pure").reset_index()
    grid = N.merge(pd.DataFrame({"month": months}), how="cross")
    y = (M.groupby(["OPERATIONAL_AREA", "family", "month"]).size().rename("lightout_pure")
         .rename_axis(["area", "family", "month"]).reset_index())
    c = grid.merge(y, on=["area", "family", "month"], how="left").fillna({"lightout_pure": 0})
    c["night_hours"] = c.month.map({m: P.night_hours(m) for m in months})
    c["hours_pure"] = c.N_pure * c.night_hours
    c["tech"] = np.where(c.family.isin(["SON", "SOX", "other"]), c.family, "LED")
    return c


def ramp(cells, g):
    """Exposure if the LED population grew at g per year up to the April 2025 snapshot."""
    c = cells.copy()
    t = pd.PeriodIndex(c.month, freq="M").to_timestamp() + pd.Timedelta(days=14)
    yrs = np.clip((SNAP[0] - t).days / 365.25, 0, None)
    c["hours_pure"] = c.hours_pure * np.clip(1 - g * yrs, 0.05, 1)
    return c


def comp_and_within(cells, levels, rng, jobs, new=False):
    full = select(cells, NS.TRAIN)
    comp = NS.Model(full, levels, ["area", "af", "am"]).fit
    am, _, _, _ = run_split({"2025": cells}, WITHIN, levels, rng, jobs, new=new, curve=False)
    s = summarise(am)
    return dict(fit_failures=float(full.y.sum()), sigma=[float(x) for x in comp.sigma], phi=float(comp.phi),
                within={k: v for k, v in s.items()
                        if k.split("|")[0] in ("poisson", "nb_pooled", "hier_pred", "eb_gamma")})


def part_inventory(jobs):
    F = P.load_faults()
    c25, _, info = P.build("2025", F=F)
    raw25, raw26 = P.load_inventory("2025"), P.load_inventory("2026")
    t25, t26 = street_table(info["inventory"]), street_table(raw26)
    base = t25[t25.led_pure]
    j = base.join(t26, rsuffix="_26", how="left")
    same = j.nfam_26.eq(1) & j.family_26.eq(j.family) & j.n_26.eq(j.n)
    cmp_ = dict(pure_led_streets_2025=len(base), points=int(base.n.sum()),
                identical=int(same.sum()), identical_points=int(base.n[same].sum()),
                same_family_other_count=int((j.nfam_26.eq(1) & j.family_26.eq(j.family) & ~j.n_26.eq(j.n)).sum()),
                family_changed_or_mixed=int((j.n_26.notna() & ~(j.nfam_26.eq(1) & j.family_26.eq(j.family))).sum()),
                missing_in_2026=int(j.n_26.isna().sum()),
                pure_led_streets_2026=int(t26.led_pure.sum()),
                new_pure_led_2026=int(len(t26[t26.led_pure].index.difference(base.index))))
    years = (SNAP[1] - SNAP[0]).days / 365.25
    tech = lambda A: A.family.isin(["SON", "SOX", "other"]).map({True: "old", False: "LED"}).value_counts()
    k25, k26 = tech(raw25), tech(raw26)
    g = float((k26["LED"] - k25["LED"]) / k25["LED"] / years)
    cmp_.update(led_points_2025=int(k25["LED"]), led_points_2026=int(k26["LED"]),
                old_points_2025=int(k25["old"]), old_points_2026=int(k26["old"]),
                years_between=years, led_growth_per_year=g,
                old_to_led_points_per_year=float((k25["old"] - k26["old"]) / years))
    print(json.dumps(cmp_, indent=1, default=float), flush=True)

    levels = levels_of(c25)
    months = sorted(c25.month.unique())
    rng = np.random.default_rng(SEED)
    res = dict(snapshots=cmp_)
    res["reference"] = comp_and_within(c25, levels, rng, jobs)
    stable = cells_on(info, base.index[same], months)
    check = cells_on(info, base.index, months)
    res["rebuild_check_fit_failures"] = float(select(check, NS.TRAIN).y.sum())
    res["stable_subset"] = comp_and_within(stable, levels_of(stable), rng, jobs, new=True)
    for name, gg in (("ramp at observed growth", g), ("ramp at twice the growth", 2 * g)):
        res[name] = comp_and_within(ramp(c25, gg), levels, rng, jobs)
    last = select(c25, ("2024-04", "2025-03"))
    m = NS.Model(last, levels, ["area", "af", "am"]).fit
    res["fit Apr 2024-Mar 2025 only"] = dict(fit_failures=float(last.y.sum()),
                                             sigma=[float(x) for x in m.sigma], phi=float(m.phi))
    dump("inventory", res)
    print(json.dumps(res, indent=1, default=float), flush=True)


# evidence
def cluster_ci(df, col, by="area", B=NB_BOOT, seed=SEED):
    """Percentile interval for the mean of col, resampling whole groups."""
    g = df.groupby(by)[col].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(g), (B, len(g)))
    vals = g["sum"].to_numpy()[pick].sum(1) / g["count"].to_numpy()[pick].sum(1)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def clopper(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta_dist.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - a / 2, k + 1, n - k)
    return [float(lo), float(hi)]


def evidence_tables(am, fm):
    out = {}
    a99 = am[am.level == 0.99]
    for (m, k), g in a99.groupby(["method", "kind"]):
        n, c = len(g), int(g.covered.sum())
        per_area = g.groupby("area").covered.mean()
        out[f"{m}|{k}"] = dict(
            n=n, exceed=n - c, cover=c / n, cp95=clopper(c, n),
            one_sided_lower95=float(beta_dist.ppf(0.05, c, n - c + 1)) if c else 0.0,
            area_cluster95=cluster_ci(g, "covered"),
            month_block95=cluster_ci(g, "covered", by="month"),
            units=float(g.units.mean()), short_per_area_month=float(g.short.mean()),
            worst_areas=per_area.sort_values().head(3).round(4).to_dict(),
            areas_below_99=int((per_area < 0.99).sum()))
    pairs = [("hier_pred", "nb_pooled"), ("hier_pred", "poisson"), ("hier_pred", "eb_gamma"),
             ("hier_plugin", "nb_pooled"), ("hier_pred", "hier_plugin")]
    diffs = {}
    for k in a99.kind.unique():
        w = a99[a99.kind == k].pivot_table(index=["area", "month"], columns="method",
                                           values=["covered", "units"])
        for x, y in pairs:
            if x not in w["covered"] or y not in w["covered"]:
                continue
            d = pd.DataFrame(dict(area=w.index.get_level_values("area"),
                                  dc=(w["covered"][x] - w["covered"][y]).to_numpy(),
                                  du=(w["units"][x] - w["units"][y]).to_numpy()))
            diffs[f"{x}-{y}|{k}"] = dict(d_cover=float(d.dc.mean()), d_cover95=cluster_ci(d, "dc"),
                                         d_units=float(d.du.mean()), d_units95=cluster_ci(d, "du"))
    scores, sdiff = {}, {}
    for (m, k), g in fm.groupby(["method", "kind"]):
        scores[f"{m}|{k}"] = dict(n=len(g), log_score=float(g.ls.mean()),
                                  ci95=cluster_ci(g, "ls"),
                                  exceed_q95=float(g.exc95.mean()), exceed_q99=float(g.exc99.mean()),
                                  pit_above_099=float((g.pit > 0.99).mean()),
                                  pit_above_095=float((g.pit > 0.95).mean()))
    for k in fm.kind.unique():
        w = fm[fm.kind == k].pivot_table(index=["area", "month", "family"], columns="method",
                                         values="ls").reset_index()
        for x, y in [("hier_pred", "nb_pooled"), ("hier_plugin", "nb_pooled"),
                     ("hier_pred", "poisson"), ("hier_pred", "eb_gamma")]:
            if x in w and y in w:
                d = pd.DataFrame(dict(area=w.area, d=w[x] - w[y]))
                sdiff[f"{x}-{y}|{k}"] = dict(d=float(d.d.mean()), ci95=cluster_ci(d, "d"))
    cost = {}
    for r in (1, 5, 10, 20, 50, 100):
        cost[r] = {f"{m}|{k}": float((g.units + r * g.short).mean())
                   for (m, k), g in a99.groupby(["method", "kind"])}
    curve = {f"{m}|{k}": {str(lev): dict(cover=float(h.covered.mean()), units=float(h.units.mean()))
                          for lev, h in g.groupby("level")}
             for (m, k), g in am.groupby(["method", "kind"]) if g.level.nunique() > 1}
    return dict(cover=out, paired=diffs, log_scores=scores, log_score_diffs=sdiff,
                cost_units_plus_ratio_times_shortage=cost, cover_units_curve=curve)


def part_evidence(jobs):
    cells = {"2025": pd.read_csv(os.path.join(OUT, "ni_cells_2025.csv")),
             "2026": pd.read_csv(os.path.join(OUT, "ni_cells_2026.csv"))}
    levels = levels_of(cells["2025"])
    res, caps_all = {}, []
    for name, split in (("within", WITHIN), ("forward", FORWARD)):
        rng = np.random.default_rng([SEED, len(name)])
        am, fm, caps, _ = run_split(cells, split, levels, rng, jobs)
        am.to_csv(os.path.join(OUT, f"ni_validation_am_{name}.csv.gz"), index=False)
        fm.to_csv(os.path.join(OUT, f"ni_validation_fm_{name}.csv.gz"), index=False)
        res[name] = evidence_tables(am, fm)
        caps_all += caps
        print(name, json.dumps(res[name]["cover"], default=float)[:3000], flush=True)
    allsig = np.array([v for c in caps_all for k, v in c["sigmas"].items() for v in v[:-1]])
    res["sd_cap"] = dict(fits=len(caps_all) * 2, max_sd=float(np.nanmax(allsig)),
                         at_cap=int(np.sum(allsig >= 2.999)), detail=caps_all)
    dump("evidence", res)


# boot
def simulate_like(fit, cells, rng):
    d = fit.d
    eta = d.X @ fit.beta + d.log_off
    for i, k in enumerate(d.random):
        eta = eta + rng.normal(0, fit.sigma[i], d.q[i])[d.idx[k]]
    mu = np.exp(eta) * rng.gamma(fit.phi, 1.0 / fit.phi, len(eta))
    return cells.assign(y=rng.poisson(mu).astype(float))


def _refit(args):
    sim, levels, random, theta_hat = args
    cells = NS.Model.prepare(SimpleNamespace(levels=levels), sim)
    fx = [c for c in cells.columns if "=" in c]
    f = hglmm.Fit(hglmm.Design(cells, fx, random, y="y", H="hours_pure")).fit(
        theta0=np.asarray(theta_hat), sigma_max=NS.SIGMA_MAX)
    return [float(x) for x in f.theta]


def boot_thetas(model, train, levels, random, B, seed, jobs):
    rng = np.random.default_rng(seed)
    sims = [simulate_like(model.fit, train, rng) for _ in range(B)]
    with Pool(jobs) as pool:
        th = pool.map(_refit, [(s, levels, random, model.fit.theta.tolist()) for s in sims])
    return np.array(th)


class Conditional:
    """The fit's Laplace approximation on the observed data at another theta."""

    def __init__(self, fit, theta):
        self.__dict__.update({k: v for k, v in fit.__dict__.items()})
        self.theta = np.asarray(theta)
        self.loglik, self.z, self.blocks = fit.laplace(self.theta, fit.z)
        self.sigma, self.phi = fit._unpack(self.theta)
        Haa, Hac, hcc = self.blocks
        Sa = Haa if hcc is None else Haa - (Hac / hcc) @ Hac.T
        self.cov_a = np.linalg.inv(Sa)

    draw_latent = hglmm.Fit.draw_latent


def boot_predictive(model, thetas, R_total=NS.R_DRAWS):
    conds = [Conditional(model.fit, th) for th in thetas]
    per = int(math.ceil(R_total / len(conds)))

    def fn(g, rng):
        Ys = [predict(c, model, g, c.draw_latent(per, rng), rng) for c in conds]
        return np.vstack(Ys)[:R_total]
    return fn


def _loo_boot(args):
    cells, area, levels, B, seed = args
    train = select(cells, WITHIN[0])
    test = select(cells, WITHIN[1])
    test = test[test.family.isin(train.family.unique()) & (test.area == area)]
    tr = train[train.area != area]
    rng = np.random.default_rng(seed)
    models = NS.fit_all(tr, levels)
    sims = [simulate_like(models["hier"].fit, tr, rng) for _ in range(B)]
    th = np.array([_refit((s, levels, ["area", "af"], models["hier"].fit.theta.tolist())) for s in sims])
    extra = {"hier_boot": boot_predictive(models["hier"], th)}
    a, f = evaluate(models, test, "new", rng, curve=False, extra=extra)
    return a, f, th.tolist()


def theta_summary(th, names):
    s = np.exp(th)
    out = {}
    for j, nm in enumerate(names):
        out[nm] = dict(p2_5=float(np.percentile(s[:, j], 2.5)), median=float(np.median(s[:, j])),
                       p97_5=float(np.percentile(s[:, j], 97.5)))
        if nm != "phi":
            out[nm].update(near_zero=float(np.mean(s[:, j] < 0.01)),
                           at_cap=float(np.mean(s[:, j] >= NS.SIGMA_MAX * 0.999)))
    return out


def part_boot(jobs, B=100):
    cells = pd.read_csv(os.path.join(OUT, "ni_cells_2025.csv"))
    levels = levels_of(cells)
    res = {}
    # components of the full model on the whole fitting period
    full = select(cells, NS.TRAIN)
    mfull = NS.Model(full, levels, ["area", "af", "am"])
    th = boot_thetas(mfull, full, levels, ["area", "af", "am"], B, SEED, jobs)
    res["components_full"] = dict(hat=[float(x) for x in np.exp(mfull.fit.theta)],
                                  boot=theta_summary(th, ["area", "area_family", "area_month", "phi"]))
    print(json.dumps(res, indent=1, default=float), flush=True)

    # within-period hierarchy: theta-integrated predictive against hier_pred
    train, test = select(cells, WITHIN[0]), select(cells, WITHIN[1])
    test = test[test.family.isin(train.family.unique())]
    models = NS.fit_all(train, levels)
    th = boot_thetas(models["hier"], train, levels, ["area", "af"], B, SEED + 1, jobs)
    res["within_hier"] = dict(hat=[float(x) for x in np.exp(models["hier"].fit.theta)],
                              boot=theta_summary(th, ["area", "area_family", "phi"]),
                              share_area=dict(zip(("p2_5", "median", "p97_5"), np.percentile(
                                  np.exp(2 * th[:, 0]) / (np.exp(2 * th[:, 0]) + np.exp(2 * th[:, 1])),
                                  [2.5, 50, 97.5]).tolist())))
    rng = np.random.default_rng(SEED + 2)
    am, fm = evaluate(models, test, "known", rng, curve=True,
                      extra={"hier_boot": boot_predictive(models["hier"], th)})
    with Pool(jobs) as pool:
        out = pool.map(_loo_boot, [(cells, a, levels, B, SEED + 10 + i)
                                   for i, a in enumerate(sorted(train.area.unique()))])
    loo_th = []
    for a, f, t in out:
        am += a
        fm += f
        loo_th += t
    am, fm = pd.DataFrame(am), pd.DataFrame(fm)
    am.to_csv(os.path.join(OUT, "ni_validation_am_boot.csv.gz"), index=False)
    fm.to_csv(os.path.join(OUT, "ni_validation_fm_boot.csv.gz"), index=False)
    ev = evidence_tables(am, fm)
    keep = lambda d: {k: v for k, v in d.items() if k.split("|")[0] in ("hier_pred", "hier_boot", "hier_plugin", "nb_pooled")}
    res["within_plans"] = dict(cover=keep(ev["cover"]), log_scores=keep(ev["log_scores"]),
                               curve=keep(ev["cover_units_curve"]))
    res["loo_boot"] = theta_summary(np.array(loo_th), ["area", "area_family", "phi"])
    dump("boot", res)
    print(json.dumps(res, indent=1, default=float), flush=True)


def collect():
    out = {}
    for part in ("flow", "dedup", "inventory", "evidence", "boot"):
        p = os.path.join(OUT, f"ni_validation_{part}.json")
        if os.path.exists(p):
            out[part] = json.load(open(p))
    with open(os.path.join(OUT, "ni_validation.json"), "w") as f:
        json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["flow", "dedup", "inventory", "evidence", "boot"])
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--B", type=int, default=100)
    ap.add_argument("--only", help="dedup: run this variant alone and merge into the saved results")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.collect:
        collect()
    elif a.part == "boot":
        part_boot(a.jobs, a.B)
    elif a.part == "dedup":
        part_dedup(a.jobs, a.only)
    elif a.part:
        globals()[f"part_{a.part}"](a.jobs)


if __name__ == "__main__":
    main()
