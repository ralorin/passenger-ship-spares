"""
Simulated passenger-ship fleet (E2, E3): records, competing predictors and the
cover their spares plans achieve on missions they did not see.

Fleet parameters are in fleet_params.json, with their sources. Every number
produced here is simulated.

Truth for the failures of reference p in zone z of ship s on mission t:

    rate = (1 / MTBF_p) * field_factor * Arrhenius(zone temperature)
           * age_factor * exp(u_s + w_sp + m_pk + e_cell)       (k = maker)
    failures ~ Poisson(N_szp * powered hours * rate)

plus, depending on the scenario, ship-wide shock events, a bad production lot,
refits that reset unit age, and preventive replacements recorded as failures.
Predictors only see what is known before sailing: product, maker, zone, class,
the climate normal of the planned itinerary, planned occupancy and unit age.

Predictors, each giving a joint 99 % stock vector for a ship's mission:
  mtbf         datasheet MTBF, Poisson, each reference at its own quantile
  mtbf_joint   datasheet MTBF, Poisson, joint plan
  nb_pooled    negative binomial GLM, no ship terms
  eb_gamma     gamma-Poisson empirical Bayes per reference
  hier_plugin  hierarchical model, parameters at their estimates
  hier_pred    hierarchical model with draws of every latent quantity
  hier_marg    hier_pred draws, each reference at its own quantile
  hier_shock   hier_pred plus a ship-mission term shared by all references
  mtbf_scaled  datasheet MTBF times one fleet-wide factor fitted to the record, joint plan
  poisson_glm  Poisson GLM with the pooled covariates and coefficient draws, joint plan
  hier_agg     hier_pred fitted and predicted on cells summed over zones (NB2 granularity check)

Heavy-tailed effects are Student-t3 on the log scale, truncated at 4 sd and
shifted so that the mean multiplier equals the Gaussian one; an untruncated t
has no finite exponential moment.

    python fleet.py --list
    python fleet.py --worker I --nworkers N [--hours 3.6]
    python fleet.py --summary
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.special import gammaln
from scipy.stats import poisson, t as student_t

import hglmm
from plan_cover import greedy_joint, plan_joint_draws_robust, plan_marginal_draws

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")
RAW = os.path.join(OUT, "raw", "fleet_v4")
PAR = json.load(open(os.path.join(HERE, "fleet_params.json"), encoding="utf-8"))

SEED = 20261103
LEVEL = 0.99
R_DRAWS = 6000
REPS = 100
K_EV = 8.617e-5
SD = dict(tau=0.3, omega=0.3, maker=0.3, eps=0.15)
SIGMA_MAX = 3.0     # cap on fitted log-rate sds; see hglmm.Fit.fit
T_TRUNC = 4.0       # heavy-tail effects are truncated at this many sds

BASE = dict(latent="normal", shock_rate=0.0, shock_size=0.0, misclass=0.0, lot=False,
            refit=False, field_factor=PAR["field_factor"], ship_frac=1.0,
            horizon_days=PAR["horizon_days"], train_days=PAR["train_days"])
REALISTIC = dict(BASE, shock_rate=PAR["shock_rate_per_ship_year"], shock_size=0.0005, misclass=0.10,
                 lot=True, refit=True)
SCENARIOS = {
    "clean": BASE,
    "baseline": REALISTIC,
    "heavy_tail": dict(REALISTIC, latent="t3"),
    "shocks": dict(REALISTIC, shock_rate=0.5, shock_size=0.001),
    "misclass": dict(REALISTIC, misclass=0.25),
    "sparse": dict(REALISTIC, ship_frac=0.5, horizon_days=365, train_days=274),
    "datasheet_ok": dict(REALISTIC, field_factor=1.0),
    "shocks_frequent": dict(REALISTIC, shock_rate=2.0, shock_size=0.0005),
    "shocks_severe": dict(REALISTIC, shock_rate=2.0, shock_size=0.002),
}
METHODS = ["mtbf", "mtbf_joint", "mtbf_scaled", "poisson_glm", "nb_pooled", "eb_gamma",
           "hier_plugin", "hier_pred", "hier_marg", "hier_shock", "hier_agg"]
PRODUCTS = list(PAR["products"])
COVARIATES = ["temp_c", "log_age", "occ_c"]
COST = np.array([PAR["products"][p]["cost"] for p in PRODUCTS])


# ---------------------------------------------------------------- simulation --
_T_SHIFT = {}


def t3_shift(sd):
    """Shift making E exp(x) equal exp(sd^2/2) for the truncated, scaled t3."""
    if sd not in _T_SHIFT:
        s = sd / math.sqrt(3.0)
        lim = T_TRUNC * sd / s
        mass = student_t.cdf(lim, 3) - student_t.cdf(-lim, 3)
        m = integrate.quad(lambda z: math.exp(s * z) * student_t.pdf(z, 3), -lim, lim)[0] / mass
        _T_SHIFT[sd] = math.log(m) - sd ** 2 / 2
    return _T_SHIFT[sd]


def latent(dist, sd, size, rng):
    if dist == "t3":
        x = rng.standard_t(3, size) * sd / math.sqrt(3.0)
        bad = np.abs(x) > T_TRUNC * sd
        while bad.any():
            x[bad] = rng.standard_t(3, bad.sum()) * sd / math.sqrt(3.0)
            bad = np.abs(x) > T_TRUNC * sd
        return x - t3_shift(sd)
    return rng.normal(0, sd, size)


def arrhenius(temp_c):
    return np.exp(PAR["activation_energy_ev"] / K_EV * (1 / 298.15 - 1 / (np.asarray(temp_c) + 273.15)))


def zone_temp(zone, ambient):
    if zone == "exterior":
        return ambient
    slope = {"technical": 0.3, "galley": 0.2}.get(zone, 0.1)
    return PAR["zone_temp_base_c"][zone] + slope * (ambient - 22.0)


def hours_per_day(use, occ, ambient):
    return {"light": 10.0 * occ, "occupancy": 24.0 * (0.3 + 0.7 * occ),
            "hvac": 24.0 * min(1.0, 0.5 + 0.03 * abs(ambient - 22.0)),
            "public": 18.0, "galley": 16.0, "night": 12.0, "always": 24.0}[use]


def zone_counts(c):
    return dict(pax_cabin=c["pax_cabins"], crew_cabin=c["crew_cabins"], public=c["public_units"],
                galley=c["galley_units"], exterior=c["exterior_units"], technical=c["technical_units"])


def itinerary(cls, horizon, rng):
    c = PAR["classes"][cls]
    regions = list(PAR["regions"])
    durations = c.get("durations_nights", PAR["durations_nights"])
    fixed = c.get("region")
    winter = fixed or rng.choice(["caribbean", "mediterranean"])
    summer = fixed or rng.choice([r for r in regions if r != "caribbean"])
    day, out = 0, []
    while day < horizon:
        dur = int(rng.choice(durations))
        month = (10 + day // 30) % 12            # day 0 is 1 November
        season = "winter" if month in (10, 11, 0, 1, 2, 3) else "summer"
        region = winter if season == "winter" else summer
        occ_plan = float(np.clip(rng.normal(PAR["occupancy_mean"], PAR["occupancy_sd"]), 0.4, 1.0))
        out.append(dict(start=day, nights=dur, region=region, season=season, occ_plan=occ_plan))
        day += dur
    return out


def simulate(scn, rng):
    """Returns the cell records (with true corrective failures) of one fleet."""
    makers = PAR["makers_per_product"]
    m_eff = {(p, k): latent(scn["latent"], SD["maker"], 1, rng)[0]
             for p in PRODUCTS for k in range(makers)}
    rows, ship = [], 0
    for cls, c in PAR["classes"].items():
        n = max(1, round(c["ships"] * scn["ship_frac"]))
        maker_of = {p: int(rng.integers(makers)) for p in PRODUCTS}
        lot = None
        if scn["lot"] and rng.random() < 0.5:
            lot = (str(rng.choice(PRODUCTS)), set(np.flatnonzero(rng.random(n) < 0.5)))
        counts = zone_counts(c)
        for i in range(n):
            u = latent(scn["latent"], SD["tau"], 1, rng)[0]
            w = {p: latent(scn["latent"], SD["omega"], 1, rng)[0] for p in PRODUCTS}
            age0 = rng.uniform(*PAR["age_years"])
            refit_day = rng.uniform(0, scn["horizon_days"]) if scn["refit"] and rng.random() < 0.3 else None
            for t, mis in enumerate(itinerary(cls, scn["horizon_days"], rng)):
                normal = PAR["regions"][mis["region"]][mis["season"]]
                ambient = normal + rng.normal(0, 2.0)
                occ = float(np.clip(mis["occ_plan"] + rng.normal(0, 0.03), 0.3, 1.0))
                # a ship-wide supply event (blackout and recovery) damages a share of every population
                shock = rng.random() < -math.expm1(-scn["shock_rate"] * mis["nights"] / 365.0)
                for p in PRODUCTS:
                    spec = PAR["products"][p]
                    for zone, per in spec["zones"].items():
                        N = counts[zone] * per
                        if N == 0:
                            continue
                        age = age0 + mis["start"] / 365.0
                        if refit_day is not None and mis["start"] >= refit_day and zone.endswith("cabin"):
                            age = (mis["start"] - refit_day) / 365.0
                        age = max(age, 0.25)
                        h_act = mis["nights"] * hours_per_day(spec["use"], occ, ambient)
                        h_plan = mis["nights"] * hours_per_day(spec["use"], mis["occ_plan"], normal)
                        rate = (scn["field_factor"] / spec["mtbf_h"] * arrhenius(zone_temp(zone, ambient))
                                * (age / 5.0) ** (PAR["weibull_shape"] - 1)
                                * math.exp(u + w[p] + m_eff[(p, maker_of[p])] + rng.normal(0, SD["eps"])))
                        if lot and lot[0] == p and i in lot[1]:
                            rate *= PAR["lot_factor"]
                        mu = N * h_act * rate
                        y = rng.poisson(mu)
                        if shock:
                            y += rng.binomial(N, scn["shock_size"])
                        prev = rng.poisson(0.3 * mu)
                        rec = y + (rng.binomial(prev, scn["misclass"]) if scn["misclass"] else 0)
                        rows.append(dict(ship=ship, cls=cls, zone=zone, product=p,
                                         maker=f"{p}:{maker_of[p]}", mission=t, sm=f"{ship}:{t}",
                                         sp=f"{ship}:{p}", start=mis["start"], nights=mis["nights"],
                                         temp_normal=normal, occ_plan=mis["occ_plan"], age=age, N=N,
                                         hours=N * h_act, hours_plan=N * h_plan, shock=int(shock),
                                         failures=float(rec), true_failures=int(y)))
            ship += 1
    d = pd.DataFrame(rows)
    d["temp_c"] = d.temp_normal - 22.0
    d["log_age"] = np.log(d.age / 5.0)
    d["occ_c"] = d.occ_plan - 0.9
    return d


# ---------------------------------------------------------------- predictors --
def dummies(d, col, levels):
    return {f"{col}={v}": (d[col] == v).astype(float) for v in levels[1:]}


class Hier:
    """Hierarchical NB model; random terms are global maker, ship, ship-product
    and optionally ship-mission (always the last, eliminated term)."""

    def __init__(self, train, levels, shock=False):
        self.levels = levels
        self.shock = shock
        self.span = {c: (train[c].min(), train[c].max()) for c in COVARIATES}
        cells = self.prepare(train)
        self.fixed = [c for c in cells.columns if "=" in c] + COVARIATES
        self.random = ["maker", "ship", "sp"] + (["sm"] if shock else [])
        self.fit = hglmm.Fit(hglmm.Design(cells, self.fixed, self.random)).fit(sigma_max=SIGMA_MAX)

    # Class is left out: size enters through exposure, and with few ships per
    # class a class effect cannot be told apart from the ship effect. Of the
    # zones only crew against passenger cabins is kept; the other zones hold a
    # single reference and are absorbed by the product effect.
    factors = ("product",)

    def prepare(self, d):
        d = d.reset_index(drop=True)
        cols = {"crew=1": (d.zone == "crew_cabin").astype(float)}
        for col in self.factors:
            cols.update(dummies(d, col, self.levels[col]))
        return pd.concat([d, pd.DataFrame(cols)], axis=1)

    def draws(self, R, rng, plugin):
        f = self.fit
        if plugin:
            return np.tile(f.z, (R, 1))
        return f.draw_latent(R, rng)

    def predict(self, new, Z, rng):
        """Counts per cell of `new` (one ship-mission), shape (R, cells)."""
        f = self.fit
        d = f.d
        new = self.prepare(new)
        # A planned mission can lie outside the record, e.g. units renewed by a
        # refit are younger than any unit seen. Hold covariates to the range of
        # the record rather than extrapolate the fitted slopes.
        for c, (lo, hi) in self.span.items():
            new[c] = new[c].clip(lo, hi)
        R = Z.shape[0]
        X = np.column_stack([np.ones(len(new))] + [new[c].to_numpy(float) for c in d.fixed])
        eta = Z[:, :d.p] @ X.T + np.log(new.hours_plan.to_numpy(float))[None, :]
        for i, k in enumerate(d.random):
            o = d.offsets[i]
            lev = self._lev(k)
            vals = new[k].to_numpy()
            known = np.array([v in lev for v in vals])
            if known.any():
                eta[:, known] += Z[:, o + np.array([lev[v] for v in vals[known]])]
            if (~known).any():
                nl, inv = np.unique(vals[~known].astype(str), return_inverse=True)
                eta[:, ~known] += (rng.standard_normal((R, len(nl))) * f.sigma[i])[:, inv]
        mu = np.exp(np.minimum(eta, 30))
        mu *= rng.gamma(f.phi, 1.0 / f.phi, mu.shape)
        return rng.poisson(mu)

    def _lev(self, k):
        if not hasattr(self, "_levmap"):
            self._levmap = {}
        if k not in self._levmap:
            self._levmap[k] = {l: j for j, l in enumerate(self.fit.d.levels[k])}
        return self._levmap[k]


class Pooled(Hier):
    def __init__(self, train, levels):
        self.levels = levels
        self.shock = False
        self.span = {c: (train[c].min(), train[c].max()) for c in COVARIATES}
        cells = self.prepare(train)
        self.fixed = [c for c in cells.columns if "=" in c] + COVARIATES
        self.random = []
        self.fit = hglmm.Fit(hglmm.Design(cells, self.fixed, [])).fit(sigma_max=SIGMA_MAX)

    factors = ("maker",)            # maker is nested in product


class PoissonGLM(Pooled):
    def __init__(self, train, levels):
        self.levels = levels
        self.shock = False
        self.span = {c: (train[c].min(), train[c].max()) for c in COVARIATES}
        cells = self.prepare(train)
        self.fixed = [c for c in cells.columns if "=" in c] + COVARIATES
        self.random = []
        self.fit = hglmm.Fit(hglmm.Design(cells, self.fixed, []), family="poisson").fit()

    def predict(self, new, Z, rng):
        f = self.fit
        new = self.prepare(new)
        for c, (lo, hi) in self.span.items():
            new[c] = new[c].clip(lo, hi)
        X = np.column_stack([np.ones(len(new))] + [new[c].to_numpy(float) for c in f.d.fixed])
        eta = Z[:, :f.d.p] @ X.T + np.log(new.hours_plan.to_numpy(float))[None, :]
        return rng.poisson(np.exp(np.minimum(eta, 30)))


def by_product(d):
    """Cells summed over zones: one cell per ship, product and mission."""
    keys = ["ship", "cls", "product", "maker", "mission", "sm", "sp", "start", "nights",
            "temp_normal", "occ_plan", "shock"]
    d = d.assign(wage=d.log_age * d.hours)
    g = d.groupby(keys, as_index=False).agg(hours=("hours", "sum"), hours_plan=("hours_plan", "sum"),
                                            failures=("failures", "sum"),
                                            true_failures=("true_failures", "sum"),
                                            wage=("wage", "sum"), N=("N", "sum"))
    g["log_age"] = g.wage / g.hours
    g["temp_c"] = g.temp_normal - 22.0
    g["occ_c"] = g.occ_plan - 0.9
    g["zone"] = "all"
    return g.drop(columns="wage")


class HierAgg(Hier):
    def prepare(self, d):
        d = d.reset_index(drop=True)
        cols = {}
        for col in self.factors:
            cols.update(dummies(d, col, self.levels[col]))
        return pd.concat([d, pd.DataFrame(cols)], axis=1)


class EBGamma:
    """Per reference, a gamma prior on the ship-level rate fitted by moments."""

    def __init__(self, train):
        self.par = {}
        g = train.groupby(["product", "ship"]).agg(y=("failures", "sum"), E=("hours", "sum")).reset_index()
        for p, h in g.groupby("product"):
            r = h.y / h.E
            rbar = h.y.sum() / h.E.sum()
            w = h.E / h.E.sum()
            v = float(np.sum(w * (r - rbar) ** 2) - rbar * np.mean(1 / h.E))
            if rbar == 0:
                # no failure on record: Jeffreys posterior of the pooled rate
                a, b = 0.5, float(h.E.sum())
            else:
                a = rbar ** 2 / v if v > 0 else 1e6
                b = a / rbar
            self.par[p] = (a, b, dict(zip(h.ship, zip(h.y, h.E))))

    def predict(self, ship, products, H, R, rng):
        out = np.empty((R, len(products)), int)
        for j, (p, h) in enumerate(zip(products, H)):
            a, b, hist = self.par[p]
            y, E = hist.get(ship, (0.0, 0.0))
            lam = rng.gamma(a + y, 1.0 / (b + E), R)
            out[:, j] = rng.poisson(lam * h)
        return out


def log_score(Y, y):
    """Mean over references of log predictive probability, from draws."""
    R = Y.shape[0]
    return float(np.mean([math.log((np.sum(Y[:, j] == y[j]) + 0.5) / (R + 1.0)) for j in range(len(y))]))


def pit(Y, y, rng):
    lo = np.mean(Y < y[None, :], axis=0)
    hi = np.mean(Y <= y[None, :], axis=0)
    return lo + rng.random(len(y)) * (hi - lo)


# ---------------------------------------------------------------- one fleet --
def one_fleet(name, rep):
    scn = SCENARIOS[name]
    rng = np.random.default_rng([SEED, list(SCENARIOS).index(name), rep])
    d = simulate(scn, rng)
    ships = d.groupby("ship").cls.first()
    new_ships = {int(rng.choice(ships[ships == c].index)) for c in ships.unique()}
    train = d[(~d.ship.isin(new_ships)) & (d.start + d.nights <= scn["train_days"])]
    test = d[d.start >= scn["train_days"]]
    levels = {c: sorted(train[c].unique()) for c in ("product", "zone", "cls", "maker")}

    t0 = time.time()
    models = dict(nb_pooled=Pooled(train, levels), hier=Hier(train, levels),
                  hier_shock=Hier(train, levels, shock=True), eb=EBGamma(train),
                  poisson_glm=PoissonGLM(train, levels), hier_agg=HierAgg(by_product(train), levels))
    t_fit = time.time() - t0
    Z = dict(nb_pooled=models["nb_pooled"].draws(R_DRAWS, rng, plugin=False),
             poisson_glm=models["poisson_glm"].draws(R_DRAWS, rng, plugin=False),
             hier_plugin=models["hier"].draws(R_DRAWS, rng, plugin=True),
             hier_pred=models["hier"].draws(R_DRAWS, rng, plugin=False),
             hier_shock=models["hier_shock"].draws(R_DRAWS, rng, plugin=False),
             hier_agg=models["hier_agg"].draws(R_DRAWS, rng, plugin=False))
    src = dict(nb_pooled="nb_pooled", poisson_glm="poisson_glm", hier_plugin="hier",
               hier_pred="hier", hier_shock="hier_shock")
    # datasheet rates times one factor fitted to the record (maximum likelihood)
    ds_train = (train.hours / train["product"].map(lambda p: PAR["products"][p]["mtbf_h"])).sum()
    factor = float(train.failures.sum() / ds_train)

    acc = {}
    for (s, t), g in test.groupby(["ship", "mission"]):
        kind = "new" if s in new_ships else "known"
        prods = sorted(g["product"].unique())
        idx = [PRODUCTS.index(p) for p in prods]
        col = np.array([prods.index(p) for p in g["product"]])
        y = g.groupby("product").true_failures.sum().reindex(prods).to_numpy()
        H = g.groupby("product").hours_plan.sum().reindex(prods).to_numpy()
        lam_ds = np.array([H[j] / PAR["products"][p]["mtbf_h"] for j, p in enumerate(prods)])
        cost = COST[idx]
        shocked = bool(g.shock.iloc[0])

        def record(method, plan, Y=None):
            short = np.maximum(y - plan, 0)
            groups = [kind] + ([kind + "_shock"] if shocked else [])
            ls = log_score(Y, y) if Y is not None else None
            for kk in groups:
                a = acc.setdefault((method, kk), dict(n=0, covered=0, units=0.0, cost=0.0, ls=0.0,
                                                      pit=[], short=0.0, short_cost=0.0))
                a["n"] += 1
                a["covered"] += int(np.all(y <= plan))
                a["units"] += float(plan.sum())
                a["cost"] += float(cost @ plan)
                a["short"] += float(short.sum())
                a["short_cost"] += float(cost @ short)
                if Y is not None:
                    a["ls"] += ls
                    if kk == kind:
                        a["pit"].extend(pit(Y, y, rng).tolist())

        record("mtbf", np.array([int(poisson.ppf(LEVEL, l)) for l in lam_ds]))
        record("mtbf_joint", greedy_joint(lambda s: float(np.prod(poisson.cdf(s, lam_ds))),
                                          np.array([int(poisson.ppf(LEVEL, l)) for l in lam_ds]), LEVEL))
        lam_sc = lam_ds * factor
        record("mtbf_scaled", greedy_joint(lambda s: float(np.prod(poisson.cdf(s, lam_sc))),
                                           np.array([int(poisson.ppf(LEVEL, l)) for l in lam_sc]), LEVEL))
        Y = models["eb"].predict(s, prods, H, R_DRAWS, rng)
        record("eb_gamma", plan_joint_draws_robust(Y, LEVEL), Y)
        for m in ("poisson_glm", "nb_pooled", "hier_plugin", "hier_pred", "hier_shock"):
            Yc = models[src[m]].predict(g, Z[m], rng)
            Y = np.zeros((R_DRAWS, len(prods)), int)
            for j in range(len(prods)):
                Y[:, j] = Yc[:, col == j].sum(axis=1)
            record(m, plan_joint_draws_robust(Y, LEVEL), Y)
            if m == "hier_pred":
                record("hier_marg", plan_marginal_draws(Y, LEVEL))
        ga = by_product(g)
        Yc = models["hier_agg"].predict(ga, Z["hier_agg"], rng)
        cola = np.array([prods.index(p) for p in ga["product"]])
        Y = np.zeros((R_DRAWS, len(prods)), int)
        for j in range(len(prods)):
            Y[:, j] = Yc[:, cola == j].sum(axis=1)
        record("hier_agg", plan_joint_draws_robust(Y, LEVEL), Y)

    rows = []
    for (method, kind), a in acc.items():
        h = np.histogram(a["pit"], bins=10, range=(0, 1))[0] if a["pit"] else np.zeros(10, int)
        rows.append(dict(scenario=name, rep=rep, method=method, kind=kind, n=a["n"],
                         covered=a["covered"], units=a["units"], cost=a["cost"],
                         short=a["short"], short_cost=a["short_cost"],
                         log_score=a["ls"] / a["n"] if a["ls"] else np.nan,
                         **{f"pit{i}": int(v) for i, v in enumerate(h)},
                         fit_seconds=t_fit, train_cells=len(train),
                         sigma_hier=json.dumps([float(x) for x in models["hier"].fit.sigma]),
                         sigma_shock=json.dumps([float(x) for x in models["hier_shock"].fit.sigma]),
                         sigma_agg=json.dumps([float(x) for x in models["hier_agg"].fit.sigma]),
                         phi=json.dumps([float(models[k].fit.phi) for k in ("hier", "hier_agg", "nb_pooled")]),
                         factor=factor))
    return rows


# ------------------------------------------------------------------- driver --
def tasks():
    return [(s, r) for r in range(REPS) for s in SCENARIOS]


def worker(i, n, hours):
    os.makedirs(RAW, exist_ok=True)
    t0 = time.time()
    for k, (s, r) in enumerate(tasks()):
        if k % n != i:
            continue
        path = os.path.join(RAW, f"{s}_{r:03d}.csv")
        if os.path.exists(path):
            continue
        if time.time() - t0 > hours * 3600:
            print("time limit reached", flush=True)
            return
        t1 = time.time()
        try:
            rows = one_fleet(s, r)
        except Exception as exc:
            print(f"{s} {r} failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
        print(f"{s} {r} {time.time() - t1:.0f}s", flush=True)


def summary():
    R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RAW, "*.csv")))])
    R.to_csv(os.path.join(OUT, "fleet_v4_reps.csv.gz"), index=False)
    out = []
    for (s, m, k), g in R.groupby(["scenario", "method", "kind"]):
        rate = g.covered / g.n
        pit_counts = g[[f"pit{i}" for i in range(10)]].sum().to_numpy()
        out.append(dict(scenario=s, method=m, kind=k, fleets=len(g), missions=int(g.n.sum()),
                        cover=float(g.covered.sum() / g.n.sum()),
                        cover_mcse=float(rate.std(ddof=1) / math.sqrt(len(g))),
                        units=float(g.units.sum() / g.n.sum()), cost=float(g.cost.sum() / g.n.sum()),
                        log_score=float(g.log_score.mean()),
                        pit_max_dev=float(np.max(np.abs(pit_counts / max(pit_counts.sum(), 1) - 0.1)))
                        if pit_counts.sum() else np.nan))
    T = pd.DataFrame(out)
    T.to_csv(os.path.join(OUT, "fleet_v4_summary.csv"), index=False)
    print(T.pivot_table(index=["scenario", "kind"], columns="method", values="cover").round(3).to_string())
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--worker", type=int)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--hours", type=float, default=3.6)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.list:
        print(f"{len(SCENARIOS)} scenarios x {REPS} fleets = {len(tasks())} tasks")
    elif a.worker is not None:
        worker(a.worker, a.nworkers, a.hours)
    elif a.summary:
        summary()
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
