"""
Real-data analogue (E7): monthly spares of LED lanterns for each maintenance
area of the Northern Ireland street lighting network.

An area plays the part of a ship, a lantern family the part of a product
reference and a month the part of a mission. Demand is the number of
"Light Out" reports on streets carrying a single lantern family (see
ni_streetlights_prep.py). The models are fitted on April 2023 - March 2025
with the April 2025 inventory and tested on April - October 2026 with the
November 2026 inventory, so every forecast is at least a year ahead; a second
split fits on the first of those two years and tests on the second. Each test
is also run leaving each area out of the fit in turn, as a holder never seen
before.

Predictors give a joint 99 % stock vector over the families an area holds:
  poisson      family and calendar-month rates, Poisson, own quantile per family
  poisson_jnt  same rates, joint plan
  nb_pooled    negative binomial GLM, no area terms
  eb_gamma     gamma-Poisson per family across areas
  hier_plugin  area and area x family effects, parameters at their estimates
  hier_pred    the same with draws of every latent quantity
  hier_marg    hier_pred draws, each family at its own quantile
  hier_shock   hier_pred plus an area x month term shared by all families

    python ni_streetlights.py
"""
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import poisson

import hglmm
from campaign import profile_ci
from fleet import EBGamma, log_score, pit
from plan_cover import greedy_joint, plan_joint_draws_robust, plan_marginal_draws

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")
TRAIN = ("2023-04", "2025-03")
# (name, training months, inventory for training, test months, inventory for test)
SPLITS = [("forward", TRAIN, "2025", ("2026-04", "2026-10"), "2026"),
          ("within", ("2023-04", "2024-03"), "2025", ("2024-04", "2025-03"), "2025")]
LEVEL = 0.99
R_DRAWS = 6000
SEED = 20261102
SIGMA_MAX = 3.0


def load(inv, period):
    c = pd.read_csv(os.path.join(OUT, f"ni_cells_{inv}.csv"))
    c = c[(c.tech == "LED") & (c.N_pure > 0) & (c.month >= period[0]) & (c.month <= period[1])].copy()
    c["y"] = c.lightout_pure.astype(float)
    c["hours_plan"] = c.hours_pure
    c["moy"] = c.month.str[5:]
    c["af"] = c.area + "|" + c.family
    c["am"] = c.area + "|" + c.month
    return c.reset_index(drop=True)


class Model:
    def __init__(self, train, levels, random, family="nb"):
        self.levels = levels
        self.random = random
        cells = self.prepare(train)
        self.fixed = [c for c in cells.columns if "=" in c]
        self.fit = hglmm.Fit(hglmm.Design(cells, self.fixed, random, y="y", H="hours_pure"),
                             family).fit(sigma_max=SIGMA_MAX)

    def prepare(self, d):
        d = d.reset_index(drop=True)
        cols = {}
        for col in ("family", "moy"):
            for v in self.levels[col][1:]:
                cols[f"{col}={v}"] = (d[col] == v).astype(float)
        return pd.concat([d, pd.DataFrame(cols)], axis=1)

    def draws(self, rng, plugin):
        f = self.fit
        return np.tile(f.z, (R_DRAWS, 1)) if plugin else f.draw_latent(R_DRAWS, rng)

    def predict(self, new, Z, rng):
        f = self.fit
        d = f.d
        new = self.prepare(new)
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
                eta[:, ~known] += (rng.standard_normal((R_DRAWS, len(nl))) * f.sigma[i])[:, inv]
        mu = np.exp(np.minimum(eta, 30))
        if f.phi is not None:
            mu *= rng.gamma(f.phi, 1.0 / f.phi, mu.shape)
        return rng.poisson(mu)


def fit_all(train, levels):
    m = dict(poisson=Model(train, levels, [], "poisson"),
             nb_pooled=Model(train, levels, []),
             hier=Model(train, levels, ["area", "af"]),
             hier_shock=Model(train, levels, ["area", "af", "am"]))
    m["eb"] = EBGamma(pd.DataFrame(dict(product=train.family, ship=train.area,
                                        failures=train.y, hours=train.hours_pure)))
    return m


def evaluate(models, test, kind, rng, acc):
    Z = dict(poisson=models["poisson"].draws(rng, plugin=True),
             nb_pooled=models["nb_pooled"].draws(rng, plugin=False),
             hier_plugin=models["hier"].draws(rng, plugin=True),
             hier_pred=models["hier"].draws(rng, plugin=False),
             hier_shock=models["hier_shock"].draws(rng, plugin=False))
    src = dict(nb_pooled="nb_pooled", hier_plugin="hier", hier_pred="hier", hier_shock="hier_shock")
    for (area, month), g in test.groupby(["area", "month"]):
        g = g.sort_values("family")
        y = g.y.to_numpy().astype(int)

        def record(method, plan, Y=None):
            a = acc.setdefault((method, kind), dict(n=0, covered=0, units=0.0, ls=0.0, pit=[]))
            a["n"] += 1
            a["covered"] += int(np.all(y <= plan))
            a["units"] += float(np.sum(plan))
            if Y is not None:
                a["ls"] += log_score(Y, y)
                a["pit"].extend(pit(Y, y, rng).tolist())

        mp = models["poisson"]
        Xb = mp.prepare(g)
        X = np.column_stack([np.ones(len(Xb))] + [Xb[c].to_numpy(float) for c in mp.fit.d.fixed])
        lam = np.exp(X @ mp.fit.beta) * g.hours_plan.to_numpy()
        start = np.array([int(poisson.ppf(LEVEL, l)) for l in lam])
        record("poisson", start, None)
        record("poisson_jnt", greedy_joint(lambda s: float(np.prod(poisson.cdf(s, lam))), start, LEVEL))
        Y = models["eb"].predict(area, g.family.tolist(), g.hours_plan.to_numpy(), R_DRAWS, rng)
        record("eb_gamma", plan_joint_draws_robust(Y, LEVEL), Y)
        for m, key in src.items():
            Y = models[key].predict(g, Z[m], rng)
            record(m, plan_joint_draws_robust(Y, LEVEL), Y)
            if m == "hier_pred":
                record("hier_marg", plan_marginal_draws(Y, LEVEL))


def components(train, levels):
    """Variance components of the full model with boundary LRTs and profile intervals."""
    full = Model(train, levels, ["area", "af", "am"]).fit
    out = dict(sigma=dict(zip(["area", "area_family", "area_month"], full.sigma.tolist())),
               phi=full.phi, loglik=full.loglik, family_effects={})
    names = ["area", "af", "am"]
    for k, nm in enumerate(names):
        red = Model(train, levels, [x for x in names if x != nm]).fit
        t, p = hglmm.lrt_boundary(full.loglik, red.loglik)
        lo, hi = profile_ci(full, k, red.loglik)
        out[f"lrt_{nm}"], out[f"p_{nm}"], out[f"ci_{nm}"] = t, p, [lo, hi]
    fx = full.d.fixed
    for j, c in enumerate(fx):
        if c.startswith("family="):
            out["family_effects"][c[7:]] = float(full.beta[1 + j])
    return out


def main():
    rng = np.random.default_rng(SEED)
    full = load("2025", TRAIN)
    levels = {c: sorted(full[c].unique()) for c in ("family", "moy")}
    comp = components(full, levels)

    rows, sizes = [], {}
    for name, tr, inv_tr, te, inv_te in SPLITS:
        train, test = load(inv_tr, tr), load(inv_te, te)
        test = test[test.family.isin(train.family.unique())]
        sizes[name] = dict(train=tr, test=te, train_failures=int(train.y.sum()),
                           test_failures=int(test.y.sum()), test_cells=len(test))
        acc = {}
        evaluate(fit_all(train, levels), test, "known", rng, acc)
        for area in sorted(train.area.unique()):
            evaluate(fit_all(train[train.area != area], levels), test[test.area == area], "new", rng, acc)
        for (method, kind), a in acc.items():
            h = np.histogram(a["pit"], bins=10, range=(0, 1))[0] if a["pit"] else np.zeros(10)
            rows.append(dict(split=name, method=method, kind=kind, n=a["n"], covered=a["covered"],
                             cover=a["covered"] / a["n"], units=a["units"] / a["n"],
                             log_score=a["ls"] / a["n"] if a["pit"] else np.nan,
                             pit_max_dev=float(np.max(np.abs(h / max(h.sum(), 1) - 0.1)))
                             if a["pit"] else np.nan))
    T = pd.DataFrame(rows).sort_values(["split", "kind", "method"])
    T.to_csv(os.path.join(OUT, "ni_results.csv"), index=False)
    summ = dict(splits=sizes, train_cells=len(full), train_failures=int(full.y.sum()),
                areas=int(full.area.nunique()), families=levels["family"], components=comp)
    with open(os.path.join(OUT, "ni_results.json"), "w") as f:
        json.dump(summ, f, indent=1, default=float)
    print(json.dumps(comp, indent=1, default=float))
    print(T.pivot_table(index="method", columns=["split", "kind"], values=["cover", "units"]).round(3))
    print(T.pivot_table(index="method", columns=["split", "kind"], values="log_score").round(3))


if __name__ == "__main__":
    main()
