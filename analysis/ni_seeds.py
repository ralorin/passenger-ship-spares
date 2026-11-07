"""
Street lighting: Monte Carlo error of selecting the 99 % vectors.

The within-period test of ni_validation.py (fit April 2023 - March 2024,
predict the next twelve months; known areas from the full fit, each new area
left out in turn) is repeated with SEEDS independent streams of predictive
draws. The fits do not depend on the seed; only the draws, and so the selected
vectors, do. Writes data/v2/ni_seeds.json.

    python ni_seeds.py [--jobs 7]
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import ni_streetlights as NS
from ni_validation import WITHIN, evaluate, levels_of, select

OUT = NS.OUT
SEED = 20261105
SEEDS = 5
METHODS = ("hier_pred", "nb_pooled", "poisson_jnt")


def one_area(args):
    cells, area, levels = args
    train, test = select(cells, WITHIN[0]), select(cells, WITHIN[1])
    test = test[test.family.isin(train.family.unique())]
    if area is None:
        models, te, kind = NS.fit_all(train, levels), test, "known"
    else:
        models = NS.fit_all(train[train.area != area], levels)
        te, kind = test[test.area == area], "new"
    out = []
    for k in range(SEEDS):
        rng = np.random.default_rng([SEED, k, 0 if area is None else 1 + sorted(
            train.area.unique()).index(area)])
        am, _ = evaluate(models, te, kind, rng, curve=False)
        am = pd.DataFrame(am)
        am["seed"] = k
        out.append(am[am.method.isin(METHODS)])
    return pd.concat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=7)
    a = ap.parse_args()
    cells = pd.read_csv(os.path.join(OUT, "ni_cells_2025.csv"))
    levels = levels_of(cells)
    areas = sorted(select(cells, WITHIN[0]).area.unique())
    with Pool(a.jobs) as pool:
        parts = pool.map(one_area, [(cells, ar, levels) for ar in [None] + areas], chunksize=1)
    A = pd.concat(parts)
    A.to_csv(os.path.join(OUT, "ni_seeds_am.csv.gz"), index=False)
    res = {}
    for (m, kind), g in A.groupby(["method", "kind"]):
        per = g.groupby("seed").agg(units=("units", "mean"), exceed=("covered", lambda c: int((1 - c).sum())),
                                    n=("covered", "size"))
        res[f"{m}|{kind}"] = dict(units=per.units.tolist(), exceed=per.exceed.tolist(),
                                  n=int(per.n.iloc[0]), units_mean=float(per.units.mean()),
                                  units_min=float(per.units.min()), units_max=float(per.units.max()))
    # paired difference predictive minus pooled, per seed
    for kind in ("known", "new"):
        p = A[(A.kind == kind)].pivot_table(index=["seed", "area", "month"], columns="method",
                                            values="units").reset_index()
        d = (p.hier_pred - p.nb_pooled).groupby(p.seed).mean()
        res[f"diff|{kind}"] = d.tolist()
    json.dump(res, open(os.path.join(OUT, "ni_seeds.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
