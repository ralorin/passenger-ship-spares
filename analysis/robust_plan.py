"""
Stock vectors for a zone when references differ, and the price of not knowing
how the latent spread splits between ship and ship-product terms.

Part 1. Smallest-cost vector (marginal analysis) against a common stock and
against per-reference quantiles, for unequal means and unit costs.

Part 2. With total spread sigma known but the share f = tau^2 / sigma^2
unknown, three plans are compared under every true f:
  known f        plan for the true split (the benchmark)
  conservative   the larger of the plans for f = 0 and f = 1
  averaged       a plan meeting the target on average over f ~ Uniform(0, 1),
                 i.e. under the mixture of the split
Cover is exact (plan_cover.Zone); costs are in units of the cheapest reference.

    python robust_plan.py
"""
import json
import math
import os

import numpy as np
import pandas as pd

from plan_cover import Zone, greedy_joint

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")
os.makedirs(OUT, exist_ok=True)

LEVEL = 0.99
SIGMA = 0.6
F_GRID = np.linspace(0, 1, 11)


def split(f):
    return SIGMA * math.sqrt(f), SIGMA * math.sqrt(1 - f)


def part1():
    rows = []
    for P in (2, 4, 8):
        for total in (2.0, 8.0, 20.0):
            for pattern in ("equal", "skewed"):
                a = np.ones(P) if pattern == "equal" else 2.0 ** np.arange(P)
                lams = total * a / a.sum()
                for costs in ("equal", "skewed"):
                    c = np.ones(P) if costs == "equal" else 5.0 ** (np.arange(P) % 2)
                    for f in (0.0, 0.5, 1.0):
                        z = Zone(lams, *split(f))
                        s_marg = z.plan_marginal(LEVEL)
                        s_com = z.plan_common(LEVEL)
                        s_opt = z.plan_joint(LEVEL, cost=c)
                        rows.append(dict(P=P, total=total, means=pattern, costs=costs, f=f,
                                         cover_marginal=z.joint(s_marg),
                                         cost_marginal=float(c @ s_marg),
                                         cost_common=float(c @ s_com), cover_common=z.joint(s_com),
                                         cost_opt=float(c @ s_opt), cover_opt=z.joint(s_opt),
                                         units_common=int(s_com.sum()), units_opt=int(s_opt.sum())))
    T = pd.DataFrame(rows)
    T["saving_vs_common"] = 1 - T.cost_opt / T.cost_common
    T.to_csv(os.path.join(OUT, "robust_part1.csv"), index=False)
    return T


def part2():
    rows = []
    for P in (2, 4, 8):
        for total in (2.0, 8.0, 20.0):
            lams = np.full(P, total / P)
            zones = {round(f, 2): Zone(lams, *split(f)) for f in F_GRID}
            zs = list(zones.values())

            def mix_joint(s):
                return float(np.mean([z.joint(s) for z in zs]))

            cdf_mix = np.mean([z.total_cdf()[:min(z.smax for z in zs) + 1] for z in zs], axis=0)
            plan = {"conservative": (np.maximum(zones[0.0].plan_joint(LEVEL), zones[1.0].plan_joint(LEVEL)),
                                     max(zones[0.0].plan_aggregate(LEVEL), zones[1.0].plan_aggregate(LEVEL))),
                    "averaged": (greedy_joint(mix_joint, zs[0].plan_marginal(LEVEL), LEVEL),
                                 int(np.argmax(cdf_mix >= LEVEL)))}
            for f, z in zones.items():
                known = (z.plan_joint(LEVEL), z.plan_aggregate(LEVEL))
                for name, (sj, sa) in [("known", known)] + list(plan.items()):
                    rows.append(dict(P=P, total=total, f=f, plan=name,
                                     joint_units=int(sj.sum()), joint_cover=z.joint(sj),
                                     agg_units=int(sa), agg_cover=z.aggregate(sa)))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "robust_part2.csv"), index=False)
    return T


def main():
    T1 = part1()
    T2 = part2()
    s1 = T1[T1.f == 0.5]
    summ = dict(
        part1=dict(
            marginal_cover_min=float(T1.cover_marginal.min()),
            saving_vs_common_max=float(T1.saving_vs_common.max()),
            saving_vs_common_median_skewed=float(
                T1[(T1.means == "skewed") | (T1.costs == "skewed")].saving_vs_common.median()),
            saving_vs_common_median_equal=float(
                T1[(T1.means == "equal") & (T1.costs == "equal")].saving_vs_common.median()),
            example=s1[(s1.P == 8) & (s1.total == 8)].to_dict("records")),
        part2={})
    for (P, total, name), g in T2.groupby(["P", "total", "plan"]):
        known = T2[(T2.P == P) & (T2.total == total) & (T2.plan == "known")].set_index("f")
        gg = g.set_index("f")
        summ["part2"][f"P{P}_L{total:g}_{name}"] = dict(
            joint_min_cover=float(gg.joint_cover.min()),
            joint_extra_units_max=int((gg.joint_units - known.joint_units).max()),
            agg_min_cover=float(gg.agg_cover.min()),
            agg_extra_units_max=int((gg.agg_units - known.agg_units).max()))
    with open(os.path.join(OUT, "robust_plan.json"), "w") as f:
        json.dump(summ, f, indent=1, default=float)
    print(json.dumps(summ["part1"] | {"example": None}, indent=1))
    for k, v in summ["part2"].items():
        if k.startswith("P8") or k.startswith("P4_L8"):
            print(k, v)


if __name__ == "__main__":
    main()
