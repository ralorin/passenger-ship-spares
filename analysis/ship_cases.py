"""
Joint plans for the power electronics of one large passenger ship, exact under
the lognormal-Poisson model, with the sensitivities that matter on board:
horizon to the next real resupply, total latent spread, exposure above plan
and a ship-wide supply event.

Expected failures per reference: installed units x planned powered hours x
datasheet rate x field factor x Arrhenius factor at the zone temperature
(fleet.py, large class, Mediterranean summer, booked occupancy 0.97, units
5 years old). The field factor and the event parameters are assumptions.

A ship-wide event with probability q per mission adds Binomial(N_p, a) failures
to every reference. Given the event state the references are conditionally
independent given u, so the mixture is written as a zone whose lattice holds
both states; branch and bound (stock_exact.exact_joint) applies unchanged, and
since each state's cover is non-decreasing in f, the plan at f = 0 stays the
robust one.

    python ship_cases.py
"""
import json
import math
import os

import numpy as np
from scipy.stats import binom

import fleet
from plan_cover import Zone
from stock_exact import exact_joint, split

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")
LEVEL = 0.99
PAR = fleet.PAR
CLASS = "large"
REGION, SEASON = "mediterranean", "summer"
OCC = 0.97
FIELD = PAR["field_factor"]


def ship():
    c = PAR["classes"][CLASS]
    counts = fleet.zone_counts(c)
    amb = PAR["regions"][REGION][SEASON]
    names, N, rate_day, cost = [], [], [], []
    for p, spec in PAR["products"].items():
        n, r = 0, 0.0
        for zone, per in spec["zones"].items():
            k = counts[zone] * per
            h = fleet.hours_per_day(spec["use"], OCC, amb)
            r += k * h * FIELD / spec["mtbf_h"] * float(fleet.arrhenius(fleet.zone_temp(zone, amb)))
            n += k
        names.append(p)
        N.append(n)
        rate_day.append(r)
        cost.append(spec["cost"])
    return names, np.array(N), np.array(rate_day), np.array(cost)


class Mixed:
    """Zone-like object: no event with weight 1 - q, event with weight q."""

    def __init__(self, z, N, a, q):
        self.P, self.lams, self.smax = z.P, z.lams, z.smax
        H = []
        for p in range(z.P):
            pmf = binom.pmf(np.arange(z.smax + 1), N[p], a)
            dens = np.diff(np.c_[np.zeros((len(z.wu), 1)), z.G[p]], axis=1)
            Hp = np.cumsum(np.array([np.convolve(row, pmf)[:z.smax + 1] for row in dens]), axis=1)
            H.append(Hp)
        self.wu = np.r_[(1 - q) * z.wu, q * z.wu]
        self.G = [np.vstack([G, Hp]) for G, Hp in zip(z.G, H)]

    def joint(self, s):
        prod = np.ones(len(self.wu))
        for p, sp in enumerate(s):
            prod *= self.G[p][:, min(int(sp), self.smax)]
        return float(self.wu @ prod)

    def plan_marginal(self, level):
        return np.array([int(np.argmax(self.wu @ G >= level)) for G in self.G])


def greedy(z, level, cost):
    s = z.plan_marginal(level).copy()
    while z.joint(s) < level:
        base = math.log(max(z.joint(s), 1e-300))
        gain = []
        for p in range(z.P):
            s[p] += 1
            gain.append((math.log(z.joint(s)) - base) / cost[p])
            s[p] -= 1
        s[int(np.argmax(gain))] += 1
    return s


def optimum(z, cost):
    s, _, proved = exact_joint(z, LEVEL, cost, greedy(z, LEVEL, cost), seconds=600)
    return s, proved


def case(days, sigma, rate_day, N, cost, a=0.0005, events_per_year=2.0):
    lams = rate_day * days
    z0, z1 = Zone(lams, *split(0.0, sigma)), Zone(lams, *split(1.0, sigma))
    zhalf = Zone(lams, *split(0.5, sigma))
    marg = z0.plan_marginal(LEVEL)
    rob, pr0 = optimum(z0, cost)
    kn1, pr1 = optimum(z1, cost)
    knh, prh = optimum(zhalf, cost)
    smax = z0.smax + 40
    z0b = Zone(lams, *split(0.0, sigma), smax=smax)
    z1b = Zone(lams, *split(1.0, sigma), smax=smax)
    hi0 = Zone(lams * 1.25, *split(0.0, sigma))
    q = -math.expm1(-events_per_year * days / 365.0)
    ev0 = Mixed(z0b, N, a, 1.0)
    mix0 = Mixed(z0b, N, a, q)
    mix1 = Mixed(z1b, N, a, q)
    rob_ev, prev = optimum(mix0, cost)
    big0 = Mixed(z0b, N, 4 * a, q)
    rob_big, prbig = optimum(big0, cost)
    r = dict(days=days, sigma=sigma, total=float(lams.sum()), lams=lams.round(4).tolist(),
             marg=marg.tolist(), marg_cover_f0=z0.joint(marg), marg_cover_f1=z1.joint(marg),
             marg_units=int(marg.sum()),
             robust=rob.tolist(), robust_units=int(rob.sum()), robust_cost=float(cost @ rob),
             robust_cover_f0=z0.joint(rob), robust_cover_f1=z1.joint(rob),
             know_f1=kn1.tolist(), know_f1_cost=float(cost @ kn1), know_half_cost=float(cost @ knh),
             premium_vs_f1=float(cost @ rob / (cost @ kn1) - 1), premium_vs_half=float(cost @ rob / (cost @ knh) - 1),
             robust_cover_exposure125=hi0.joint(rob),
             event_q=q, robust_cover_given_event=ev0.joint(rob), robust_cover_with_events=mix0.joint(rob),
             event_plan=rob_ev.tolist(), event_plan_units=int(rob_ev.sum()), event_plan_cost=float(cost @ rob_ev),
             event_plan_cover_f1=mix1.joint(rob_ev),
             big_event_plan_units=int(rob_big.sum()), big_event_plan_cost=float(cost @ rob_big),
             proved=bool(pr0 and pr1 and prh and prev and prbig))
    print(days, sigma, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "lams"},
          flush=True)
    return r


def main():
    names, N, rate_day, cost = ship()
    rows = [case(d, s, rate_day, N, cost) for d, s in
            ((7, 0.6), (14, 0.6), (28, 0.6), (7, 0.3), (7, 0.9))]
    info = dict(cls=CLASS, region=REGION, season=SEASON, occupancy=OCC, field_factor=FIELD,
                products=names, installed=N.tolist(), failures_per_day=rate_day.tolist(),
                cost=cost.tolist(), event_share=0.0005, events_per_year=2.0, cases=rows)
    json.dump(info, open(os.path.join(OUT, "ship_cases.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
