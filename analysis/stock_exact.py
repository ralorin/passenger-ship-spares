"""
Cost-minimal stock vectors under the joint criterion, solved exactly, and the
plans a planner might use instead.

    minimise  c's   subject to  J(s) = Pr(Y_p <= s_p for all p) >= level,  s integer

J is computed exactly (plan_cover.Zone). The problem is solved by branch and
bound: each s_p is at least its own marginal quantile at `level` (necessary,
since J(s) <= Pr(Y_p <= s_p)), the greedy solution gives the first incumbent,
and a branch is cut when even the largest affordable stock of the remaining
references cannot reach the target. The last reference is solved directly.

Plans compared, for the grid of robust_plan.py part 1:
  common      smallest common stock
  equal_risk  every reference at a common marginal level, the smallest that
              meets the joint target
  indep       exact optimum of the problem with the references independent,
              i.e. at a ship-wide share f = 0; feasible for every f (Prop. 3)
  greedy      marginal analysis, largest gain in log J per unit cost
  exact       branch and bound

Part 2 checks, on a fine grid of f, that the joint cover of every plan is
non-decreasing in f, and evaluates the aggregate plans over the whole grid.

    python stock_exact.py
"""
import json
import math
import os
import time

import numpy as np
import pandas as pd

from plan_cover import Zone, greedy_joint

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")
LEVEL = 0.99
SIGMA = 0.6
LIMIT = 600          # seconds of branch and bound per problem


def split(f, sigma=SIGMA):
    return sigma * math.sqrt(f), sigma * math.sqrt(1 - f)


class TimeUp(Exception):
    pass


def exact_joint(z, level, cost, incumbent, seconds=None):
    """Returns (best vector, nodes, proved). proved is False when the time ran out."""
    P = z.P
    t_end = None if seconds is None else time.time() + seconds
    cost = np.asarray(cost, float)
    lo = z.plan_marginal(level)
    best = np.array(incumbent, int)
    best_cost = float(cost @ best)
    nodes = 0
    base = float(cost @ lo)

    def ub(p, spent, rest_min):
        return min(z.smax, lo[p] + int(math.floor((best_cost - spent - rest_min) / cost[p] + 1e-9)))

    rest = np.r_[np.cumsum((cost * lo)[::-1])[::-1][1:], 0.0]   # min cost of references after p
    # J is symmetric in references with equal mean and cost: keep their stocks non-increasing
    same = [p > 0 and z.lams[p] == z.lams[p - 1] and cost[p] == cost[p - 1] for p in range(P)]

    def dfs(p, prefix, spent, s):
        nonlocal best, best_cost, nodes
        nodes += 1
        if t_end is not None and nodes % 2000 == 0 and time.time() > t_end:
            raise TimeUp
        if p == P - 1:
            top = ub(p, spent, 0.0)
            if same[p]:
                top = min(top, s[-1])
            if top < lo[p]:
                return
            vals = (z.wu * prefix) @ z.G[p][:, lo[p]:top + 1]
            ok = np.flatnonzero(vals >= level)
            if ok.size:
                sp = lo[p] + ok[0]
                c = spent + cost[p] * sp
                if c < best_cost - 1e-9:
                    best_cost, best = c, np.r_[s, sp].astype(int)
            return
        top = ub(p, spent, rest[p])
        if same[p]:
            top = min(top, s[-1])
        if top < lo[p]:
            return
        later = range(p + 1, P)
        for sp in range(lo[p], top + 1):
            new = prefix * z.G[p][:, sp]
            spent2 = spent + cost[p] * sp
            # optimistic cover: each later reference at the largest stock the budget
            # allows while all the others stay at their lower bounds
            tail = np.ones(len(z.wu))
            for q in later:
                tail *= z.G[q][:, ub(q, spent2, rest[p] - cost[q] * lo[q])]
            if float(z.wu @ (new * tail)) < level:
                continue
            dfs(p + 1, new, spent2, s + [sp])

    try:
        dfs(0, np.ones(len(z.wu)), 0.0, [])
        proved = True
    except TimeUp:
        proved = False
    return best, nodes, proved


def lower_bound(z, level, cost):
    """Valid lower bound on the optimum of the joint problem.

    By Hoelder, J(s) <= prod_p (E_u G_p(s_p|u)^P)^(1/P), so the separable problem
    min c's subject to sum_p h_p(s_p) >= log(level), h_p = log(E_u G_p^P)/P, is a
    relaxation; its Lagrangian dual bounds it from below."""
    P = z.P
    cost = np.asarray(cost, float)
    h = [np.log(np.maximum(z.wu @ (G ** P), 1e-300)) / P for G in z.G]
    s = np.arange(z.smax + 1)
    best = -np.inf
    for mu in np.exp(np.linspace(np.log(1e-2), np.log(1e5), 400)):
        val = mu * math.log(level)
        for p in range(P):
            val += np.min(cost[p] * s - mu * h[p])
        best = max(best, val)
    return float(best)


def equal_risk(z, level):
    """Common marginal level for all references, smallest meeting the joint target."""
    cdfs = [z.wu @ G for G in z.G]

    def plan(a):
        return np.array([int(np.argmax(c >= a)) if (c >= a).any() else z.smax for c in cdfs])

    lo, hi = level, 1.0 - 1e-12
    if z.joint(plan(lo)) >= level:
        return plan(lo)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if z.joint(plan(mid)) >= level:
            hi = mid
        else:
            lo = mid
    return plan(hi)


def one_case(args):
    P, total, means, costs, f = args
    a = np.ones(P) if means == "equal" else 2.0 ** np.arange(P)
    lams = total * a / a.sum()
    c = np.ones(P) if costs == "equal" else 5.0 ** (np.arange(P) % 2)
    z0 = Zone(lams, *split(0.0))
    z = Zone(lams, *split(f))
    t0 = time.time()
    plans = dict(common=z.plan_common(LEVEL), equal_risk=equal_risk(z, LEVEL),
                 greedy=z.plan_joint(LEVEL, cost=c))
    ex, nodes, proved = exact_joint(z, LEVEL, c, plans["greedy"], seconds=LIMIT)
    plans["exact"] = ex
    ind, _, proved0 = exact_joint(z0, LEVEL, c, z0.plan_joint(LEVEL, cost=c), seconds=LIMIT)
    plans["indep"] = ind
    r = dict(P=P, total=total, means=means, costs=costs, f=f, nodes=nodes, proved=proved,
             proved_indep=proved0, lower_bound=lower_bound(z, LEVEL, c) if not proved else float(c @ ex),
             seconds=time.time() - t0)
    for k, s in plans.items():
        r[f"cost_{k}"] = float(c @ s)
        r[f"units_{k}"] = int(s.sum())
        r[f"cover_{k}"] = z.joint(s)
        r[f"plan_{k}"] = " ".join(map(str, s))
    print(P, total, means, costs, f, nodes, f"{r['seconds']:.1f}s", flush=True)
    return r


def part1(workers=4):
    from multiprocessing import Pool
    cases = [(P, total, means, costs, f) for P in (2, 4, 8) for total in (2.0, 8.0, 20.0)
             for means in ("equal", "skewed") for costs in ("equal", "skewed") for f in (0.0, 0.5, 1.0)]
    with Pool(workers) as pool:
        rows = pool.map(one_case, sorted(cases, key=lambda x: -x[0] * x[1]), chunksize=1)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "stock_exact_part1.csv"), index=False)
    return T


def part2():
    """Monotonicity in f of the joint cover, and the aggregate over the whole f grid."""
    fgrid = np.linspace(0, 1, 101)
    rows, mono = [], []
    for P in (2, 4, 8):
        for total in (2.0, 8.0, 20.0):
            for means in ("equal", "skewed"):
                a = np.ones(P) if means == "equal" else 2.0 ** np.arange(P)
                lams = total * a / a.sum()
                zs = [Zone(lams, *split(f)) for f in fgrid]
                cands = {"f0": zs[0].plan_joint(LEVEL), "f1": zs[-1].plan_joint(LEVEL),
                         "f05": zs[50].plan_joint(LEVEL), "marg": zs[0].plan_marginal(LEVEL)}
                for k, s in cands.items():
                    J = np.array([z.joint(s) for z in zs])
                    mono.append(dict(P=P, total=total, means=means, plan=k,
                                     min_step=float(np.min(np.diff(J))), J0=J[0], J1=J[-1]))
                aggs = np.array([z.plan_aggregate(LEVEL) for z in zs])
                cover_max = np.array([z.aggregate(aggs.max()) for z in zs])
                cover_end = np.array([z.aggregate(max(aggs[0], aggs[-1])) for z in zs])
                rows.append(dict(P=P, total=total, means=means, agg_f0=int(aggs[0]),
                                 agg_f1=int(aggs[-1]), agg_max=int(aggs.max()),
                                 argmax_f=float(fgrid[int(np.argmax(aggs))]),
                                 agg_nondecreasing=bool(np.all(np.diff(aggs) >= 0)),
                                 cover_rob_min=float(cover_max.min()),
                                 cover_end_min=float(cover_end.min())))
    M = pd.DataFrame(mono)
    A = pd.DataFrame(rows)
    M.to_csv(os.path.join(OUT, "stock_exact_mono.csv"), index=False)
    A.to_csv(os.path.join(OUT, "stock_exact_agg.csv"), index=False)
    return M, A


def part3():
    """Total spread unknown as well: plans for sigma = 0.6 under other sigmas."""
    rows = []
    for P in (4, 8):
        for total in (8.0, 20.0):
            lams = np.full(P, total / P)
            plan0 = Zone(lams, *split(0.0)).plan_joint(LEVEL)
            agg1 = Zone(lams, *split(1.0)).plan_aggregate(LEVEL)
            for sig in (0.3, 0.45, 0.6, 0.75, 0.9):
                for f in (0.0, 1.0):
                    z = Zone(lams, *split(f, sig))
                    rows.append(dict(P=P, total=total, sigma=sig, f=f,
                                     joint_cover=z.joint(plan0), agg_cover=z.aggregate(agg1),
                                     joint_units_needed=int(z.plan_joint(LEVEL).sum()),
                                     joint_units_plan=int(plan0.sum()),
                                     agg_needed=int(z.plan_aggregate(LEVEL)), agg_plan=int(agg1)))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "stock_exact_sigma.csv"), index=False)
    return T


def main():
    T = part1(3)
    M, A = part2()
    S = part3()
    diff = (T.cost_greedy - T.cost_exact) / T.cost_exact
    uneq = (T.means == "skewed") | (T.costs == "skewed")
    summ = dict(
        cases=len(T), proved=int(T.proved.sum()), proved_indep=int(T.proved_indep.sum()),
        gap_unproved_max=float(((T.cost_exact - T.lower_bound) / T.lower_bound)[~T.proved].max())
        if (~T.proved).any() else 0.0,
        greedy_optimal=int((diff < 1e-9).sum()),
        greedy_gap_max=float(diff.max()), greedy_gap_mean=float(diff.mean()),
        exact_feasible=bool((T.cover_exact >= LEVEL).all()),
        save_vs_common_median_uneq=float((1 - T.cost_exact / T.cost_common)[uneq].median()),
        save_vs_common_max=float((1 - T.cost_exact / T.cost_common).max()),
        save_vs_common_median_eq=float((1 - T.cost_exact / T.cost_common)[~uneq].median()),
        save_vs_equal_risk_median_uneq=float((1 - T.cost_exact / T.cost_equal_risk)[uneq].median()),
        save_vs_equal_risk_max=float((1 - T.cost_exact / T.cost_equal_risk).max()),
        save_vs_equal_risk_median_eq=float((1 - T.cost_exact / T.cost_equal_risk)[~uneq].median()),
        indep_extra_vs_exact_median=float((T.cost_indep / T.cost_exact - 1)[T.f > 0].median()),
        indep_extra_vs_exact_max=float((T.cost_indep / T.cost_exact - 1).max()),
        indep_cover_min=float(T.cover_indep.min()),
        mono_min_step=float(M.min_step.min()), mono_cases=len(M),
        agg_argmax_f=sorted(set(A.argmax_f.tolist())),
        agg_nondecreasing_all=bool(A.agg_nondecreasing.all()),
        agg_rob_cover_min=float(A.cover_rob_min.min()),
        agg_end_cover_min=float(A.cover_end_min.min()),
        sigma_joint_cover_min=float(S.joint_cover.min()),
        sigma_agg_cover_min=float(S.agg_cover.min()),
        max_nodes=int(T.nodes.max()), max_seconds=float(T.seconds.max()))
    with open(os.path.join(OUT, "stock_exact.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
