"""
Does modelling the dependence between references pay in the simulated fleet?

The fleets of fleet.py are regenerated with the same seeds (same records, same
held-out ships), the pooled and hierarchical models are refitted, and for every
predicted ship-mission two predictive laws are built from the SAME hierarchical
draws:

  joint   the draws as they are, one row per draw of the whole vector;
  perm    each reference column permuted independently, which keeps every
          empirical marginal exactly and breaks the dependence between
          references. Its limit, the product of the empirical marginal
          distribution functions, is solved exactly as well (prod).

Each law gets the same 99 % unit-minimal vector, certified for the draws
(plan_exact.opt_draws, opt_product), and the marginal-level heuristic of
fleet.py for comparison. Comparators that learn field rates without ship
terms (scaled datasheet rate, pooled negative binomial) and the datasheet
benchmark are solved the same way, so the table separates learning the rate
from modelling the dependence.

Where the scenario has ship-wide supply events, the hierarchical draws are
also mixed with an explicit incident: with probability 1 - exp(-rate * nights /
365) a draw adds Binomial(N_p, a) failures to every reference p. The plan with
the generating rate and size is a plan with KNOWN event parameters; rate and
size are also misstated by factors of 0.5 and 2.

On the first STAB_REPS fleets of some scenarios the hierarchical draws are
repeated with a fresh seed at 6 000 and at 24 000 draws, to separate the Monte
Carlo error of selecting the vector from the evaluation error between fleets.

One row per predicted ship-mission is written; summaries are made by
fleet_dep_summary.py.

    python fleet_dep.py --worker I --nworkers N [--hours 3.6] [--reps 60]
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import poisson

import fleet
from fleet import COST, LEVEL, PAR, PRODUCTS, SCENARIOS, SEED, Hier, Pooled, simulate
from plan_cover import plan_joint_draws_robust
from plan_exact import ecdf_log, opt_draws, opt_product

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "v2", "raw", "fleet_dep")
R_DRAWS = 6000
R_BIG = 24000
REPS = 60
STAB_REPS = 15
STAB_SCN = ("clean", "baseline", "heavy_tail", "shocks_frequent")
EVENT_SCN = ("baseline", "shocks", "shocks_frequent", "shocks_severe")
EV_MISSTATE = dict(ev_known=(1, 1), ev_rate_half=(0.5, 1), ev_rate_double=(2, 1),
                   ev_size_half=(1, 0.5), ev_size_double=(1, 2))


def jcover(Y, s):
    return float(np.mean(np.all(Y <= np.asarray(s)[None, :], axis=1)))


def prod_cover(Y, s):
    return float(np.prod([np.mean(Y[:, j] <= s[j]) for j in range(Y.shape[1])]))


def permute(Y, rng):
    return np.column_stack([Y[rng.permutation(len(Y)), j] for j in range(Y.shape[1])])


def opt_prod_draws(Y):
    sm = Y.max(axis=0)
    return opt_product([ecdf_log(Y[:, j], int(sm[j])) for j in range(Y.shape[1])], LEVEL)


def opt_poisson(lam):
    out = []
    for l in lam:
        top = int(poisson.ppf(1 - 1e-12, l)) + 5
        out.append(poisson.logcdf(np.arange(top + 1), l))
    return opt_product(out, LEVEL)


def certified(Y):
    h = plan_joint_draws_robust(Y, LEVEL)
    s, cert, lb, nodes = opt_draws(Y, LEVEL, ub=h)
    return s, h, cert, lb


def counts_per_ref(model, g, Z, rng, col, P):
    Yc = model.predict(g, Z, rng)
    Y = np.zeros((Z.shape[0], P), int)
    for j in range(P):
        Y[:, j] = Yc[:, col == j].sum(axis=1)
    return Y


def one_fleet(name, rep):
    scn = SCENARIOS[name]
    # identical records and held-out ships to fleet.py
    rng = np.random.default_rng([SEED, list(SCENARIOS).index(name), rep])
    d = simulate(scn, rng)
    ships = d.groupby("ship").cls.first()
    new_ships = {int(rng.choice(ships[ships == c].index)) for c in ships.unique()}
    train = d[(~d.ship.isin(new_ships)) & (d.start + d.nights <= scn["train_days"])]
    test = d[d.start >= scn["train_days"]]
    levels = {c: sorted(train[c].unique()) for c in ("product", "zone", "cls", "maker")}
    rng = np.random.default_rng([SEED, list(SCENARIOS).index(name), rep, 7])
    rng_perm = np.random.default_rng([SEED, list(SCENARIOS).index(name), rep, 8])

    t0 = time.time()
    hier, pooled = Hier(train, levels), Pooled(train, levels)
    t_fit = time.time() - t0
    stab = name in STAB_SCN and rep < STAB_REPS
    Zh = hier.draws(R_DRAWS, rng, plugin=False)
    Zp = pooled.draws(R_DRAWS, rng, plugin=False)
    if stab:
        Zh2 = hier.draws(R_DRAWS, rng, plugin=False)
        Zh3 = hier.draws(R_BIG, rng, plugin=False)
    ds_train = (train.hours / train["product"].map(lambda p: PAR["products"][p]["mtbf_h"])).sum()
    factor = float(train.failures.sum() / ds_train)

    rows = []
    for (s, t), g in test.groupby(["ship", "mission"]):
        prods = sorted(g["product"].unique())
        P = len(prods)
        col = np.array([prods.index(p) for p in g["product"]])
        y = g.groupby("product").true_failures.sum().reindex(prods).to_numpy()
        H = g.groupby("product").hours_plan.sum().reindex(prods).to_numpy()
        Ninst = g.groupby("product").N.sum().reindex(prods).to_numpy()
        lam_ds = np.array([H[j] / PAR["products"][p]["mtbf_h"] for j, p in enumerate(prods)])
        cost = COST[[PRODUCTS.index(p) for p in prods]]
        nights = int(g.nights.iloc[0])
        r = dict(scenario=name, rep=rep, ship=int(s), mission=int(t), cls=g.cls.iloc[0],
                 kind="new" if s in new_ships else "known", shock=int(g.shock.iloc[0]),
                 nights=nights, P=P, y_tot=int(y.sum()))

        def rec(key, plan, Y=None):
            plan = np.asarray(plan, int)
            r[f"{key}_units"] = int(plan.sum())
            r[f"{key}_cost"] = float(cost @ plan)
            r[f"{key}_cov"] = int(np.all(y <= plan))
            r[f"{key}_short"] = int(np.maximum(y - plan, 0).sum())
            if Y is not None:
                r[f"{key}_Jdraw"] = jcover(Y, plan)

        rec("mtbf", opt_poisson(lam_ds))
        rec("scaled", opt_poisson(lam_ds * factor))
        Yp = counts_per_ref(pooled, g, Zp, rng, col, P)
        sp, _, cert, _ = certified(Yp)
        rec("pooled", sp, Yp)
        r["pooled_cert"] = int(cert)

        Y = counts_per_ref(hier, g, Zh, rng, col, P)
        Yperm = permute(Y, rng_perm)
        Yperm2 = permute(Y, rng_perm)
        assert all(np.array_equal(np.sort(Y[:, j]), np.sort(Yperm[:, j])) for j in range(P))
        sj, hj, cj, lbj = certified(Y)
        si, hi, ci, lbi = certified(Yperm)
        si2, _, _, _ = certified(Yperm2)
        sprod = opt_prod_draws(Y)
        rec("joint", sj, Y)
        rec("joint_h", hj, Y)
        rec("perm", si, Y)
        rec("perm_h", hi, Y)
        rec("prod", sprod, Y)
        r.update(joint_cert=int(cj), joint_lb=lbj, perm_cert=int(ci), perm_lb=lbi,
                 perm2_units=int(si2.sum()), perm2_same=int(np.array_equal(si, si2)),
                 # dependence left in the permuted draws by finite R, at the plan chosen
                 perm_Jperm=jcover(Yperm, si), perm_Jprod=prod_cover(Y, si),
                 joint_Jprod=prod_cover(Y, sj), prod_Jprod=prod_cover(Y, sprod),
                 marg_same=1)
        rec("marg", [int(np.sort(Y[:, j])[math.ceil(LEVEL * R_DRAWS) - 1]) for j in range(P)], Y)

        if stab:
            Y2 = counts_per_ref(hier, g, Zh2, rng, col, P)
            Y3 = counts_per_ref(hier, g, Zh3, rng, col, P)
            for key, YY in (("s2", Y2), ("big", Y3)):
                a = certified(YY)[0]
                b = certified(permute(YY, rng_perm))[0]
                rec(f"joint_{key}", a, YY)
                rec(f"perm_{key}", b, YY)
                r[f"joint_{key}_same"] = int(np.array_equal(a, sj))
                r[f"perm_{key}_same"] = int(np.array_equal(b, si))
            # honest estimate of the cover of the first plans, on independent draws
            r["joint_Jfresh"] = jcover(Y3, sj)
            r["perm_Jfresh"] = jcover(Y3, si)

        if name in EVENT_SCN:
            ev = rng.random(R_DRAWS)          # common to every misstatement
            for key, (fr, fs) in EV_MISSTATE.items():
                q = -math.expm1(-scn["shock_rate"] * fr * nights / 365.0)
                hit = ev < q
                extra = np.zeros((R_DRAWS, P), int)
                if hit.any():
                    extra[hit] = rng.binomial(np.broadcast_to(Ninst, (int(hit.sum()), P)),
                                              min(1.0, scn["shock_size"] * fs))
                Ye = Y + extra
                se, he, ce, lbe = certified(Ye)
                rec(key, se, Ye)
                r[f"{key}_cert"] = int(ce)
        rows.append(r)
    for r in rows:
        r["fit_seconds"] = t_fit
        r["sigma_hier"] = ";".join(f"{x:.4f}" for x in hier.fit.sigma)
        r["phi_hier"] = float(hier.fit.phi)
        r["factor"] = factor
    return rows


def tasks(reps):
    return [(s, r) for r in range(reps) for s in SCENARIOS]


def worker(i, n, hours, reps):
    os.makedirs(RAW, exist_ok=True)
    t0 = time.time()
    for k, (s, r) in enumerate(tasks(reps)):
        if k % n != i:
            continue
        path = os.path.join(RAW, f"{s}_{r:03d}.csv.gz")
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
        pd.DataFrame(rows).to_csv(path + ".tmp", index=False, compression="gzip")
        os.replace(path + ".tmp", path)
        print(f"{s} {r} {time.time() - t1:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--hours", type=float, default=3.6)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--one", nargs=2)
    a = ap.parse_args()
    if a.one:
        t = time.time()
        rows = one_fleet(a.one[0], int(a.one[1]))
        D = pd.DataFrame(rows)
        print(len(D), "missions", f"{time.time() - t:.0f}s")
        cols = [c for c in D.columns if c.endswith("_cov") or c.endswith("_units")]
        print(D.groupby("kind")[cols].mean().T.round(3).to_string())
        print("cert", D.joint_cert.mean(), D.perm_cert.mean())
    elif a.worker is not None:
        worker(a.worker, a.nworkers, a.hours, a.reps)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
