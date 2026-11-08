"""
Sensitivity of the bootstrap results of Section IV to the three-start rule of
hglmm.Fit.fit, on campaigns of campaign_boot.py rerun on one machine with the
same seeds.

  keep    refits with no accepted run (dropped by the rule) are kept at their
          best run. Same fits, same refits and the same random streams for the
          draws as the rule, so the difference is due to those refits alone.
  polish  every accepted fit and refit is continued with much tighter
          tolerances (hglmm.POLISH). The bootstrap data are simulated from the
          polished estimate, with the same random numbers.

For each campaign: units and exact cover of the predictive and bootstrap plans
under the rule, and under each variant where it applies.

    python rule_checks.py --worker I --nworkers N
    python rule_checks.py --summary          # data/v2/rule_checks.csv.gz, .json
"""
import argparse
import copy
import glob
import json
import os
import time

os.environ["SPARES_RULE"] = "1"

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402

import hglmm                  # noqa: E402
from campaign import fit_all, simulate, truth_zone        # noqa: E402
from campaign_boot import B, SEED, cells, pred_plan       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")
RAW = os.path.join(OUT, "raw", "rule_checks")


def plans_for(full, theta_point, thetas, P, rng, z):
    """Predictive and bootstrap plans from one rng state, scored exactly."""
    r = {}
    g = copy.deepcopy(rng)
    sj, _ = pred_plan([hglmm.at_theta(full, theta_point)], P, g)
    r["pred_units"], r["pred_cover"] = int(sj.sum()), z.joint(sj)
    g = copy.deepcopy(rng)
    sj, _ = pred_plan([hglmm.at_theta(full, th) for th in thetas], P, g)
    r["boot_units"], r["boot_cover"] = int(sj.sum()), z.joint(sj)
    return r


def one(c, rep, polish):
    hglmm.POLISH = polish
    rng = np.random.default_rng([SEED, c["cell"], rep])
    d = simulate(c, rng)
    P = c["P"]
    full, _, _ = fit_all(d, P)
    info = {}
    base = full.theta_rule if polish is False else full.theta
    est = hglmm.at_theta(full, base)
    hglmm.bootstrap_theta(est, B, rng, rule=True, info=info)
    z = truth_zone(c)
    allr = info["all"]
    acc = [(t_rule if not polish else t) for ok, t_rule, t in allr if ok]
    out = dict(cell=c["cell"], rep=rep, refits=len(allr), accepted=len(acc),
               fit_accepted=int(full.accepted))
    tag = "polish" if polish else "rule"
    for k, v in plans_for(est, base, acc, P, rng, z).items():
        out[f"{tag}_{k}"] = v
    if not polish:
        keep = [t_rule for ok, t_rule, t in allr]
        for k, v in plans_for(est, base, keep, P, rng, z).items():
            out[f"keep_{k}"] = v
    return out


def tasks():
    keep = pd.read_csv(os.path.join(OUT, "check_keep_list.csv"))
    pol = pd.read_csv(os.path.join(OUT, "check_polish_list.csv"))
    pairs = set(map(tuple, keep.values.tolist())) | set(map(tuple, pol.values.tolist()))
    polish = set(map(tuple, pol.values.tolist()))
    cl = {c["cell"]: c for c in cells()}
    out = []
    for cell, rep in sorted(pairs):
        out.append((cl[cell], rep, False))
        if (cell, rep) in polish:
            out.append((cl[cell], rep, True))
    return out


def worker(i, n):
    os.makedirs(RAW, exist_ok=True)
    for k, (c, rep, polish) in enumerate(tasks()):
        if k % n != i:
            continue
        path = os.path.join(RAW, f"c{c['cell']:03d}_r{rep:03d}_{'polish' if polish else 'rule'}.json")
        if os.path.exists(path):
            continue
        t0 = time.time()
        r = one(c, rep, polish)
        r["seconds"] = time.time() - t0
        json.dump(r, open(path, "w", encoding="utf-8"), indent=1, default=float)
        print(c["cell"], rep, polish, f"{r['seconds']:.0f}s", flush=True)


def summary():
    rows = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(RAW, "*.json")))]
    A = pd.DataFrame([r for r in rows if "rule_pred_units" in r]).drop(columns="seconds")
    Pz = pd.DataFrame([r for r in rows if "polish_pred_units" in r])
    Pz = Pz[["cell", "rep", "accepted"] + [c for c in Pz.columns if c.startswith("polish_")]]
    D = A.merge(Pz, on=["cell", "rep"], how="left", suffixes=("", "_polish"))
    D.to_csv(os.path.join(OUT, "rule_checks.csv.gz"), index=False)
    print(len(A), "campaigns,", len(Pz), "polished")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary() if a.summary else worker(a.worker, a.nworkers)


if __name__ == "__main__":
    main()
