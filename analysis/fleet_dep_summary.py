"""
Summaries of fleet_dep.py: cover, units and shortfall per plan with intervals
that resample whole fleets, the paired joint-minus-fixed-marginal contrast,
the optimisation gap of the heuristic, the Monte Carlo error of selecting the
vector, and the incident plans on missions with and without an event.

    python fleet_dep_summary.py        -> data/v2/fleet_dep.json, fleet_dep_rows.csv.gz
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")
RAW = os.path.join(OUT, "raw", "fleet_dep")
B = 4000
MAX_REPS = 30
PLANS = ["mtbf", "scaled", "pooled", "marg", "joint", "perm", "prod", "joint_h", "perm_h"]
EV = ["ev_known", "ev_rate_half", "ev_rate_double", "ev_size_half", "ev_size_double"]
rng = np.random.default_rng(20261105)


def ratio_ci(num, den):
    """Ratio of sums over fleets and a cluster-bootstrap 95 % interval.
    num, den: arrays with one entry per fleet."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    est = num.sum() / den.sum()
    idx = rng.integers(0, len(num), (B, len(num)))
    bs = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return [float(est), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def per_fleet(g, cols):
    return g.groupby("rep")[cols].sum()


def plan_stats(g, keys):
    out = {}
    n = g.groupby("rep").size()
    for k in keys:
        if f"{k}_cov" not in g:
            continue
        f = per_fleet(g, [f"{k}_cov", f"{k}_units", f"{k}_short", f"{k}_cost"])
        out[k] = dict(cover=ratio_ci(f[f"{k}_cov"], n), units=ratio_ci(f[f"{k}_units"], n),
                      short=ratio_ci(f[f"{k}_short"], n), cost=float(f[f"{k}_cost"].sum() / n.sum()))
    return out


def paired(g, a, b):
    n = g.groupby("rep").size()
    f = g.assign(dc=g[f"{a}_cov"] - g[f"{b}_cov"], du=g[f"{a}_units"] - g[f"{b}_units"],
                 ds=g[f"{a}_short"] - g[f"{b}_short"]).groupby("rep")[["dc", "du", "ds"]].sum()
    return dict(cover=ratio_ci(f.dc, n), units=ratio_ci(f.du, n), short=ratio_ci(f.ds, n),
                same_vector_share=None)


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.csv.gz")))
    D = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # keep the fleets 0..n-1 finished in every scenario, so scenarios share fleets
    done = D.groupby("rep").scenario.nunique()
    n = 0
    while n in done.index and done[n] == D.scenario.nunique():
        n += 1
    D = D[D.rep < min(n, MAX_REPS)]
    D.to_csv(os.path.join(OUT, "fleet_dep_rows.csv.gz"), index=False)
    res = dict(fleets={s: int(g.rep.nunique()) for s, g in D.groupby("scenario")},
               missions={s: int(len(g)) for s, g in D.groupby("scenario")})
    for scn, gs in D.groupby("scenario"):
        r = {}
        for kind, g in gs.groupby("kind"):
            e = dict(n=int(len(g)), plans=plan_stats(g, PLANS + EV))
            e["joint_minus_perm"] = paired(g, "joint", "perm")
            e["joint_minus_prod"] = paired(g, "joint", "prod")
            e["joint_minus_pooled"] = paired(g, "joint", "pooled")
            e["pooled_minus_scaled"] = paired(g, "pooled", "scaled")
            e["same_joint_perm"] = float(np.mean(g.joint_units == g.perm_units))
            e["units_joint_lt_perm"] = float(np.mean(g.joint_units < g.perm_units))
            e["units_joint_gt_perm"] = float(np.mean(g.joint_units > g.perm_units))
            # optimisation gap of the heuristic, and certification
            e["cert_joint"] = float(g.joint_cert.mean())
            e["cert_perm"] = float(g.perm_cert.mean())
            e["heur_gap_joint"] = float((g.joint_h_units - g.joint_units).mean())
            e["heur_worse_joint"] = float(np.mean(g.joint_h_units > g.joint_units))
            e["heur_gap_perm"] = float((g.perm_h_units - g.perm_units).mean())
            # finite-R dependence left in the permuted draws, at the chosen plan
            e["perm_resid_mean_abs"] = float(np.mean(np.abs(g.perm_Jperm - g.perm_Jprod)))
            e["perm_resid_max_abs"] = float(np.max(np.abs(g.perm_Jperm - g.perm_Jprod)))
            e["perm_seed_same"] = float(g.perm2_same.mean())
            e["perm_seed_units_absdiff"] = float(np.mean(np.abs(g.perm2_units - g.perm_units)))
            # dependence in the joint draws: cover of the fixed-marginal plan under them
            e["perm_Jjoint_mean"] = float(g.perm_Jdraw.mean())
            e["joint_Jprod_mean"] = float(g.joint_Jprod.mean())
            e["joint_Jprod_min"] = float(g.joint_Jprod.min())
            if "joint_s2_units" in g and g.joint_s2_units.notna().any():
                h = g[g.joint_s2_units.notna()]
                e["stab"] = dict(
                    fleets=int(h.rep.nunique()), missions=int(len(h)),
                    same_seed=float(h.joint_s2_same.mean()), same_big=float(h.joint_big_same.mean()),
                    perm_same_seed=float(h.perm_s2_same.mean()), perm_same_big=float(h.perm_big_same.mean()),
                    absdiff_seed=float(np.mean(np.abs(h.joint_s2_units - h.joint_units))),
                    absdiff_big=float(np.mean(np.abs(h.joint_big_units - h.joint_units))),
                    contrast_r6000=paired(h, "joint", "perm"),
                    contrast_seed2=paired(h, "joint_s2", "perm_s2"),
                    contrast_r24000=paired(h, "joint_big", "perm_big"),
                    Jfresh_joint=float(h.joint_Jfresh.mean()), Jfresh_perm=float(h.perm_Jfresh.mean()),
                    Jfresh_joint_p10=float(h.joint_Jfresh.quantile(0.1)))
            if "ev_known_cov" in g and g.ev_known_cov.notna().any():
                ev = {}
                for lab, h in (("all", g), ("event", g[g.shock == 1]), ("no_event", g[g.shock == 0])):
                    if not len(h):
                        continue
                    ev[lab] = dict(n=int(len(h)), fleets=int(h.rep.nunique()),
                                   plans=plan_stats(h, ["joint"] + EV))
                ev["known_minus_omit_all"] = paired(g, "ev_known", "joint")
                ev["cert_known"] = float(g.ev_known_cert.mean())
                e["events"] = ev
            r[kind] = e
        res[scn] = r
    json.dump(res, open(os.path.join(OUT, "fleet_dep.json"), "w"), indent=1)
    for scn in res:
        if scn in ("fleets", "missions"):
            continue
        for kind, e in res[scn].items():
            p = e["plans"]
            jp = e["joint_minus_perm"]
            print(f"{scn:16s} {kind:5s} n={e['n']:6d} joint {100 * p['joint']['cover'][0]:.2f}/"
                  f"{p['joint']['units'][0]:.2f} perm {100 * p['perm']['cover'][0]:.2f}/{p['perm']['units'][0]:.2f}"
                  f"  d cov {100 * jp['cover'][0]:+.2f} [{100 * jp['cover'][1]:+.2f},{100 * jp['cover'][2]:+.2f}]"
                  f" d units {jp['units'][0]:+.3f} [{jp['units'][1]:+.3f},{jp['units'][2]:+.3f}]"
                  f" pooled {100 * p['pooled']['cover'][0]:.2f}/{p['pooled']['units'][0]:.2f}"
                  f" scaled {100 * p['scaled']['cover'][0]:.2f}/{p['scaled']['units'][0]:.2f}"
                  f" mtbf {100 * p['mtbf']['cover'][0]:.2f}")
            if "events" in e:
                for lab in ("event", "no_event"):
                    if lab in e["events"]:
                        q = e["events"][lab]["plans"]
                        print(f"   {lab:8s} n={e['events'][lab]['n']:5d} " + " ".join(
                            f"{k}:{100 * q[k]['cover'][0]:.1f}/{q[k]['units'][0]:.2f}" for k in ["joint"] + EV))
    print(json.dumps({k: res[k] for k in ("fleets", "missions")}))


if __name__ == "__main__":
    main()
