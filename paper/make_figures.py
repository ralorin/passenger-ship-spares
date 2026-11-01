"""
Figures of the manuscript. Each figure is built from one result file and its
caption in the manuscript states its provenance.

  fig1_cover        ANALYTIC   exact cover against lambda for several sigma,
                               with the closed form overlaid, showing that a
                               single kappa does not fix the cover
  fig2_multiproduct ANALYTIC   99% plan for a zone against the number of
                               products, under the two allocations
  fig3_design       SIM        recovery of the allocation against campaign size

    python make_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "analysis"))

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 300})
COL = 3.4          # IEEE single column, inches


def fig1():
    from theory_cover import exact_cover, plan_quantile
    lam = np.concatenate([np.linspace(0.1, 2, 16), np.linspace(2.4, 30, 30),
                          np.linspace(32, 400, 34)])
    NF = 601        # figure-resolution quadrature; verified against 4001 below
    fig, ax = plt.subplots(2, 1, figsize=(COL, 4.2))

    # (a) exact cover against lambda, one curve per sigma
    for s, st in ((0.1, "-"), (0.3, "--"), (0.6, "-."), (1.0, ":")):
        q = plan_quantile(lam, 99)
        ex = exact_cover(q, lam, s, NF)
        ax[0].semilogx(lam, 100 * ex, st, lw=1.2, label=f"$\\sigma={s}$")
    ax[0].axhline(99, color="k", lw=0.6)
    ax[0].set_xlabel("expected failures $\\lambda$")
    ax[0].set_ylabel("delivered cover of a 99\\% plan (\\%)")
    ax[0].set_title("(a) exact cover", fontsize=8)
    ax[0].legend(loc="lower left")
    ax[0].set_ylim(80, 100.5)

    # (b) along curves of constant kappa: exact spreads, closed form does not
    for kap, st in ((0.5, "-"), (1.0, "--"), (2.0, "-.")):
        s = kap / np.sqrt(lam)
        ex = np.array([float(exact_cover(plan_quantile(l, 99), l, si, NF)[0])
                       for l, si in zip(lam, s)])
        ax[1].semilogx(lam, 100 * ex, st, lw=1.2, label=f"exact, $\\kappa={kap}$")
        rule = norm.cdf(norm.ppf(0.99) / np.sqrt(1 + kap ** 2))
        ax[1].axhline(100 * rule, color="k", lw=0.6, ls=st, alpha=0.55)
    ax[1].set_xlabel("expected failures $\\lambda$ (with $\\sigma=\\kappa/\\sqrt{\\lambda}$)")
    ax[1].set_ylabel("delivered cover (\\%)")
    ax[1].set_title("(b) constant $\\kappa$", fontsize=8)
    ax[1].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_cover.pdf"))
    plt.close(fig)
    # the figure grid must not misrepresent the exact curve: check it against the
    # manuscript's quadrature at a few points
    worst = max(abs(float(exact_cover(plan_quantile(l, 99), l, sg, NF)[0])
                    - float(exact_cover(plan_quantile(l, 99), l, sg)[0]))
                for l in (0.5, 5.0, 50.0, 400.0) for sg in (0.1, 0.6, 1.0))
    print(f"fig1 quadrature check: max deviation {worst:.2e}")


def fig2():
    P = pd.DataFrame(json.load(open(os.path.join(DATA, "multiproduct.json")))["plans"])
    X = pd.DataFrame(json.load(open(os.path.join(DATA, "multiproduct.json")))["cross"])
    fig, ax = plt.subplots(2, 1, figsize=(COL, 4.0))
    style = {"A: ship-wide only": ("o-", "ship-wide"),
             "mixed: equal split": ("s--", "equal split"),
             "B: interaction only": ("^-.", "interaction")}
    for k, (st, lab) in style.items():
        d = P[P.structure == k].sort_values("P")
        ax[0].plot(d.P, d.q99, st, lw=1.2, ms=4, label=lab)
    ax[0].set_xlabel("number of products $P$ sharing the zone")
    ax[0].set_ylabel("99\\% plan (units)")
    ax[0].set_title("(a) spares for one zone", fontsize=8)
    ax[0].legend()

    d = X[(X.planned_under == "B: interaction only") &
          (X.truth == "A: ship-wide only")].sort_values("P")
    ax[1].plot(d.P, 100 * d.delivered, "^-.", lw=1.2, ms=4,
               label="interaction plan,\nship-wide truth")
    d2 = X[(X.planned_under == "A: ship-wide only") &
           (X.truth == "B: interaction only")].sort_values("P")
    ax[1].plot(d2.P, 100 * d2.delivered, "o-", lw=1.2, ms=4,
               label="ship-wide plan,\ninteraction truth")
    ax[1].axhline(99, color="k", lw=0.6)
    ax[1].set_xlabel("number of products $P$")
    ax[1].set_ylabel("cover delivered (\\%)")
    ax[1].set_title("(b) plan under the wrong structure", fontsize=8)
    ax[1].legend(loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_multiproduct.pdf"))
    plt.close(fig)


def fig3():
    path = os.path.join(DATA, "design_sim.json")
    if not os.path.exists(path):
        print("design_sim.json missing; skipping fig3")
        return
    S = pd.DataFrame(json.load(open(path))["summary"])
    fig, ax = plt.subplots(1, 2, figsize=(COL * 2, 2.2))
    d = S.dropna(subset=["allocation_recovered"])
    for (M, m), g in d[d.m_exp == 1.0].groupby(["M", "m_exp"]):
        g = g.sort_values("S")
        ax[0].plot(g.S, 100 * g.allocation_recovered, "o-", lw=1.2, ms=4,
                   label=f"{int(M)} mission" + ("s" if M > 1 else ""))
    ax[0].axhline(50, color="k", lw=0.6, ls=":")
    ax[0].set_xlabel("ships in the campaign")
    ax[0].set_ylabel("allocation recovered (\\%)")
    ax[0].set_title("(a) telling $\\tau$ from $\\omega$", fontsize=8)
    ax[0].legend()

    e = S[(S.tau > 0) & (S.omega > 0) & (S.m_exp == 1.0)].sort_values(["M", "S"])
    for M, g in e.groupby("M"):
        ax[1].plot(g.S, 100 * g.cover_mean, "o-", lw=1.2, ms=4,
                   label=f"{int(M)} missions, mean")
        ax[1].plot(g.S, 100 * g.cover_p10, "s--", lw=1.0, ms=3, alpha=0.7,
                   label=f"{int(M)} missions, worst tenth")
    ax[1].axhline(99, color="k", lw=0.6)
    ax[1].set_xlabel("ships in the campaign")
    ax[1].set_ylabel("cover of the 99\\% plan (\\%)")
    ax[1].set_title("(b) plan for a new ship", fontsize=8)
    ax[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_design.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    print("fig1_cover.pdf")
    fig2()
    print("fig2_multiproduct.pdf")
    fig3()
    print("done:", sorted(os.listdir(FIG)))
