"""
Figure of the manuscript, built from one result file; its caption states the
provenance.

  fig_zone   ANALYTIC   99% plan for a zone against the number of references,
                        under the two allocations, and the cover of a plan
                        built under the wrong one

    python make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 300})
COL = 3.4          # IEEE single column, inches


def fig_zone():
    M = json.load(open(os.path.join(DATA, "multiproduct.json")))
    P = pd.DataFrame(M["plans"])
    X = pd.DataFrame(M["cross"])
    fig, ax = plt.subplots(2, 1, figsize=(COL, 4.0))
    style = {"A: ship-wide only": ("o-", "ship-wide"),
             "mixed: equal split": ("s--", "equal split"),
             "B: interaction only": ("^-.", "interaction")}
    for k, (st, lab) in style.items():
        d = P[P.structure == k].sort_values("P")
        ax[0].plot(d.P, d.q99, st, lw=1.2, ms=4, label=lab)
    ax[0].set_xlabel("number of references $P$ sharing the zone")
    ax[0].set_ylabel("99\\% plan (units)")
    ax[0].set_title("(a) spares for one zone", fontsize=8)
    ax[0].legend()

    d = X[(X.planned_under == "B: interaction only") &
          (X.truth == "A: ship-wide only")].sort_values("P")
    ax[1].plot(d.P, 100 * d.delivered, "^-.", lw=1.2, ms=4,
               label="interaction plan,\nship-wide truth")
    d = X[(X.planned_under == "A: ship-wide only") &
          (X.truth == "B: interaction only")].sort_values("P")
    ax[1].plot(d.P, 100 * d.delivered, "o-", lw=1.2, ms=4,
               label="ship-wide plan,\ninteraction truth")
    ax[1].axhline(99, color="k", lw=0.6)
    ax[1].set_xlabel("number of references $P$")
    ax[1].set_ylabel("cover delivered (\\%)")
    ax[1].set_title("(b) plan under the wrong structure", fontsize=8)
    ax[1].legend(loc="center left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_zone.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_zone()
    print("figures:", sorted(os.listdir(FIG)))
