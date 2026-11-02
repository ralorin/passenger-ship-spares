# Spares provisioning for power electronics on passenger ships

Code, results and LaTeX sources for the article *Spares Provisioning for Power
Electronics on Passenger Ships from Divided Fleet Records* (P. Lamo and
R. S. Alonso), prepared for IEEE Transactions on Reliability.

No shipboard failure records are used, because none are public. Every number in
the article is analytical, simulated under declared parameters, or measured on a
public street-lighting network, and `paper/numbers_manifest.csv` and
`paper/numbers_v2_manifest.csv` say which for each one.

## Layout

    analysis/   estimators, experiments and SLURM scripts
    data/       result files of the analytical part (Sections IV and V)
    data/v2/    result files of the experiments; raw/ holds downloads and
                per-task outputs and is not tracked
    paper/      LaTeX sources; numbers*.tex and tables/ are generated

### Estimators

| File | Contents |
| --- | --- |
| `glmm.py` | Negative-binomial mixed model with crossed random effects, Laplace approximation; reference implementation |
| `hglmm.py` | Same objective, eliminating the largest diagonal block in closed form at each Newton step; used by every experiment |
| `plan_cover.py` | Exact joint and aggregate cover of a stock vector, cost-minimal vectors, and plans from predictive draws |

### Experiments

| Script | Section | Kind | Output |
| --- | --- | --- | --- |
| `theory_cover.py` | IV | analytical | `data/theory_*` |
| `single_plan.py` | IV | analytical | `data/v2/single_plan*` |
| `multiproduct.py`, `stock_vector.py` | V | analytical | `data/multiproduct*`, `data/stock_vector*` |
| `robust_plan.py` | V | analytical | `data/v2/robust_*` |
| `federated.py` | VI | simulation | `data/v2/federated.*` |
| `campaign.py` | VII | simulation | `data/v2/campaign_*` |
| `fleet.py` (+ `fleet_params.json`) | VIII | simulation | `data/v2/fleet_*` |
| `ni_streetlights_prep.py`, `ni_streetlights.py` | IX | public data | `data/v2/ni_*` |
| `real_data_audit.py` | X | public data | `data/real_data_audit.json` |

`fleet_params.json` lists every fleet parameter with its source in a `_src`
field. `real_data_audit.py` documents the search for public shipboard records
that Section X reports as unsuccessful.

## Reproducing

Python 3.11 with numpy, scipy, pandas and matplotlib, and a TeX distribution with
IEEEtran.

The analytical scripts, `federated.py` and the street-lighting analysis run on a
laptop in minutes to about an hour:

    cd analysis
    python theory_cover.py && python multiproduct.py && python stock_vector.py
    python single_plan.py && python robust_plan.py && python federated.py
    python ni_streetlights_prep.py --inventory 2025
    python ni_streetlights_prep.py --inventory 2026
    python ni_streetlights.py

The street-lighting data are downloaded from data.gov.uk and the DfI asset
portal; the sources and dates are in the docstring of `ni_streetlights_prep.py`.
The fault file for 2025/26 was no longer published when the data were taken, so
the forward test uses the current-year file.

The two large simulations are split into independent tasks. On a SLURM cluster:

    sbatch slurm/campaign.sbatch      # 155 000 campaigns, about 3 h on 28 cores
    sbatch slurm/fleet.sbatch         # 1 400 fleets, about 3 h 40 min on 28 cores
    python campaign.py --summary
    python fleet.py --summary

Each task writes one file per unit of work and skips the ones already written,
so a job stopped by the time limit is completed by submitting it again. Seeds
are fixed per task, and results do not depend on how the tasks are split.

Then, to rebuild the article:

    cd ../paper
    python make_numbers.py && python make_numbers_v2.py && python make_figures.py
    pdflatex main && bibtex main && pdflatex main && pdflatex main

## Provenance tags

| Tag | Meaning |
| --- | --- |
| ANALYTIC | exact calculation or quadrature, no data |
| SIM | simulation under declared parameters |
| PUBLICDATA | measured on the Northern Ireland street-lighting records |

There is no tag for a shipboard measurement.
