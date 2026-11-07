# Spares provisioning for power electronics on passenger ships

Code, results and LaTeX sources for the article *Ship-Wide and Ship–Product
Heterogeneity in Joint Spares Provisioning for Passenger-Ship Power Electronics*
(P. Lamo and R. S. Alonso), prepared for IEEE Transactions on Reliability.

No shipboard failure records are used: no public record with failures per unit and exposure was found for these products. Every number in
the article is analytical, simulated under declared parameters, or measured on a
public street-lighting network, and `paper/numbers*_manifest.csv` say which for
each one.

## Layout

    analysis/   estimators, experiments and SLURM scripts
    data/       result files of the analytical part
    data/v2/    result files of the experiments; raw/ holds downloads and
                per-task outputs and is not tracked
    paper/      LaTeX sources; numbers*.tex and tables/ are generated

### Estimators

| File | Contents |
| --- | --- |
| `glmm.py` | Negative-binomial mixed model with crossed random effects, Laplace approximation; reference implementation |
| `hglmm.py` | Same objective, eliminating the largest diagonal block at each Newton step; parametric bootstrap of the variance parameters (`bootstrap_theta`, `at_theta`) |
| `plan_cover.py` | Exact joint and aggregate cover of a stock vector, marginal analysis, and plans from predictive draws |

### Experiments

| Script | Section | Kind | Output |
| --- | --- | --- | --- |
| `single_plan.py`, `multiproduct.py`, `stock_vector.py`, `robust_plan.py` | III | analytical | `data/multiproduct*`, `data/v2/single_plan*`, `data/v2/robust_*` |
| `stock_exact.py` | III | analytical | `data/v2/stock_exact*` (branch and bound, Hölder bound, monotonicity in f) |
| `stock_f0_dp.py` | III | analytical | `data/v2/stock_f0_dp.json` (searches not finished at f = 0, solved exactly by dynamic programming) |
| `ship_cases.py` | III-D | analytical | `data/v2/ship_cases.json` (one large ship: horizon, spread, exposure, supply events) |
| `bridge.py` | V-C | analytical and simulation | `data/v2/bridge.json` (large ship, marginals fixed per horizon of 7 to 56 days: dependent against independence plans, exact and from permuted draws, 100 seeds) |
| `federated.py`, `fed_predictive.py` | (not in the article) | simulation | `data/v2/federated.*`, `data/v2/fed_predictive.*` |
| `campaign.py`, `campaign_extra.py` | IV | simulation | `data/v2/campaign_*` |
| `campaign_boot.py` | IV | simulation | `data/v2/campaign_boot_*` (40 campaigns per cell, 20 refits each) |
| `campaign_bconv.py` | IV | simulation | `data/v2/campaign_bconv*` (20, 40 and 80 refits; redrawn predictive samples) |
| `campaign_reg.py` | IV | simulation | `data/v2/campaign_reg_reps.csv.gz` (small campaigns: fixed, bootstrapped, capped and half-normal penalised spreads; trace of extreme plans) |
| `campaign_diag.py` | IV | simulation | `data/v2/campaign_diag_{hpc,local}_*.json` (extreme bootstrap plans followed step by step on two machines; bootstrap over fresh streams at a fixed estimate) |
| `campaign_starts.py` | IV | simulation | `data/v2/campaign_starts.json` (main-grid campaigns fitted from one start and from three; higher maxima and changed test decisions) |
| `fleet.py` (+ `fleet_params.json`), `fleet_extra.py` | V | simulation | `data/v2/fleet_v4_*`, `data/v2/fleet_extra.json` (earlier fleet study) |
| `fleet_dep.py`, `fleet_dep_summary.py` (+ `plan_exact.py`) | V | simulation | `data/v2/fleet_dep*` (joint against fixed-marginal plans from the same draws, certified unit-minimal vectors, incident plans, seed and draw-count checks) |
| `ni_streetlights_prep.py`, `ni_streetlights.py`, `ni_validation.py`, `ni_audit.py`, `ni_seeds.py` | VI | public data | `data/v2/ni_*` |
| `real_data_audit.py` | VII | public data | `data/real_data_audit.json` |

With `SPARES_RULE=1` in the environment, `campaign_boot.py`, `campaign_reg.py`
and `campaign_bconv.py` fit every point estimate and refit with the three-start
rule of `hglmm.Fit.fit(rule=True)`: the criterion is maximised from the given
start, the default one and small spreads; a run is accepted when its largest
projected gradient is at most 0.05; the accepted run with the largest criterion
is kept, and a refit with none accepted is dropped. These runs write `*_rule*`
files, which Section IV of the article uses; the files without the suffix are
the first run, from one start, kept for the comparison in Section IV-D.

`fleet_params.json` lists every fleet parameter with its source in a `_src`
field. Section V of the article uses `fleet_dep.py`, which regenerates the
fleets of `fleet.py` with the same seeds; `fleet_dep_summary.py` keeps the
first 30 fleets per scenario, all finished in every scenario.

## Reproducing

Python 3.10 or 3.11 with numpy 2.2, scipy 1.15, pandas 2.3 and matplotlib, and a
TeX distribution with IEEEtran.

Analytical scripts, the federated checks and the street-lighting analysis run on
a laptop (`stock_exact.py` about 2 h on three cores; `ni_validation.py` about
1.5 h):

    cd analysis
    python multiproduct.py && python stock_vector.py && python single_plan.py
    python robust_plan.py && python stock_exact.py
    python federated.py && python fed_predictive.py
    python ni_streetlights_prep.py --inventory 2025
    python ni_streetlights_prep.py --inventory 2026
    python ni_streetlights.py && python ni_validation.py && python ni_audit.py
    python ship_cases.py && python campaign_bconv.py && python ni_seeds.py
    python bridge.py                    # about 5 min
    python stock_f0_dp.py               # seconds
    python campaign_starts.py --reps 10 --workers 6

The street-lighting data are downloaded from data.gov.uk and the DfI asset
portal; the sources and dates are in the docstring of `ni_streetlights_prep.py`.

The large simulations are split into independent tasks. On a SLURM cluster:

    sbatch slurm/campaign.sbatch        # 155 000 campaigns, about 3 h on 28 cores
    sbatch slurm/fleet.sbatch           # 900 fleets, about 3.7 min each per worker
    sbatch slurm/campaign_boot.sbatch   # 1 280 campaigns, about 1 min each
    python campaign.py --summary && python campaign_extra.py
    python fleet.py --summary && python fleet_extra.py
    python campaign_boot.py --summary
    sbatch slurm/reg.sbatch             # 480 small campaigns, about 30 s each
    sbatch slurm/dep.sbatch             # fleets of fleet_dep.py, about 4 min each
    python fleet_dep_summary.py
    sbatch slurm/rule.sbatch            # Tables III and IV campaigns with the convergence rule
    sbatch slurm/rule_bconv.sbatch      # 20 against 80 refits with the rule
    sbatch slurm/diag.sbatch            # campaign_diag.py on the cluster
    SPARES_RULE=1 python campaign_boot.py --summary
    SPARES_RULE=1 python campaign_reg.py --summary

`campaign_boot.py --worker I --nworkers N` also runs without SLURM. Each task
writes one file per unit of work and skips the ones already written, so a job
stopped by the time limit is completed by submitting it again. Seeds are fixed
per task (NumPy PCG64), and results do not depend on how tasks are split. Each
machine reproduces its own results exactly, but they can differ between
machines: the point estimates of the smallest campaigns differ in the third or
fourth digit where the criterion is flat, the refits then diverge, and the most
extreme bootstrap plans change. `campaign_diag.py` shows that these extremes
also vary by orders of magnitude between bootstrap streams on one machine.

Then, to rebuild the article (`make_numbers_v7.py` runs v6, which runs v5):

    cd ../paper
    python make_numbers.py && python make_numbers_v2.py && python make_numbers_v3.py && python make_numbers_v4.py && python make_numbers_v7.py
    pdflatex main && bibtex main && pdflatex main && pdflatex main

## Provenance tags

| Tag | Meaning |
| --- | --- |
| ANALYTIC | exact calculation or quadrature, no data |
| SIM | simulation under declared parameters |
| PUBLICDATA | measured on the Northern Ireland street-lighting records |

There is no tag for a shipboard measurement.
