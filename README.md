# Spares Provisioning for Shipboard Power Electronics — methodological manuscript

Complete rewrite of the earlier manuscript with the Backblaze drive evaluation
removed in full. No shipboard failure records exist publicly and none are used;
the article is explicitly a methodological contribution.

## Layout

    paper/        LaTeX sources, generated numbers.tex, tables/, figures/,
                  numbers_manifest.csv (provenance of all 81 reported values)
    analysis/     theory_cover.py    analytical: exact cover, kappa limits, margin factor
                  multiproduct.py    analytical: zone total by exact convolution
                  design_sim.py      simulation: campaigns (run first)
                  design_summary.py  simulation: summary + delivered cover
                  real_data_audit.py measured: audit of the four candidate sources
                  glmm.py            hierarchical count-model estimator
    data/         result files, and the three retained capacitor tables
    deliverables/ a_claims_disposition.md      what was removed/kept/reformulated
                  b_title_abstract_contributions.md
                  c_manuscript_NAVAL_SPARES_METHOD.pdf
                  d_float_inventory.md         provenance of every table and figure
                  e_pending_for_validation.md  records and analyses still required

## Reproducing

    cd analysis && python theory_cover.py && python multiproduct.py \
        && python real_data_audit.py && python design_sim.py && python design_summary.py
    cd ../paper && python make_numbers.py && python make_figures.py \
        && latexmk -pdf main.tex

`design_sim.py` takes about 10 minutes on 8 cores; `design_summary.py` a further
few minutes. Everything else runs in under two minutes.

## Provenance rule

Every reported value carries one of three tags in `paper/numbers_manifest.csv`:
ANALYTIC (40), SIM (27), REALDATA (14). There is no tag for a shipboard
measurement, because there is none.
