"""
Audit of the power-electronics evidence available, and of what each candidate
source can support.

Backs the statement in the article that no shipboard failure records were
available. For every source examined it records provenance, structure and the
reason it can or cannot test the provisioning decision. The decision needs, per
population, an installed count, an exposure, a failure count over a bounded
horizon and a data-holder label; a source missing any of these cannot test it.

Sources examined
  1. NASA PCoE "Capacitor Electrical Stress" (in hand, tabulated). Aluminium
     electrolytic capacitors under raised DC bias; ESR and capacitance measured
     by impedance sweep. Real power-electronic components.
  2. NASA PCoE "IGBT Accelerated Aging" (downloaded and inspected, discarded).
     Six devices, electrical-characterisation waveforms, no fleet structure.
  3. OEDI / NREL "Extreme Weather Impacts on Utility-Scale PV Plant
     Performance" (downloaded and inspected, discarded). 174 anonymised plants,
     daily rows; weather-event O&M tickets and performance ratio, but no
     inverter replacement counts, no installed inverter population and no
     exposure.
  4. Shipboard power-electronics field records: none found in the public
     domain.

Only source 1 is used in the article, and only to characterise the degradation
mechanism and the censoring it produces, not to estimate a failure rate, a
temperature effect, a coverage or a spares count.

    python real_data_audit.py
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

REQUIRED = ["installed count per population", "exposure per population",
            "failure count over a bounded horizon", "data-holder label"]

SOURCES = [
    dict(key="capacitor_es",
         name="NASA PCoE, Capacitor Electrical Stress",
         provenance="NASA Prognostics Center of Excellence data repository; raw "
                    "MATLAB v7.3 impedance-sweep files, read directly (no figure "
                    "digitisation).",
         device="Aluminium electrolytic capacitors, nominally 2000 uF",
         environment="Laboratory, raised DC bias at 10/12/14 V, room ambient",
         has=["exposure per population"],
         lacks=["installed count per population", "failure count over a bounded horizon",
                "data-holder label"],
         used_for="Failure mechanism (ESR growth) and the censoring it produces. "
                  "Not used for any rate, coefficient, coverage or spares count.",
         verdict="retained, context only"),
    dict(key="igbt_aging",
         name="NASA PCoE, IGBT Accelerated Aging",
         provenance="https://phm-datasets.s3.amazonaws.com/NASA/8.+IGBT+Accelerated"
                    "+Aging.zip (230 MB), downloaded and unpacked 2026-09-27.",
         device="IGBT IRG4BC30K",
         environment="Laboratory, thermal overstress, DC and square-wave gate bias",
         has=[],
         lacks=REQUIRED,
         used_for="Not used.",
         verdict="discarded: six devices, waveform-level characterisation, no "
                 "populations, no exposure in service and no data holders"),
    dict(key="oedi_pv_om",
         name="OEDI/NREL, Extreme Weather Impacts on Utility-Scale PV Plant "
              "Performance",
         provenance="https://data.openei.org/files/4055/oedi_data.xlsx, CC-BY 4.0, "
                    "downloaded 2026-09-27; 51 504 plant-days, 174 anonymised plants, "
                    "2018-02-22 to 2020-03-01.",
         device="Utility-scale PV plants (inverters not individually identified)",
         environment="Field, continental United States",
         has=["data-holder label"],
         lacks=["installed count per population", "exposure per population",
                "failure count over a bounded horizon"],
         used_for="Not used.",
         verdict="discarded: the O&M fields are weather-event tickets and an "
                 "affected-asset TYPE, not replacement counts; no inverter "
                 "population and no exposure",
         ),
    dict(key="shipboard",
         name="Shipboard power-electronics field failure records",
         provenance="Searched: no public source located. Port-state-control "
                    "deficiency databases record inspection findings per ship but "
                    "no installed populations or exposure; marine casualty "
                    "databases record incidents without denominators.",
         device="Converters, inverters, chargers, power supplies on board",
         environment="Passenger ships in service",
         has=[],
         lacks=REQUIRED,
         used_for="Not used: none available.",
         verdict="unavailable"),
]


def capacitor_summary():
    """Summary of the capacitor data, the one source retained.

    The per-unit `censored` flag shipped with the tabulation is not used. It
    marks a unit as failed as soon as one impedance sweep reaches 280% of the
    initial ESR; in this data set every such crossing happens in the first few
    sweeps and is followed by a decline, so the flag records a transient, not a
    crossing sustained to end of life. The trajectories are recomputed here and
    both readings are reported.
    """
    u = pd.read_csv(os.path.join(DATA, "capacitor_units.csv"))
    s = pd.read_csv(os.path.join(DATA, "capacitor_sweeps.csv"))
    meta = json.load(open(os.path.join(DATA, "capacitor_meta.json")))
    crit = float(meta.get("esr_failure_ratio", 2.8))

    rows = []
    for unit, g in s.groupby("unit"):
        g = g.sort_values("sweep")
        e0 = float(g.esr_ohm.iloc[:3].median())
        r = (g.esr_ohm / e0).to_numpy(float)
        over = r >= crit
        sustained = bool((np.convolve(over.astype(int), np.ones(2, int), "valid") >= 2).any())
        rows.append(dict(unit=unit, stress_V=int(g.stress_V.iloc[0]),
                         hours=float(g.hours.max()), esr0=e0,
                         max_ratio=float(r.max()),
                         sweep_of_max=int(g.sweep.iloc[int(r.argmax())]),
                         final_ratio=float(r[-1]),
                         any_crossing=bool(over.any()), n_sweeps_over=int(over.sum()),
                         sustained_crossing=sustained,
                         ends_above_criterion=bool(r[-1] >= crit)))
    U = pd.DataFrame(rows)
    by = (U.groupby("stress_V")
            .agg(units=("unit", "size"),
                 max_ratio_mean=("max_ratio", "mean"),
                 final_ratio_mean=("final_ratio", "mean"),
                 final_ratio_min=("final_ratio", "min"),
                 final_ratio_max=("final_ratio", "max"),
                 any_crossing=("any_crossing", "sum"),
                 sustained=("sustained_crossing", "sum"),
                 ends_above=("ends_above_criterion", "sum"))
            .reset_index())
    out = dict(source=meta.get("source"), units=int(len(U)),
               stress_levels=[int(x) for x in sorted(U.stress_V.unique())],
               failure_criterion_esr_ratio=crit,
               reference_frequency_Hz=meta.get("reference_frequency_Hz"),
               test_hours=float(U.hours.max()),
               units_with_any_crossing=int(U.any_crossing.sum()),
               units_with_sustained_crossing=int(U.sustained_crossing.sum()),
               units_ending_above_criterion=int(U.ends_above_criterion.sum()),
               max_final_ratio=float(U.final_ratio.max()),
               median_sweep_of_first_crossing=float(
                   U.loc[U.any_crossing, "sweep_of_max"].median()) if U.any_crossing.any() else None,
               within_stress_final_ratio_sd=float(U.groupby("stress_V").final_ratio.std().mean()),
               shipped_flag_failures=int((~u.censored.astype(bool)).sum()),
               by_stress=by.to_dict("records"))
    return out, by


def main():
    cap, by = capacitor_summary()
    audit = []
    for s in SOURCES:
        a = dict(s)
        a["meets_all_requirements"] = not s["lacks"]
        audit.append(a)
    res = dict(requirements=REQUIRED, sources=audit, capacitor=cap)
    json.dump(res, open(os.path.join(DATA, "real_data_audit.json"), "w"),
              indent=1, default=float)

    print("Requirements for a source to test the provisioning decision:")
    for r in REQUIRED:
        print("  -", r)
    print("\nSources examined:")
    for s in audit:
        print(f"\n  {s['name']}")
        print(f"    verdict : {s['verdict']}")
        print(f"    lacks   : {', '.join(s['lacks']) if s['lacks'] else 'nothing'}")
    print("\nThe one retained source, in full:")
    for k, v in cap.items():
        if k != "by_stress":
            print(f"  {k}: {v}")
    print(by.to_string(index=False))
    print("\nNo source examined meets all four requirements, so no empirical "
          "evaluation of the decision is reported in this article.")


if __name__ == "__main__":
    main()
