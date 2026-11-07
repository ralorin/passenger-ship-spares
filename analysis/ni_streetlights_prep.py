"""
Northern Ireland street lighting (DfI Roads open data, Open Government Licence):
monthly fault counts per operational area and LED lantern family, with the
installed population as exposure.

Faults carry no asset identifier, only "STREET, TOWN" and the operational area,
so each fault is shared among the lanterns on its street in proportion to
their numbers. On streets lit by a single family the attribution is exact;
those counts are kept apart (the *_pure columns) and are the ones the analysis
uses. The inventory carries no operational area; a street takes the area its
faults were reported under, and a street with no faults takes the area of the
nearest street that has one.

Inputs (downloaded 2026-11-01 into data/v2/raw/ni/, not tracked):
  LightingFaults{2021..2024}.csv, LightingFaultsCurrentYear.csv
      https://data.gov.uk/dataset/692f16a2-a1fa-43ab-853c-fe399336bfc2/lighting-faults
  Street Lighting Assets.csv (export of 2026-11-01) and the 2025-04-24 export
      https://dfi.highway-iams.uk/OpenData/Street%20Lighting%20Assets.csv

    python ni_streetlights_prep.py [--inventory 2025|2026]
"""
import argparse
import glob
import json
import math
import os
import re

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "v2", "raw", "ni")
OUT = os.path.join(HERE, "..", "data", "v2")

KEEP = {"Light Out": "light_out", "All lights out in street": "all_out"}
FAMILIES = [("Urbis Axia", r"URBIS AXIA"), ("Urbis Ampera", r"URBIS AMPERA"),
            ("Philips Lumistreet", r"LUMI?N?STREET"), ("Philips Digistreet", r"DIGISTREET"),
            ("LCL Quantum", r"QUANTUM"), ("Thorn R2L2", r"R2L2"), ("Ignis Mini", r"IGNIS")]
LAT = 54.6
REPEAT_DAYS = 7      # a report this soon after the last one on the street is the same repair
NOT_LANTERN = r"cable|NIE|working on attendance|not out|no fault|all ok|operational"
SKIPPED = {}         # malformed lines (unquoted commas in free-text comments)


def family(name):
    s = str(name).upper()
    if "LED" not in s:
        return "SOX" if "SOX" in s else "SON" if "SON" in s else "other"
    for fam, pat in FAMILIES:
        if re.search(pat, s):
            return fam
    return "LED other"


def norm(s):
    return re.sub(r"\s+", " ", str(s).upper().strip())


def night_hours(month):
    """Dusk-to-dawn hours in a calendar month at latitude LAT (sunrise/sunset formula)."""
    start = pd.Period(month, "M").start_time
    days = pd.date_range(start, periods=start.days_in_month, freq="D")
    n = days.dayofyear.to_numpy()
    dec = np.radians(23.44) * np.sin(2 * np.pi * (284 + n) / 365)
    phi = math.radians(LAT)
    cosh = (np.sin(np.radians(-0.833)) - math.sin(phi) * np.sin(dec)) / (math.cos(phi) * np.cos(dec))
    day = 2 * np.degrees(np.arccos(np.clip(cosh, -1, 1))) / 15
    return float(np.sum(24 - day))


def load_faults():
    frames = []
    for f in sorted(glob.glob(os.path.join(RAW, "LightingFaults*.csv"))):
        d = pd.read_csv(f, encoding="latin-1", low_memory=False, on_bad_lines="skip")
        with open(f, encoding="latin-1") as fh:
            SKIPPED[os.path.basename(f)] = sum(1 for _ in fh) - 1 - len(d)
        d = d.rename(columns={"OPERTAIONAL_AREA": "OPERATIONAL_AREA"})
        d["file"] = os.path.basename(f)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True).drop_duplicates("DATA_UID")
    d["date"] = pd.to_datetime(d.DATE_RECORDED, dayfirst=True, errors="coerce")
    d["month"] = d.date.dt.to_period("M").astype(str)
    parts = d.SECTION_NAME.astype(str).str.rsplit(",", n=1, expand=True)
    d["street"] = parts[0].map(norm)
    d["town"] = parts[1].map(norm) if parts.shape[1] > 1 else ""
    return d


def load_inventory(which):
    path = (os.path.join(RAW, "wayback", "Street Lighting Assets 20250424.csv") if which == "2025"
            else os.path.join(RAW, "Street Lighting Assets.csv"))
    a = pd.read_csv(path, encoding="latin-1", low_memory=False)
    a["street"] = a.STREET_DESCRIPTOR.map(norm)
    a["town"] = a.TOWN_NAME.map(norm)
    a["family"] = a.LANTERN_BOQ_NAME.map(family)
    return a


def assign_area(a, faults):
    """Area of each inventory point: modal fault area of its street, else nearest such street."""
    known = (faults[faults.OPERATIONAL_AREA != "Unable to determine"]
             .groupby(["street", "town"]).OPERATIONAL_AREA.agg(lambda x: x.mode().iloc[0]))
    a = a.join(known.rename("area"), on=["street", "town"])
    ok = a.area.notna() & a.EASTING.notna() & a.NORTHING.notna()
    miss = a.area.isna() & a.EASTING.notna() & a.NORTHING.notna()
    tree = cKDTree(a.loc[ok, ["EASTING", "NORTHING"]].to_numpy())
    _, idx = tree.query(a.loc[miss, ["EASTING", "NORTHING"]].to_numpy())
    a.loc[miss, "area"] = a.loc[ok, "area"].to_numpy()[idx]
    return a, float(ok.mean()), float(miss.mean())


def build(which="2025", repeat_days=REPEAT_DAYS, merge_max_points=None, F=None, A=None):
    """Cell table and preparation report. merge_max_points, if given, merges
    repeat reports only on streets with at most that many lighting points."""
    F = load_faults() if F is None else F
    A, share_direct, share_nn = assign_area(load_inventory(which) if A is None else A, F)
    A = A[A.area.notna()]

    types = F.FAULT_TYPE.value_counts()
    F = F[F.FAULT_TYPE.isin(KEEP) & (F.ASSET_TYPE == "Lighting Point")
          & (F.OPERATIONAL_AREA != "Unable to determine") & F.date.notna()].copy()
    F["kind"] = F.FAULT_TYPE.map(KEEP)
    n_kept = len(F)
    not_lantern = F.CLEARED_COMMENTS.fillna("").str.contains(NOT_LANTERN, case=False, regex=True)
    F = F[~not_lantern].sort_values("date")
    small = None
    if merge_max_points is not None:
        size = A.groupby(["street", "town"]).size()
        small = set(size[size <= merge_max_points].index)
    # keep a report only if it starts a new episode on its street
    keep = np.zeros(len(F), bool)
    start_of = {}
    for i, (k, t) in enumerate(zip(zip(F.street, F.town, F.kind), F.date)):
        if small is not None and k[:2] not in small:
            keep[i] = True
            continue
        if k not in start_of or (t - start_of[k]).days > repeat_days:
            start_of[k] = t
            keep[i] = True
    dropped = dict(not_lantern=int(not_lantern.sum()), repeats=int((~keep).sum()))
    F = F[keep]

    # share each fault among the families on its street
    street_mix = A.groupby(["street", "town", "family"]).size().rename("n").reset_index()
    street_mix["share"] = street_mix.n / street_mix.groupby(["street", "town"]).n.transform("sum")
    street_mix["pure"] = street_mix.groupby(["street", "town"]).family.transform("nunique") == 1
    M = F.merge(street_mix, on=["street", "town"], how="left")
    matched = M.groupby("DATA_UID").family.apply(lambda x: x.notna().any())
    M = M[M.family.notna()]

    # streets carrying a single family give exact attribution; populations and
    # counts are also kept on those streets alone (the *_pure columns)
    A = A.merge(street_mix[["street", "town", "family", "pure"]], on=["street", "town", "family"])
    months = sorted(F.month.unique())
    N = (A.groupby(["area", "family"]).agg(N=("pure", "size"), N_pure=("pure", "sum")).reset_index())
    grid = N.merge(pd.DataFrame({"month": months}), how="cross")
    M["lo"] = M.kind == "light_out"
    M["ao"] = M.kind == "all_out"
    agg = (M.assign(w=M.share * M.lo, w_all=M.share * M.ao,
                    p=M.pure & M.lo, p_all=M.pure & M.ao)
           .groupby(["OPERATIONAL_AREA", "family", "month"])[["w", "w_all", "p", "p_all"]].sum()
           .rename_axis(["area", "family", "month"]).reset_index())
    cells = grid.merge(agg, on=["area", "family", "month"], how="left")
    cells[["w", "w_all", "p", "p_all"]] = cells[["w", "w_all", "p", "p_all"]].fillna(0.0)
    cells = cells.rename(columns={"w": "lightout_frac", "w_all": "allout_frac",
                                  "p": "lightout_pure", "p_all": "allout_pure"})
    cells[["lightout_pure", "allout_pure"]] = cells[["lightout_pure", "allout_pure"]].astype(int)
    nh = {m: night_hours(m) for m in months}
    cells["night_hours"] = cells.month.map(nh)
    cells["hours"] = cells.N * cells.night_hours
    cells["hours_pure"] = cells.N_pure * cells.night_hours
    cells["tech"] = np.where(cells.family.isin(["SON", "SOX", "other"]), cells.family, "LED")

    led = A[~A.family.isin(["SON", "SOX", "other"])]
    report = dict(
        inventory=which, inventory_points=len(A),
        led_points=len(led), families=led.family.value_counts().to_dict(),
        led_points_pure=int(led.pure.sum()), families_pure=led[led.pure].family.value_counts().to_dict(),
        areas=int(A.area.nunique()), area_direct_share=share_direct, area_nearest_share=share_nn,
        fault_types_all=types.to_dict(), faults_kept=int(len(F)),
        faults_matched_share=float(matched.mean()), lines_skipped=SKIPPED,
        faults_before_cleaning=n_kept, dropped=dropped,
        faults_on_pure_streets_share=float(M.groupby("DATA_UID").pure.first().mean()),
        months=[months[0], months[-1]], n_months=len(months))
    return cells, report, dict(faults=F, matched=M, inventory=A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default="2025", choices=["2025", "2026"])
    args = ap.parse_args()
    cells, report, _ = build(args.inventory)
    cells.to_csv(os.path.join(OUT, f"ni_cells_{args.inventory}.csv"), index=False)
    json.dump(report, open(os.path.join(OUT, f"ni_prep_{args.inventory}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "fault_types_all"}, indent=1))


if __name__ == "__main__":
    main()
