"""
Checks on the street-lighting inputs that the article reports.

- Records lost to malformed lines. Text lines minus parsed rows over-counts
  them, because closure comments can hold quoted line breaks; here a record is
  lost when the CSV parser finds the wrong number of fields.
- Date range of each fault file, which shows the year missing from the portal.
- Share of inventory points whose area comes from the nearest reported street.
- Light-out reports per energised lantern-hour on single-family LED streets in
  the fitting window, overall and by family.

    python ni_audit.py
"""
import csv
import glob
import io
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "v2")
RAW = os.path.join(OUT, "raw", "ni")
KEEP = ("Light Out", "All lights out in street")


def main():
    files = {}
    for f in sorted(glob.glob(os.path.join(RAW, "LightingFaults*.csv"))):
        text = open(f, encoding="latin-1").read()
        rows = list(csv.reader(io.StringIO(text)))
        bad = [r for r in rows[1:] if len(r) != len(rows[0])]
        d = pd.read_csv(f, encoding="latin-1", low_memory=False, on_bad_lines="skip")
        t = pd.to_datetime(d.DATE_RECORDED, errors="coerce", dayfirst=True)
        files[os.path.basename(f)] = dict(
            text_lines=text.count("\n") - 1, records=len(rows) - 1, parsed=len(d),
            malformed=len(bad), malformed_light_out=sum(any(k in x for x in r for k in KEEP) for r in bad),
            first=str(t.min()), last=str(t.max()))
    prep = json.load(open(os.path.join(OUT, "ni_prep_2025.json")))
    c = pd.read_csv(os.path.join(OUT, "ni_cells_2025.csv"))
    w = c[(c.tech == "LED") & (c.month >= "2023-04") & (c.month <= "2025-03")]
    fam = w.groupby("family").agg(y=("lightout_pure", "sum"), H=("hours_pure", "sum"))
    fam = fam[fam.index != "LED other"]
    out = dict(files=files,
               malformed=sum(v["malformed"] for v in files.values()),
               malformed_light_out=sum(v["malformed_light_out"] for v in files.values()),
               area_nearest_share=prep["area_nearest_share"],
               reports=float(w.lightout_pure.sum()), lantern_hours=float(w.hours_pure.sum()),
               hours_per_report=float(w.hours_pure.sum() / w.lightout_pure.sum()),
               hours_per_report_family={k: float(v) for k, v in (fam.H / fam.y).items()},
               reports_per_point_year=float(w.lightout_pure.sum() / w.groupby("month").N_pure.sum().mean() / 2))
    json.dump(out, open(os.path.join(OUT, "ni_audit.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "files"}, indent=1))
    for k, v in files.items():
        print(k, v)


if __name__ == "__main__":
    main()
