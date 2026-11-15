"""Example motions for the live "Follow a mover" view -> site/data/track_example.json, from data/forced/<id>.parquet only.

For each asteroid: take the SPAN_H-hour stretch with the most forced-photometry rows, fit a straight line to the
catalogue-orbit positions (columns mjd, RA, DEC) against time, and write the position at the middle time (t0),
the speed (arcsec/h), the direction (degrees east of north) and the fit's rms. The first id is the page's example;
the others are listed with it (14052 is one of the asteroids of finding 2 and is too faint for the quick live view).
Usage: .venv/bin/python scripts/build_track_example.py
"""
import csv, glob, json
import numpy as np
import pandas as pd

IDS = ["4260", "14052"]
SPAN_H = 20.0
FOVS = [1, 2, 4, 8]       # arcmin, as offered by the live page
MARGIN = 30.0             # arcsec kept between the object's farthest position and the edge of the view
OUT = "site/data/track_example.json"


def names():
    out = {}
    for p in glob.glob("data/sbdb_*.csv"):
        with open(p) as f:
            out.update({r["pdes"]: " ".join(r["full_name"].split()) for r in csv.DictReader(f)})
    return out


def one(pdes, name):
    d = pd.read_parquet(f"data/forced/{pdes}.parquet").dropna(subset=["mjd", "RA", "DEC"]).sort_values("mjd")
    t = d.mjd.to_numpy()
    starts = [(np.sum((t >= a) & (t <= a + SPAN_H / 24)), a) for a in np.unique(t)]
    n, a = max(starts)
    w = d[(d.mjd >= a) & (d.mjd <= a + SPAN_H / 24)]
    t0 = (w.mjd.min() + w.mjd.max()) / 2
    hrs = (w.mjd.to_numpy() - t0) * 24
    dec_m = w.DEC.mean()
    x = (w.RA.to_numpy() - w.RA.mean()) * np.cos(np.radians(dec_m)) * 3600   # arcsec east
    y = (w.DEC.to_numpy() - w.DEC.mean()) * 3600                             # arcsec north
    px, py = np.polyfit(hrs, x, 1), np.polyfit(hrs, y, 1)
    rms = float(np.sqrt(np.mean((x - np.polyval(px, hrs)) ** 2 + (y - np.polyval(py, hrs)) ** 2)))
    ra0 = w.RA.mean() + px[1] / 3600 / np.cos(np.radians(dec_m))
    dec0 = w.DEC.mean() + py[1] / 3600
    rate = float(np.hypot(px[0], py[0]))
    pa = float(np.degrees(np.arctan2(px[0], py[0])) % 360)
    reach = rate * SPAN_H / 2 + MARGIN
    fov = next((f for f in FOVS if f * 60 / 2 >= reach), FOVS[-1])
    q = f"ra={ra0:.5f}&dec={dec0:.5f}&fov={fov}&mode=track&rate={rate:.2f}&pa={pa:.1f}&t0={t0:.4f}&span={SPAN_H:.0f}"
    return dict(id=pdes, name=name, V=round(float(w.V.median()), 2), ra=round(float(ra0), 5), dec=round(float(dec0), 5),
                rate=round(rate, 2), pa=round(pa, 1), t0=round(float(t0), 4), span=SPAN_H, fov=fov,
                n_rows=int(n), hours=round(float(hrs.max() - hrs.min()), 2), fit_rms_arcsec=round(rms, 2),
                href=f"timemachine.html?{q}#live", source=f"data/forced/{pdes}.parquet (mjd, RA, DEC, V)")


def main():
    nm = names()
    rows = [one(p, nm.get(p, p)) for p in IDS]
    out = dict(rows[0], others=rows[1:], generated_with="scripts/build_track_example.py", span_rule=f"busiest {SPAN_H:.0f} h of forced-photometry rows; straight-line fit")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for r in rows:
        print(r["name"], r["href"], "rms", r["fit_rms_arcsec"], "n", r["n_rows"])


if __name__ == "__main__":
    main()
