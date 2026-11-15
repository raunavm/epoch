"""E3 ecliptic fields (docs/PROTOCOLS.md E3): default_rng(20261116); per field, ecliptic longitude U(0, 360) then
ecliptic latitude U(-2, 2) deg (mean ecliptic and equinox of J2000, astropy BarycentricMeanEcliptic) -> ICRS.
Writes data/findings/e3_fields.json and prints one build_field.py command per field (48 px, frame cap 250).
"""
import json
import numpy as np
from astropy.coordinates import SkyCoord, BarycentricMeanEcliptic
import astropy.units as u

rng = np.random.default_rng(20261116)
out = []
for i in range(10):
    lon, lat = rng.uniform(0, 360), rng.uniform(-2, 2)
    c = SkyCoord(lon * u.deg, lat * u.deg, frame=BarycentricMeanEcliptic()).icrs
    out.append(dict(field=f"e3_ecl{i:02d}", ecl_lon=round(lon, 6), ecl_lat=round(lat, 6),
                    ra=round(float(c.ra.deg), 6), dec=round(float(c.dec.deg), 6)))
json.dump(dict(seed=20261116, frame="BarycentricMeanEcliptic (J2000)", size_px=48, frame_cap=250, fields=out),
          open("data/findings/e3_fields.json", "w"), indent=1)
for f in out:
    print(f"OMP_NUM_THREADS=1 .venv/bin/python scripts/build_field.py --name {f['field']} --ra {f['ra']} --dec {f['dec']} "
          f"--size 48 --max-frames 250 --title 'E3 ecliptic field {f['field'][-2:]} (lon {f['ecl_lon']:.2f}, lat {f['ecl_lat']:+.2f})'"
          f" > data/findings/run_{f['field']}.log 2>&1")
