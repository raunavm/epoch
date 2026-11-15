"""Raw native-pixel cutouts of the 10 E3 ecliptic fields in the shared data/efields/ format (data/efields/README.md),
written with scripts/e2_cache.py:build (same frame cap, same stamp format, READY last). Resumable: rerun to continue.
  OMP_NUM_THREADS=1 .venv/bin/python scripts/e3_cache.py
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2_cache  # noqa: E402

df, src = e2_cache.load_index()
assert src == e2_cache.INDEX, src
e3 = json.load(open("data/findings/e3_fields.json"))
with ThreadPoolExecutor(e2_cache.WORKERS) as ex:
    for f in e3["fields"]:
        e2_cache.build(f["field"], f["ra"], f["dec"], df, ex,
                       dict(role="E3 ecliptic test field", ecl_lon=f["ecl_lon"], ecl_lat=f["ecl_lat"],
                            draw="docs/PROTOCOLS.md E3, default_rng(20261116), data/findings/e3_fields.json"))
