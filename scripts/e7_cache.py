"""Raw native-pixel cutouts of the E7 target fields (docs/PROTOCOLS.md E7) in the shared data/efields/ format
(data/efields/README.md), written with scripts/e2_cache.py:build (same frame cap, stamp format, READY last).
Centres = site/data/timemachine/<field>.json (ra, dec). Resumable: rerun to continue.
  OMP_NUM_THREADS=1 .venv/bin/python scripts/e7_cache.py
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2_cache  # noqa: E402

df, src = e2_cache.load_index()
assert src == e2_cache.INDEX, src
with ThreadPoolExecutor(3) as ex:
    for f in ("rwaur", "rdra", "epsindib"):
        tm = json.load(open(f"site/data/timemachine/{f}.json"))
        e2_cache.build(f"e7_{f}", tm["ra"], tm["dec"], df, ex,
                       dict(role="E7 target field", target=tm["name"], centre_source=f"site/data/timemachine/{f}.json"))
