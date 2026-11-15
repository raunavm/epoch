"""E13b (docs/protocols/E13b.md): stacked faint asteroids against 200 empty paths.
Usage: .venv/bin/python scripts/run_e13b.py
"""
import json, os, sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from build_reveal import items_for, HALF  # noqa: E402
from orrery.stack import combine  # noqa: E402

RADII = (5, 6, 7, 8, 9)
OFFS = [(r, a) for j, r in enumerate(RADII) for a in np.arange(40) * 2 * np.pi / 40 + (np.pi / 40 if j % 2 else 0)]


def one(pdes):
    df = pd.read_parquet(f"data/forced/{pdes}.parquet")
    it, meta = items_for(pdes, df)
    _, _, A, E = combine(it, HALF)
    real = float(A / E)
    nulls = []
    for r, a in OFFS:
        itn, _ = items_for(pdes, df, offset=(r * np.cos(a), r * np.sin(a)))
        _, _, An, En = combine(itn, HALF)
        nulls.append(float(An / En))
    nulls = np.array(nulls)
    ok = np.isfinite(nulls)
    nn = nulls[ok]
    p = (1 + int((nn >= real).sum())) / (1 + len(nn))
    z = (real - nn.mean()) / nn.std(ddof=1)
    return dict(pdes=str(pdes), name=meta["name"], n_pictures=len(it), real_nominal=round(real, 2), n_nulls=int(len(nn)),
                null_mean=round(float(nn.mean()), 3), null_std=round(float(nn.std(ddof=1)), 3), null_max=round(float(nn.max()), 2),
                empirical_p=round(p, 5), z=round(float(z), 2),
                verdict="detected against 200 empty paths" if p <= 1 / 201 + 1e-12 and z >= 5 else "not detected at the E13b standard",
                null_snrs=[round(float(x), 3) for x in nulls])


if __name__ == "__main__":
    ids = ["14052", "11874", "98397", "80253"]
    with ProcessPoolExecutor(2) as ex:
        res = list(ex.map(one, ids))
    for r in res:
        print({k: v for k, v in r.items() if k != "null_snrs"}, flush=True)
    json.dump(dict(protocol="docs/protocols/E13b.md", results=res), open("data/findings/e13b.json", "w"), indent=1)
