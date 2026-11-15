"""Split the frame index into small sky tiles for the browser (no server, no IRSA at view time).

Tiling (must match site/js/live/index.js):
  dec band  b = clamp(floor((dec + 90) / 4), 0, 44)
  band centre dec_c = -90 + 4 b + 2 ;  nra(b) = max(1, floor(360 cos(dec_c) / 4))
  RA bin    r = floor(ra / 360 * nra(b)) mod nra(b)
A frame is assigned to the tile of its centre. Frames are up to ~3.5 deg across, so a viewer
loads the 3x3 neighbourhood of tiles around the target and tests each footprint polygon.

Per tile JSON (columnar): {"release", "proc":[...], "o":[obs_id], "p":[proc index], "d":[detector],
"t":[mjd], "c":[[ra0,dec0,ra1,dec1,ra2,dec2,ra3,dec3]]}. S3 key =
  {release}/level2/{week}/{proc}/{d}/level2_{obs_id}D{d}_spx_{proc}.fits, week = obs_id[:10].
Usage: .venv/bin/python scripts/build_tiles.py --release dr1 [--parts]
"""
import argparse, json, math, os
import numpy as np
import pandas as pd


def band(dec):
    return np.clip(np.floor((np.asarray(dec) + 90) / 4).astype(int), 0, 44)


def nra(b):
    dc = -90 + 4 * np.asarray(b) + 2
    return np.maximum(1, np.floor(360 * np.cos(np.radians(dc)) / 4).astype(int))


def tile_of(ra, dec):
    b = band(dec)
    n = nra(b)
    r = np.floor(np.asarray(ra) % 360 / 360 * n).astype(int) % n
    return b, r


def main(release, from_parts):
    if from_parts:
        d = f"data/index/parts_{release}"
        df = pd.concat([pd.read_parquet(f"{d}/{f}") for f in sorted(os.listdir(d))], ignore_index=True)
    else:
        df = pd.read_parquet(f"data/index/frames_{release}.parquet")
    if "error" in df:
        df = df[df["error"].isna()]
    df = df[df["dqa"] == "Pass"].copy()
    df["proc"] = df["key"].str.split("/").str[3]
    procs = sorted(df["proc"].unique())
    pidx = {p: i for i, p in enumerate(procs)}
    b, r = tile_of(df["ra"].values, df["dec"].values)
    df["tile"] = [f"{x}_{y}" for x, y in zip(b, r)]
    out = f"site/data/index/{release}"
    os.makedirs(out, exist_ok=True)
    meta = {"release": release, "n_frames": int(len(df)), "mjd_min": float(df.mjd.min()),
            "mjd_max": float(df.mjd.max()), "tiles": {}, "procs": procs,
            "bucket": "https://nasa-irsa-spherex.s3.amazonaws.com",
            "built_from": "S3 ListObjectsV2 + header range reads (scripts/build_index.py)"}
    for t, g in df.groupby("tile"):
        g = g.sort_values("mjd")
        c = np.stack([np.stack(g.c_ra.values), np.stack(g.c_dec.values)], -1).reshape(len(g), 8)
        rec = {"release": release, "proc": procs, "o": g.obs_id.tolist(), "p": [pidx[p] for p in g.proc],
               "d": g.detector.astype(int).tolist(), "t": [round(x, 6) for x in g.mjd],
               "c": np.round(c, 4).tolist()}
        with open(f"{out}/{t}.json", "w") as f:
            json.dump(rec, f, separators=(",", ":"))
        meta["tiles"][t] = int(len(g))
    with open(f"{out}/meta.json", "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    sizes = [os.path.getsize(f"{out}/{t}.json") for t in meta["tiles"]]
    print(f"{len(df)} frames, {len(sizes)} tiles, max {max(sizes)/1e3:.0f} kB, total {sum(sizes)/1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="dr1")
    ap.add_argument("--parts", action="store_true")
    a = ap.parse_args()
    main(a.release, a.parts)
