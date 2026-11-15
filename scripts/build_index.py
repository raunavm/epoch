"""Build a complete frame index of a SPHEREx release directly from the public S3 bucket.

No IRSA service is used: keys are listed with S3 ListObjectsV2 and each file's
primary + IMAGE headers are read with one HTTP range request (first 28800 bytes).

Output: data/index/frames_<release>.parquet with one row per Level-2 file:
key, obs_id, detector, mjd (MJD-AVG, UTC), corners (RA/Dec of the 4 image corners
with full SIP), centre RA/Dec, PSF_FWHM, L2DQAFLG.

Usage: .venv/bin/python scripts/build_index.py --release dr1 [--limit-prefixes N]
Idempotent: per-prefix results are checkpointed in data/index/parts/.
"""
import argparse, asyncio, json, os, re, sys, time
import xml.etree.ElementTree as ET
import aiohttp
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
import warnings

warnings.simplefilter("ignore", FITSFixedWarning)
BUCKET = "https://nasa-irsa-spherex.s3.amazonaws.com"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
HDR_BYTES = 28800  # primary (1 block) + IMAGE header (292 cards -> 9 blocks), measured on a DR1 file


async def list_keys(session, prefix, delimiter=None):
    keys, prefixes, token = [], [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        for attempt in range(6):
            try:
                async with session.get(BUCKET + "/", params=params) as r:
                    r.raise_for_status()
                    root = ET.fromstring(await r.read())
                break
            except Exception:
                await asyncio.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"listing failed: {prefix}")
        keys += [c.find("s3:Key", NS).text for c in root.findall("s3:Contents", NS)]
        prefixes += [p.find("s3:Prefix", NS).text for p in root.findall("s3:CommonPrefixes", NS)]
        if root.find("s3:IsTruncated", NS).text != "true":
            return keys, prefixes
        token = root.find("s3:NextContinuationToken", NS).text


def parse_header(key, raw):
    # Split the two headers: the primary header ends at the first block containing END.
    blocks = [raw[i:i + 2880] for i in range(0, len(raw), 2880)]
    hdrs, cur = [], b""
    for b in blocks:
        cur += b
        if re.search(rb"END {77}", b) or b.rstrip().endswith(b"END"):
            hdrs.append(cur)
            cur = b""
        if len(hdrs) == 2:
            break
    h = fits.Header.fromstring(hdrs[1].decode("ascii", "replace"))
    w = WCS(h)
    nx, ny = h["NAXIS1"], h["NAXIS2"]
    corners = w.all_pix2world([[0, 0], [nx - 1, 0], [nx - 1, ny - 1], [0, ny - 1]], 0)
    centre = w.all_pix2world([[(nx - 1) / 2, (ny - 1) / 2]], 0)[0]
    return dict(
        key=key, obs_id=h.get("OBSID"), detector=int(h.get("DETECTOR")), mjd=float(h["MJD-AVG"]),
        ra=float(centre[0]), dec=float(centre[1]),
        c_ra=[float(x) for x in corners[:, 0]], c_dec=[float(x) for x in corners[:, 1]],
        psf_fwhm=float(h.get("PSF_FWHM", np.nan)), dqa=str(h.get("L2DQAFLG", "")),
    )


async def read_header(session, sem, key):
    async with sem:
        for attempt in range(6):
            try:
                async with session.get(f"{BUCKET}/{key}", headers={"Range": f"bytes=0-{HDR_BYTES - 1}"}) as r:
                    r.raise_for_status()
                    raw = await r.read()
                return parse_header(key, raw)
            except Exception as e:
                err = e
                await asyncio.sleep(1.5 ** attempt)
        return dict(key=key, error=repr(err))


async def main(release, limit, conc, shard=0, nshard=1, merge=True):
    out = f"data/index/parts_{release}"
    os.makedirs(out, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=conc)) as s:
        _, weeks = await list_keys(s, f"{release}/level2/", delimiter="/")
        weeks = weeks[:limit] if limit else weeks
        weeks = weeks[shard::nshard]
        sem = asyncio.Semaphore(conc)
        t0 = time.time()
        for i, wk in enumerate(weeks):
            part = f"{out}/{wk.strip('/').split('/')[-1]}.parquet"
            if os.path.exists(part):
                continue
            keys, _ = await list_keys(s, wk)
            keys = [k for k in keys if k.endswith(".fits")]
            rows = await asyncio.gather(*[read_header(s, sem, k) for k in keys])
            pd.DataFrame(rows).to_parquet(part)
            n_err = sum("error" in r and isinstance(r.get("error"), str) for r in rows)
            print(f"[{i+1}/{len(weeks)}] {wk} n={len(keys)} err={n_err} elapsed={time.time()-t0:.0f}s", flush=True)
    if not merge:
        return
    parts = [pd.read_parquet(f"{out}/{f}") for f in sorted(os.listdir(out)) if f.endswith(".parquet")]
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(f"data/index/frames_{release}.parquet")
    print("total", len(df), "errors", df["error"].notna().sum() if "error" in df else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="dr1")
    ap.add_argument("--limit-prefixes", type=int, default=0)
    ap.add_argument("--conc", type=int, default=48)
    ap.add_argument("--shard", type=str, default="0/1", help="i/n: process every n-th week prefix starting at i")
    ap.add_argument("--merge-only", action="store_true")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    asyncio.run(main(a.release, a.limit_prefixes, a.conc if not a.merge_only else 1, i, n if not a.merge_only else 10**6, merge=(n == 1 or a.merge_only)))
