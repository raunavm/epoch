"""E10 — does the per-star change test recover known variable stars? (docs/PROTOCOLS.md PHASE 2 / E10;
frozen choices: docs/protocols/E10.md). Resumable; every stage writes to disk.

  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
  .venv/bin/python scripts/run_e10.py sample            # draw V + C (fixed seed) -> data/phase2/e10/sample.json
  .venv/bin/python scripts/run_e10.py throughput        # time the spare group (not in the sample)
  .venv/bin/python scripts/run_e10.py phot [n_groups]   # per-picture photometry -> data/phase2/e10/phot/<group>.parquet
  .venv/bin/python scripts/run_e10.py calibrate         # calibration half ONLY: floor + thresholds (frozen)
  .venv/bin/python scripts/run_e10.py analyze           # all stars -> data/findings/e10.json, e10_stars.parquet, figure
"""
import io, json, os, sys, threading, time, warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import requests  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.io.fits.hdu.compressed._codecs import Rice1  # noqa: E402
from astropy.wcs import WCS, FITSFixedWarning  # noqa: E402
from orrery import e6, e7  # noqa: E402
from orrery.data import BUCKET  # noqa: E402
from orrery.e1 import GAIA_EPOCH, wave_map  # noqa: E402
from orrery.fit import flux_mJy  # noqa: E402

warnings.simplefilter("ignore", FITSFixedWarning)
warnings.filterwarnings("ignore")

D = "data/phase2/e10"
INDEX = "data/index/frames_dr1.parquet"
GATE = "data/findings/e7_gate0.json"
THR = "data/findings/e10_thresholds.json"
OUT = "data/findings/e10.json"
STARS = "data/findings/e10_stars.parquet"
FIG = "site/data/figs/e10_recall.png"
GAIA_TAP = "https://gaia.aip.de/tap"
VIZ = "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv"       # CDS Strasbourg was unavailable on 2026-11-15
UA = {"User-Agent": "python-client"}

SEED = 20261120
QUOTA = {"LPV": 26, "RR": 14, "ECL": 14, "CEP": 8}            # variables per type (sum 62)
N_COMP = 4                                                     # comparison stars per variable
G_LO, G_HI, B_MIN, AMP_MIN = 9.0, 14.0, 15.0, 0.3
R_GROUP, R_COMP_MIN, R_COMP_MAX = 0.56, 0.05, 0.5              # deg
MAX_GAIA_ROWS = 20000                                          # G < 19 sources in the group cone (crowding cap)
CLOUDS = ((80.89, -69.76, 10.0), (13.19, -72.83, 6.0))         # LMC, SMC exclusion (ra, dec, radius deg)
CAP, HALF, EDGE = 96, 8, 10
MIN_WIN_FRAMES, MIN_SEP_DAYS = 20, 60.0
R_NB = (HALF + 4) * 6.2 / 3600 * 1.42                          # neighbour cone per star (deg)
MIN_PICS, MIN_WIN_PICS, MIN_DET_PICS, MIN_DW_PICS = 20, 5, 5, 4
PROCS, THREADS = 6, 8
AMP_BINS = [(0.3, 0.5), (0.5, 1.0), (1.0, 99.0)]


# ---------------------------------------------------------------- small helpers
def sep_deg(ra1, dec1, ra2, dec2):
    a1, d1, a2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    return np.degrees(2 * np.arcsin(np.sqrt(np.sin((d1 - d2) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((a1 - a2) / 2) ** 2)))


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def tap(q, maxrec=2000000):
    import pyvo
    for i in range(4):
        try:
            return pyvo.dal.TAPService(GAIA_TAP).run_sync(q, maxrec=maxrec).to_table().to_pandas()
        except Exception as e:
            print("  tap retry", repr(e)[:120], flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError("Gaia TAP failed")


def vizier_cone(source, ra, dec, r_deg, out):
    """Rows of a VizieR table in a cone (CfA mirror, TSV). Returns DataFrame of strings; None if unreachable."""
    p = {"-source": source, "-c": f"{ra:.6f} {dec:+.6f}", "-c.rd": f"{r_deg}", "-out.max": "unlimited", "-out": out}
    for i in range(3):
        try:
            r = requests.get(VIZ, params=p, headers=UA, timeout=(10, 60))
            r.raise_for_status()
            lines = [ln for ln in r.text.splitlines() if ln and not ln.startswith("#")]
            if len(lines) < 3:
                return pd.DataFrame()
            cols = lines[0].split("\t")
            rows = [ln.split("\t") for ln in lines[3:]]
            return pd.DataFrame([[c.strip() for c in rw] for rw in rows if len(rw) == len(cols)], columns=cols)
        except Exception:
            time.sleep(3 * (i + 1))
    return None


# ---------------------------------------------------------------- SPHEREx coverage
_IDX = {}


def index():
    if not _IDX:
        df = pd.read_parquet(INDEX)
        df = df[df["dqa"] == "Pass"].reset_index(drop=True)
        a, d = np.radians(df.ra.values), np.radians(df.dec.values)
        _IDX.update(df=df, u=np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)], 1),
                    cra=np.stack(df.c_ra.values), cdec=np.stack(df.c_dec.values))
    return _IDX


def covering(ra, dec):
    """Index rows whose footprint polygon contains (ra, dec) (gnomonic projection about the point)."""
    I = index()
    a, d = np.radians(ra), np.radians(dec)
    v = np.array([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])
    k = np.flatnonzero(I["u"] @ v > np.cos(np.radians(2.8)))
    if len(k) == 0:
        return I["df"].iloc[k]
    ca, cd = np.radians(I["cra"][k]), np.radians(I["cdec"][k])
    cosc = np.sin(d) * np.sin(cd) + np.cos(d) * np.cos(cd) * np.cos(ca - a)
    xi = np.cos(cd) * np.sin(ca - a) / cosc
    eta = (np.cos(d) * np.sin(cd) - np.sin(d) * np.cos(cd) * np.cos(ca - a)) / cosc
    o = np.argsort(np.arctan2(eta - eta.mean(1, keepdims=True), xi - xi.mean(1, keepdims=True)), 1)
    px, py = np.take_along_axis(xi, o, 1), np.take_along_axis(eta, o, 1)
    cr = (np.roll(px, -1, 1) - px) * (0 - py) - (np.roll(py, -1, 1) - py) * (0 - px)
    inside = (cr > 0).all(1) | (cr < 0).all(1)
    return I["df"].iloc[k[inside]]


def select_frames(ra, dec):
    """Frozen frame rule: covering frames, windows at gaps > 30 d, even draw per (window, detector) cell, cap CAP.
    Returns (frames DataFrame with window column, window summary) or (None, reason)."""
    c = covering(ra, dec).sort_values("mjd", kind="stable").reset_index(drop=True)
    if len(c) < 2 * MIN_WIN_FRAMES:
        return None, f"{len(c)} covering frames"
    c["window"] = e7.windows(c.mjd.values)
    ws = c.groupby("window").mjd.agg(["count", "median"])
    ws = ws[ws["count"] >= MIN_WIN_FRAMES]
    if len(ws) < 2 or ws["median"].max() - ws["median"].min() < MIN_SEP_DAYS:
        return None, f"windows {ws['count'].tolist()}"
    c = c[c.window.isin(ws.index)]
    cells = [g.sort_values("mjd", kind="stable") for _, g in c.groupby(["window", "detector"])]
    q = max(1, CAP // len(cells))
    pick = [g.iloc[np.unique(np.linspace(0, len(g) - 1, min(len(g), q)).round().astype(int))] for g in cells]
    f = pd.concat(pick).sort_values("mjd", kind="stable").reset_index(drop=True)
    return f[["key", "obs_id", "detector", "mjd", "window"]], dict(n_covering=int(len(c)), n_windows=int(len(ws)),
                                                                  window_median_mjd=[float(x) for x in ws["median"]])


# ---------------------------------------------------------------- sample draw
def sample(ext=False):
    """ext=False: the frozen first draw. ext=True: the pre-registered extension (docs/protocols/E10.md, addendum):
    the same permuted candidate lists continued past the sources already used, the same rules and quotas, a separate
    generator (SEED + 10) for the comparison-star choice and (SEED + 11) for the calibration / test split."""
    os.makedirs(f"{D}/groups", exist_ok=True)
    path = f"{D}/sample_ext.json" if ext else f"{D}/sample.json"
    if os.path.exists(path):
        sys.exit(f"{path} exists (frozen)")
    rng = np.random.default_rng(SEED)
    first = json.load(open(f"{D}/sample.json")) if ext else None
    pools = {}
    cols = "source_id, ra, dec, l, b, pmra, pmdec, phot_g_mean_mag, bp_rp, phot_variable_flag"
    for typ, f in (("LPV", "pool_lpv_raw"), ("ECL", "pool_ecl_raw")):
        raw = pd.read_parquet(f"{D}/{f}.parquet").sort_values("source_id").reset_index(drop=True)
        sub = raw.iloc[rng.permutation(len(raw))[:3000]]
        ids = sub.source_id.astype("int64").tolist()
        with ThreadPoolExecutor(3) as tp:
            g = pd.concat(list(tp.map(lambda i: tap(f"SELECT {cols} FROM gaiadr3.gaia_source WHERE source_id IN "
                                                    f"({','.join(map(str, ids[i:i + 500]))})"), range(0, len(ids), 500))))
        m = sub.merge(g, on="source_id")
        pools[typ] = m.set_index("source_id").loc[[i for i in ids if i in set(m.source_id)]].reset_index()
    for typ, f in (("RR", "pool_rr"), ("CEP", "pool_cep")):
        raw = pd.read_parquet(f"{D}/{f}.parquet").sort_values("source_id").reset_index(drop=True)
        pools[typ] = raw.iloc[rng.permutation(len(raw))].reset_index(drop=True)
    log, cand = {}, {}
    for typ, p in pools.items():
        ok = (p.phot_g_mean_mag > G_LO) & (p.phot_g_mean_mag < G_HI) & (p.b.abs() > B_MIN) & \
             (p.trimmed_range_mag_g_fov >= AMP_MIN)
        for ra0, de0, r0 in CLOUDS:
            ok &= sep_deg(p.ra.values, p.dec.values, ra0, de0) > r0
        log[typ] = dict(pool=int(len(p)), pass_cuts=int(ok.sum()))
        cand[typ] = p[ok].reset_index(drop=True)
    groups, rejected = [], []
    want = dict(QUOTA)
    used, prev = set(), []
    if ext:
        prev = first["groups"] + first["spare"]
        used = {g["V"]["source_id"] for g in prev} | {r["source_id"] for r in first["rejected"]}
        cand = {t: c[~c.source_id.isin(used)].reset_index(drop=True) for t, c in cand.items()}
        rng = np.random.default_rng(SEED + 10)
    else:
        want["LPV"] += 1                                        # one spare LPV group for the throughput test
    for typ in ("LPV", "RR", "ECL", "CEP"):
        n, it = 0, cand[typ].itertuples()
        while n < want[typ]:
            batch = []
            for r in it:                                        # next candidates that pass the coverage rule
                fr, info = select_frames(float(r.ra), float(r.dec))
                if fr is None:
                    rejected.append(dict(type=typ, source_id=int(r.source_id), reason="coverage: " + info))
                    continue
                batch.append(r)
                if len(batch) >= 2 * (want[typ] - n) + 2:
                    break
            if not batch:
                break
            with ThreadPoolExecutor(4) as tp:                   # prefetch the Gaia cones (network only)
                list(tp.map(lambda r: fetch_cone(int(r.source_id), float(r.ra), float(r.dec)), batch))
            for r in batch:
                if n >= want[typ]:
                    break
                g, why = build_group(typ, r, rng, [x["ra"] for x in groups + prev], [x["dec"] for x in groups + prev])
                if g is None:
                    rejected.append(dict(type=typ, source_id=int(r.source_id), reason=why))
                    continue
                g["rank"] = n
                g["spare"] = bool(not ext and typ == "LPV" and n == QUOTA["LPV"])
                g["priority"] = (n + 0.5) / QUOTA[typ]
                groups.append(g)
                n += 1
                print(typ, n, g["gid"], f"G={g['V']['G']:.2f} amp={g['V']['amp']:.2f} nC={len(g['C'])} frames={g['n_frames']}", flush=True)
        log[typ]["accepted"] = n
    main = sorted([g for g in groups if not g["spare"]], key=lambda g: (g["priority"], g["type"]))
    cids = sorted(c["source_id"] for g in main for c in g["C"])
    perm = np.random.default_rng(SEED + (11 if ext else 1)).permutation(len(cids))
    half = {cids[j]: ("calibration" if i < len(cids) // 2 else "test") for i, j in enumerate(perm)}
    for g in main:
        for c in g["C"]:
            c["half"] = half[c["source_id"]]
    out = dict(experiment="E10", extension=ext, seed=SEED, split_seed=SEED + (11 if ext else 1), comp_seed=SEED + 10 if ext else SEED, index=INDEX, n_index_frames=int(len(index()["df"])),
               quota=QUOTA, n_comp_per_variable=N_COMP, pool_log=log, rejected=rejected,
               groups=main, spare=[g for g in groups if g["spare"]],
               n_V=len(main), n_C=len(cids), n_C_calibration=sum(v == "calibration" for v in half.values()),
               n_C_test=sum(v == "test" for v in half.values()), drawn_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(out, open(path, "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("n_V", "n_C", "n_C_calibration", "n_C_test", "pool_log")}, indent=1))


def fetch_cone(sid, ra, dec):
    """Gaia G < 19 cone of a group (cached). A 0.1 deg count is taken first; if it implies more than MAX_GAIA_ROWS
    in the full cone, only a marker file is written. Returns the path of the parquet or of the marker."""
    gp, mk = f"{D}/groups/{sid}_gaia.parquet", f"{D}/groups/{sid}_crowded.txt"
    if os.path.exists(gp):
        return gp
    if os.path.exists(mk):
        return mk
    cone = "1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{},{},{})) AND phot_g_mean_mag<19"
    n = int(tap("SELECT COUNT(*) AS n FROM gaiadr3.gaia_source WHERE " + cone.format(ra, dec, 0.1)).iloc[0, 0])
    if n * (R_GROUP / 0.1) ** 2 > MAX_GAIA_ROWS:
        open(mk, "w").write(f"{n} Gaia G<19 sources within 0.1 deg (scaled: {n * (R_GROUP / 0.1) ** 2:.0f} in the cone)")
        return mk
    ga = tap("SELECT source_id, ra, dec, pmra, pmdec, phot_g_mean_mag, phot_variable_flag FROM gaiadr3.gaia_source "
             "WHERE " + cone.format(ra, dec, R_GROUP), maxrec=200000)
    ga.to_parquet(gp + ".tmp", index=False)
    os.replace(gp + ".tmp", gp)
    return gp


def build_group(typ, r, rng, gra, gdec):
    ra, dec, sid = float(r.ra), float(r.dec), int(r.source_id)
    if len(gra) and (sep_deg(np.array(gra), np.array(gdec), ra, dec) < 2 * R_GROUP).any():
        return None, "overlaps an accepted group"
    frames, info = select_frames(ra, dec)
    if frames is None:
        return None, "coverage: " + info
    gp = fetch_cone(sid, ra, dec)
    if not gp.endswith(".parquet"):
        return None, "crowded: " + open(gp).read()
    ga = pd.read_parquet(gp)
    if len(ga) > MAX_GAIA_ROWS:
        return None, f"crowded: {len(ga)} Gaia G<19 sources in the cone"
    vsx = vizier_cone("B/vsx/vsx", ra, dec, R_GROUP, "OID,Name,V,Type,max,n_max,min,n_min,f_min,Period,RAJ2000,DEJ2000")
    if vsx is None:
        return None, "VSX unreachable"
    vra = pd.to_numeric(vsx.get("RAJ2000", pd.Series(dtype=float)), errors="coerce").values
    vde = pd.to_numeric(vsx.get("DEJ2000", pd.Series(dtype=float)), errors="coerce").values

    def vsx_match(a, d, rad=5.0):
        if len(vra) == 0:
            return None
        s = sep_deg(vra, vde, a, d) * 3600
        j = int(np.nanargmin(s))
        return dict(vsx.iloc[j][["OID", "Name", "Type", "max", "n_max", "min", "n_min", "f_min", "Period"]], sep=float(s[j])) \
            if s[j] < rad else None

    vm = vsx_match(ra, dec)
    if vm is None and typ != "LPV":
        return None, "no VSX entry within 5 arcsec"
    s = sep_deg(ga.ra.values, ga.dec.values, ra, dec)
    el = ga[(s > R_COMP_MIN) & (s < R_COMP_MAX) & (ga.phot_g_mean_mag > G_LO) & (ga.phot_g_mean_mag < G_HI) &
            (ga.phot_variable_flag != "VARIABLE")].sort_values("source_id")
    el = el[[vsx_match(a, d) is None for a, d in zip(el.ra, el.dec)]]
    if len(el) < 2:
        return None, f"{len(el)} eligible comparison stars"
    ch = el.iloc[np.sort(rng.choice(len(el), min(N_COMP, len(el)), replace=False))]
    frames.to_parquet(f"{D}/groups/{sid}_frames.parquet", index=False)
    V = dict(source_id=sid, ra=ra, dec=dec, G=float(r.phot_g_mean_mag), bp_rp=float(r.bp_rp), b=float(r.b),
             amp=float(r.trimmed_range_mag_g_fov), amp_range=float(r.range_mag_g_fov), type=typ, vsx=vm,
             period_d=(1 / float(r.frequency) if "frequency" in r._fields and r.frequency > 0 else
                       (float(r.pf) if "pf" in r._fields and np.isfinite(r.pf) else None)))
    C = [dict(source_id=int(c.source_id), ra=float(c.ra), dec=float(c.dec), G=float(c.phot_g_mean_mag),
              sep_deg=float(sep_deg(c.ra, c.dec, ra, dec))) for c in ch.itertuples()]
    return dict(gid=str(sid), type=typ, ra=ra, dec=dec, V=V, C=C, n_eligible_comp=int(len(el)),
                n_gaia_cone=int(len(ga)), n_vsx_cone=int(len(vsx)), n_frames=int(len(frames)), coverage=info), None


# ---------------------------------------------------------------- pixel reads (range requests with timeouts)
_tls = threading.local()


def _get(url, a, b):
    for i in range(3):
        try:
            if not hasattr(_tls, "s"):
                _tls.s = requests.Session()
            r = _tls.s.get(url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=(10, 30))
            r.raise_for_status()
            assert len(r.content) == b - a
            return r.content
        except Exception:
            _tls.__dict__.pop("s", None)
            if i == 2:
                raise
            time.sleep(1.5 * (i + 1))


def _hdr(url, off, buf=None, boff=0, n=23040):
    """FITS header at byte `off` (from buf if it holds it, else fetched). Returns (header, data offset)."""
    b = buf[off - boff:] if buf is not None and boff <= off < boff + len(buf) else _get(url, off, off + n)
    for k in range(0, len(b) - 79, 80):
        if b[k:k + 8] == b"END     ":
            return fits.Header.fromstring(b[:k + 80].decode("ascii")), off + (k // 2880 + 1) * 2880
    return _hdr(url, off, None, 0, 4 * max(n, len(b)))


def _pad(n):
    return (n + 2879) // 2880 * 2880


def measure_picture(key, stars):
    """All group stars in one picture. stars: list of dict(sid, nb = array [n, 5] ra, dec, pmra, pmdec, G of Gaia
    G < 19 neighbours incl. the star itself, tgt = indices of the star's blend group). Returns (rows, stamps)."""
    url = f"{BUCKET}/{key}"
    b0 = _get(url, 0, 46080)
    _, o = _hdr(url, 0, b0, 0)
    h, img_off = _hdr(url, o, b0, 0)
    assert h["EXTNAME"] == "IMAGE" and h["BITPIX"] == -32 and h["NAXIS1"] == 2040 and "BZERO" not in h
    w = WCS(h)
    mjd, det, omega, obs = float(h["MJD-AVG"]), int(h["DETECTOR"]), float(h["OMEGA_MEDIAN"]), h["OBSID"]
    yr = (mjd - GAIA_EPOCH) / 365.25
    todo = []
    for s in stars:
        nb = s["nb"]
        de = nb[:, 1] + np.nan_to_num(nb[:, 3]) * yr / 3.6e6
        ra = nb[:, 0] + np.nan_to_num(nb[:, 2]) * yr / 3.6e6 / np.cos(np.radians(nb[:, 1]))
        x, y = w.all_world2pix(ra, de, 0)
        tx, ty = float(x[s["tgt"][0]]), float(y[s["tgt"][0]])
        if not (EDGE <= tx <= 2039 - EDGE and EDGE <= ty <= 2039 - EDGE):
            continue
        todo.append((s, x, y, tx, ty))
    if not todo:
        return [], []
    fo = img_off + _pad(2040 * 2040 * 4)
    b1 = _get(url, fo, fo + 8640 + 16320)
    fh, fl_off = _hdr(url, fo, b1, fo)
    assert fh["EXTNAME"] == "FLAGS" and fh["ZCMPTYPE"] == "RICE_1" and fh["ZTILE1"] == 2040 and fh["ZTILE2"] == 1
    assert fh["NAXIS1"] == 8 and fh["NAXIS2"] == 2040 and fh.get("THEAP", 8 * 2040) == 8 * 2040
    tab = b1[fl_off - fo:fl_off - fo + 16320] if fl_off - fo + 16320 <= len(b1) else _get(url, fl_off, fl_off + 16320)
    desc_all = np.frombuffer(tab, ">i4").reshape(-1, 2)
    heap = fl_off + 8 * 2040
    vh, var_off = _hdr(url, fl_off + _pad(8 * 2040 + fh["PCOUNT"]), None, 0, 8640)
    assert vh["EXTNAME"] == "VARIANCE" and vh["BITPIX"] == -32 and "BZERO" not in vh and "BSCALE" not in vh
    codec = Rice1(blocksize=fh.get("ZVAL1", 32), bytepix=4, tilesize=2040)
    rows, stamps = [], []
    for s, x, y, tx, ty in todo:
        ix, iy = int(round(tx)), int(round(ty))
        x0, y0, x1, y1 = ix - HALF, iy - HALF, ix + HALF + 1, iy + HALF + 1
        img = np.frombuffer(_get(url, img_off + y0 * 8160, img_off + y1 * 8160), ">f4").reshape(-1, 2040)[:, x0:x1].astype(np.float32)
        var = np.frombuffer(_get(url, var_off + y0 * 8160, var_off + y1 * 8160), ">f4").reshape(-1, 2040)[:, x0:x1].astype(np.float32)
        desc = desc_all[y0:y1]
        lo, hi = int(desc[:, 1].min()), int((desc[:, 1] + desc[:, 0]).max())
        buf = _get(url, heap + lo, heap + hi)
        flg = np.stack([codec.decode(np.frombuffer(buf[o_ - lo:o_ - lo + n], np.uint8)) for n, o_ in desc]).astype(np.int32)[:, x0:x1]
        H = W = 2 * HALF + 1
        sx, sy = x - x0, y - y0
        inb = (sx >= -3) & (sx <= W + 2) & (sy >= -3) & (sy <= H + 2)
        inb[s["tgt"]] = True
        k = np.flatnonzero(inb)
        cols = e6.star_columns(np.stack([sx[k], sy[k]], 1), det, x0, y0, (H, W))
        grp = [int(np.flatnonzero(k == t)[0]) for t in s["tgt"]]
        a, e, npix = e7.forced_photometry(img, var, flg, cols, [grp])[0]
        yy, xx = np.indices((H, W))
        rr = np.hypot(xx - (tx - x0), yy - (ty - y0))
        bad = ((flg & e7.PHOT_BITS) != 0) | ~np.isfinite(var) | (var <= 0) | ~np.isfinite(img)
        rows.append(dict(sid=s["sid"], key=key, obs_id=obs, det=det, mjd=mjd, x=tx, y=ty,
                         lam=float(wave_map(det, tx, ty)),
                         flux=flux_mJy(a, omega) if np.isfinite(a) else np.nan,
                         err=flux_mJy(e, omega) if np.isfinite(e) else np.nan, npix=int(npix),
                         sat=bool((flg[rr <= 1.0] & e7.SAT_BITS).any()), nmask2=int(bad[rr <= 2.0].sum()),
                         flags_or2=int(np.bitwise_or.reduce(flg[rr <= 2.0])), n_cols=len(k)))
        stamps.append(np.stack([img, var, flg.astype(np.float32)]))
    return rows, stamps


def _chunk(args):
    keys, stars = args
    out_r, out_s, errs = [], [], []

    def one(k):
        try:
            return measure_picture(k, stars)
        except Exception as ex:
            return repr(ex)[:200]

    with ThreadPoolExecutor(THREADS) as tp:
        for k, res in zip(keys, tp.map(one, keys)):
            if isinstance(res, str):
                errs.append(dict(key=k, error=res))
            else:
                out_r += res[0]
                out_s += res[1]
    return out_r, out_s, errs


def group_stars(g):
    ga = pd.read_parquet(f"{D}/groups/{g['gid']}_gaia.parquet")
    arr = ga[["ra", "dec", "pmra", "pmdec", "phot_g_mean_mag"]].values.astype(float)
    ids = ga.source_id.values
    out = []
    for st in [g["V"]] + g["C"]:
        s = sep_deg(arr[:, 0], arr[:, 1], st["ra"], st["dec"])
        k = np.flatnonzero(s < R_NB)
        me = np.flatnonzero(ids[k] == st["source_id"])
        assert len(me) == 1, (st["source_id"], len(me))
        bl = np.flatnonzero(s[k] * 3600 < e7.BLEND_ARCSEC)
        tgt = [int(me[0])] + [int(q) for q in bl if q != me[0]]
        out.append(dict(sid=int(st["source_id"]), nb=arr[k], tgt=tgt, nb_ids=ids[k]))
    return out


def phot_group(g, ex, outdir):
    p = f"{outdir}/{g['gid']}.parquet"
    if os.path.exists(p):
        return None
    t0 = time.time()
    frames, _ = select_frames(g["V"]["ra"], g["V"]["dec"])     # frozen rule with the final cap (CAP)
    frames.to_parquet(f"{outdir}/{g['gid']}_frames.parquet", index=False)
    stars = [dict(sid=s["sid"], nb=s["nb"], tgt=s["tgt"]) for s in group_stars(g)]
    keys = frames.key.tolist()
    chunks = [keys[i::PROCS] for i in range(PROCS)]
    futs = [ex.submit(_chunk, (c, stars)) for c in chunks if c]
    done, pending = wait(futs, timeout=max(240, 4 * len(keys)))          # whole-group budget; stragglers are dropped
    rows, stamps, errs = [], [], []
    for f in done:
        try:
            r, s, e = f.result()
            rows += r
            stamps += s
            errs += e
        except Exception as ex_:
            errs.append(dict(key="chunk", error=repr(ex_)[:200]))
    for f in pending:
        f.cancel()
    df = pd.DataFrame(rows)
    if len(df):
        df = df.merge(frames[["key", "window"]], on="key", how="left")
        df["gid"] = g["gid"]
        np.savez_compressed(f"{outdir}/{g['gid']}_stamps.npz", stamps=np.array(stamps, np.float32),
                            sid=df.sid.values, key=df.key.values.astype(str))
    df.to_parquet(p + ".tmp", index=False)
    os.replace(p + ".tmp", p)
    rec = dict(gid=g["gid"], n_frames=len(keys), n_rows=len(df), n_errors=len(errs), n_chunks_timed_out=len(pending),
               errors=errs[:20], seconds=round(time.time() - t0, 1))
    json.dump(rec, open(f"{outdir}/{g['gid']}.log.json", "w"))
    print(json.dumps({k: rec[k] for k in ("gid", "n_frames", "n_rows", "n_errors", "n_chunks_timed_out", "seconds")}), flush=True)
    return rec


def tp_pseudo():
    """Throughput timing on 5 stars that are NOT in the sample (around an E7 control-field star)."""
    gid, sid0 = "tp_e2_00", 3494399346375718656
    ga = pd.read_parquet(f"{D}/groups/{gid}_gaia.parquet")
    me = ga[ga.source_id == sid0]
    ra, dec = float(me.ra.iloc[0]), float(me.dec.iloc[0])
    s = sep_deg(ga.ra.values, ga.dec.values, ra, dec)
    el = ga[(s > 0.05) & (s < 0.5) & (ga.phot_g_mean_mag > 9) & (ga.phot_g_mean_mag < 14)].sort_values("source_id")
    ch = el.iloc[np.random.default_rng(1).choice(len(el), 4, replace=False)]
    g = dict(gid=gid, V=dict(source_id=sid0, ra=ra, dec=dec),
             C=[dict(source_id=int(c.source_id), ra=float(c.ra), dec=float(c.dec)) for c in ch.itertuples()])
    os.makedirs(f"{D}/throughput", exist_ok=True)
    with ProcessPoolExecutor(PROCS) as ex:
        print(phot_group(g, ex, f"{D}/throughput"))


def load_sample():
    """First draw plus, if present, the pre-registered extension (groups appended; logs merged)."""
    S = json.load(open(f"{D}/sample.json"))
    if os.path.exists(f"{D}/sample_ext.json"):
        E = json.load(open(f"{D}/sample_ext.json"))
        S["groups"] = S["groups"] + E["groups"]
        S["rejected"] = S["rejected"] + E["rejected"]
        S["extension"] = dict(seed=E["seed"], comp_seed=E["comp_seed"], split_seed=E["split_seed"], pool_log=E["pool_log"],
                              n_V=E["n_V"], n_C=E["n_C"], drawn_local=E["drawn_local"])
    return S


def throughput():
    S = json.load(open(f"{D}/sample.json"))
    os.makedirs(f"{D}/throughput", exist_ok=True)
    with ProcessPoolExecutor(PROCS) as ex:
        for g in S["spare"]:
            phot_group(g, ex, f"{D}/throughput")


def phot():
    S = load_sample()
    groups = S["groups"][:int(sys.argv[2])] if len(sys.argv) > 2 else S["groups"]
    os.makedirs(f"{D}/phot", exist_ok=True)
    ex = ProcessPoolExecutor(PROCS)
    for i, g in enumerate(groups):
        try:
            r = phot_group(g, ex, f"{D}/phot")
        except Exception as e:
            print(g["gid"], "FAILED", repr(e)[:200], flush=True)
            continue
        if r and r["n_chunks_timed_out"]:                 # a hung worker: replace the pool
            ex.shutdown(wait=False, cancel_futures=True)
            ex = ProcessPoolExecutor(PROCS)
    ex.shutdown(wait=False, cancel_futures=True)
    print("phot done", time.strftime("%H:%M:%S"), flush=True)


# ---------------------------------------------------------------- per-star analysis
def usable(df):
    return df[np.isfinite(df.flux) & np.isfinite(df.err) & (df.err > 0) & (df.npix >= e7.MIN_USABLE_PIX) & ~df.sat]


def prep(df, s_d, clean=False):
    """Usable pictures of one star with Gate-0 errors; drops windows / detectors with too few pictures."""
    n_all = len(df)
    sat_frac = float(df.sat.mean()) if n_all else np.nan
    u = usable(df)
    if clean:
        u = u[u.nmask2 == 0]
    u = u[u.groupby("window").flux.transform("size") >= MIN_WIN_PICS]
    u = u[u.groupby("det").flux.transform("size") >= MIN_DET_PICS]
    u = u[u.groupby("window").flux.transform("size") >= MIN_WIN_PICS].sort_values("mjd")
    d = dict(n_all=n_all, sat_frac=sat_frac, n_usable=len(u), n_windows=int(u.window.nunique()) if len(u) else 0)
    if len(u) < MIN_PICS or d["n_windows"] < 2:
        d["status"] = "not testable"
        return d
    d.update(status="ok", flux=u.flux.values, e0=u.err.values * np.array([s_d[str(x)] for x in u.det]),
             lam=u.lam.values, det=u.det.values, win=u.window.values, mjd=u.mjd.values)
    return d


def errors(d, f_d):
    """sigma^2 = (s_d e)^2 + (f_d |static model|)^2, model iterated twice."""
    f = np.array([f_d[str(x)] for x in d["det"]])
    e = d["e0"]
    for _ in range(2):
        m = e7.static_fit(d["flux"], e, d["lam"], d["det"])[0]
        e = np.sqrt(d["e0"] ** 2 + (f * np.abs(m)) ** 2)
    return e


def test_star(d, f_d):
    e = errors(d, f_d)
    _, per, chi_s, npar = e7.static_fit(d["flux"], e, d["lam"], d["det"])
    c = e7.change_fit(d["flux"], e, d["lam"], d["det"], d["win"])
    out = dict(T=chi_s - c["chi2"], chi2_static=chi_s, dof_static=len(e) - npar,
               windows=[int(x) for x in c["windows"]], a_w=[float(x) for x in c["a"]],
               a_err=[float(x) for x in c["a_err"]])
    terms = []
    for dd in np.unique(d["det"]):
        s = d["det"] == dd
        wn = pd.Series(d["win"][s]).value_counts()
        s &= np.isin(d["win"], wn[wn >= MIN_DW_PICS].index)
        if len(np.unique(d["win"][s])) < 2:
            continue
        cd = e7.change_fit(d["flux"][s], e[s], d["lam"][s], d["det"][s], d["win"][s])
        for w_, a_, ae_ in zip(cd["windows"][1:], cd["a"][1:], cd["a_err"][1:]):
            if np.isfinite(ae_) and ae_ > 0:
                terms.append(dict(det=int(dd), window=int(w_), ref_window=int(cd["windows"][0]), a=float(a_),
                                  err=float(ae_), n=int(s.sum())))
    out["det_terms"] = terms
    out["M"] = max([abs(t["a"]) / t["err"] for t in terms], default=0.0)
    return out


def n_changed(terms, dcut):
    """Largest number of detectors with |a_dw|/err > dcut in the same window and with the same sign."""
    best = 0
    for w_ in {t["window"] for t in terms}:
        for sg in (1, -1):
            best = max(best, sum(1 for t in terms if t["window"] == w_ and sg * t["a"] / t["err"] > dcut))
    return best


def load_phot(gids):
    return pd.concat([pd.read_parquet(f"{D}/phot/{g}.parquet") for g in gids if os.path.exists(f"{D}/phot/{g}.parquet")],
                     ignore_index=True)


def star_table(S):
    rows = []
    for g in S["groups"]:
        rows.append(dict(sid=g["V"]["source_id"], gid=g["gid"], cls="V", half="", type=g["type"], G=g["V"]["G"],
                         amp=g["V"]["amp"], ra=g["V"]["ra"], dec=g["V"]["dec"]))
        for c in g["C"]:
            rows.append(dict(sid=c["source_id"], gid=g["gid"], cls="C", half=c["half"], type="", G=c["G"],
                             amp=np.nan, ra=c["ra"], dec=c["dec"]))
    return pd.DataFrame(rows)


def calibrate():
    if os.path.exists(THR):
        sys.exit(f"{THR} exists (frozen)")
    S = load_sample()
    s_d = json.load(open(GATE))["s_d"]
    st = star_table(S)
    done = {g["gid"] for g in S["groups"] if os.path.exists(f"{D}/phot/{g['gid']}.parquet")}
    cal = st[(st.cls == "C") & (st.half == "calibration") & st.gid.isin(done)]
    ph = load_phot(sorted(cal.gid.unique()))
    ph = ph[ph.sid.isin(set(cal.sid))]                           # calibration half ONLY
    data = {sid: prep(df, s_d) for sid, df in ph.groupby("sid")}
    ok = {k: v for k, v in data.items() if v["status"] == "ok"}
    f_d, red0, red1 = {}, {}, {}
    for dd in "123456":
        def med(f):
            v = []
            for d in ok.values():
                s = d["det"] == int(dd)
                if s.sum() <= 5:
                    continue
                sub = dict(flux=d["flux"][s], e0=d["e0"][s], lam=d["lam"][s], det=d["det"][s])
                e = errors(sub, {dd: f})
                v.append(e7.static_fit(sub["flux"], e, sub["lam"], sub["det"])[2] / (s.sum() - 3))
            return float(np.median(v)), len(v)
        red0[dd], n = med(0.0)
        if red0[dd] <= 1:
            f_d[dd] = 0.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if med(mid)[0] > 1 else (lo, mid)
            f_d[dd] = hi
        red1[dd] = med(f_d[dd])[0]
        print("detector", dd, "n", n, "median red chi2 before", round(red0[dd], 2), "floor", round(f_d[dd], 4), flush=True)
    res = {k: test_star(v, f_d) for k, v in ok.items()}
    T = np.sort([r["T"] for r in res.values()])[::-1]
    M = np.sort([r["M"] for r in res.values()])[::-1]
    n = len(T)
    k = int(np.floor(0.01 * n))
    dts = np.array([abs(t["a"]) / t["err"] for r in res.values() for t in r["det_terms"]])
    aw = np.array([a for r in res.values() for a in r["a_w"][1:]])
    out = dict(experiment="E10", stage="thresholds frozen on the comparison-star calibration half only",
               n_calibration_drawn=int(len(cal)), n_calibration_testable=n,
               n_not_testable=int(len(data) - n), gate0_s_d=s_d, floor_f_d=f_d,
               median_red_chi2_before_floor=red0, median_red_chi2_after_floor=red1,
               rule_1pct="flagged iff T > T_cut_1pct, T_cut_1pct = the (floor(0.01 n)+1)-th largest calibration T",
               T_cut_1pct=float(T[k]), M_cut_1pct=float(M[k]), T_cut_candidate=float(T[0]),
               det_term_cut=float(np.percentile(dts, 99)) if len(dts) else None, n_det_terms=int(len(dts)),
               calibration_T_sorted=[float(x) for x in T], calibration_M_sorted=[float(x) for x in M],
               calibration_a_w=dict(n=int(len(aw)), sigma_mad=float(1.4826 * np.median(np.abs(aw - np.median(aw)))),
                                    std=float(np.std(aw)), abs_p95=float(np.percentile(np.abs(aw), 95)),
                                    abs_max=float(np.abs(aw).max())),
                groups_used=sorted(done), frozen_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(out, open(THR, "w"), indent=1)
    print(json.dumps({k_: out[k_] for k_ in ("n_calibration_testable", "floor_f_d", "T_cut_1pct", "M_cut_1pct",
                                             "T_cut_candidate", "det_term_cut", "calibration_a_w")}, indent=1))


def analyze():
    S = load_sample()
    th = json.load(open(THR))
    s_d, f_d = th["gate0_s_d"], th["floor_f_d"]
    st = star_table(S)
    st = st[st.gid.isin(th["groups_used"])].reset_index(drop=True)
    ph = load_phot(th["groups_used"])
    by = dict(tuple(ph.groupby("sid")))
    recs = []
    for r in st.itertuples():
        df = by.get(r.sid)
        rec = dict(r._asdict())
        rec.pop("Index")
        if df is None:
            rec.update(status="no pictures", n_all=0)
            recs.append(rec)
            continue
        d = prep(df, s_d)
        rec.update(status=d["status"], n_all=d["n_all"], n_usable=d["n_usable"], n_windows=d["n_windows"],
                   sat_frac=d["sat_frac"], frac_nmask2=float((df.nmask2 > 0).mean()))
        if d["status"] == "ok":
            t = test_star(d, f_d)
            rec.update(T=t["T"], M=t["M"], chi2_static=t["chi2_static"], dof_static=t["dof_static"],
                       windows=json.dumps(t["windows"]), a_w=json.dumps(t["a_w"]), a_err=json.dumps(t["a_err"]),
                       det_terms=json.dumps(t["det_terms"]), n_det_changed=n_changed(t["det_terms"], th["det_term_cut"]),
                       max_abs_a_w=float(np.max(np.abs(t["a_w"]))), flag_T=bool(t["T"] > th["T_cut_1pct"]),
                       flag_M=bool(t["M"] > th["M_cut_1pct"]), above_candidate_cut=bool(t["T"] > th["T_cut_candidate"]))
            dc = prep(df, s_d, clean=True)
            rec["T_clean"] = test_star(dc, f_d)["T"] if dc["status"] == "ok" else np.nan
            rec["n_clean"] = dc["n_usable"]
        recs.append(rec)
    R = pd.DataFrame(recs)
    R.to_parquet(STARS, index=False)
    report(R, S, th)


def rate(mask, sel):
    k, n = int((mask & sel).sum()), int(sel.sum())
    return dict(k=k, n=n, rate=round(k / n, 4) if n else None, ci95=wilson(k, n))


def report(R, S, th):
    ok = R.status == "ok"
    fT, fM = R.get("flag_T", False) == True, R.get("flag_M", False) == True       # noqa: E712
    V, C = R.cls == "V", R.cls == "C"
    test, cal = C & (R.half == "test"), C & (R.half == "calibration")
    rec = dict(all_variables=rate(fT, V), testable_variables=rate(fT, V & ok),
               by_type={t: dict(all=rate(fT, V & (R.type == t)), testable=rate(fT, V & ok & (R.type == t)))
                        for t in QUOTA},
               by_amplitude={f"{lo}-{hi}": dict(all=rate(fT, V & (R.amp >= lo) & (R.amp < hi)),
                                                testable=rate(fT, V & ok & (R.amp >= lo) & (R.amp < hi)))
                             for lo, hi in AMP_BINS},
               by_type_and_amplitude={t: {f"{lo}-{hi}": rate(fT, V & ok & (R.type == t) & (R.amp >= lo) & (R.amp < hi))
                                          for lo, hi in AMP_BINS} for t in QUOTA},
               secondary_M=dict(all_variables=rate(fM, V), testable_variables=rate(fM, V & ok),
                                by_type={t: rate(fM, V & ok & (R.type == t)) for t in QUOTA}))
    fp = dict(test_half_T=rate(fT, test & ok), test_half_M=rate(fM, test & ok),
              calibration_half_T=rate(fT, cal & ok), n_test_drawn=int(test.sum()), n_test_testable=int((test & ok).sum()))
    aw = np.array([a for s in R[test & ok].a_w for a in json.loads(s)[1:]])
    aw_err = np.array([a for s in R[test & ok].a_err for a in json.loads(s)[1:]])
    flagged_v = R[V & ok & fT]
    floor = dict(per_picture_floor_f_d_percent={d: round(100 * f, 2) for d, f in th["floor_f_d"].items()},
                 test_half_a_w=dict(n=int(len(aw)), sigma_mad_percent=round(100 * 1.4826 * float(np.median(np.abs(aw - np.median(aw)))), 2),
                                    abs_p95_percent=round(100 * float(np.percentile(np.abs(aw), 95)), 2),
                                    abs_max_percent=round(100 * float(np.abs(aw).max()), 2),
                                    median_formal_err_percent=round(100 * float(np.median(aw_err)), 2)) if len(aw) else None,
                 smallest_abs_a_w_among_flagged_variables_percent=round(100 * float(flagged_v.max_abs_a_w.min()), 2) if len(flagged_v) else None,
                 note="sigma_mad of a_w among test-half comparison stars is the empirical 1-sigma error of a between-visit "
                      "brightness factor; changes below ~3x that are not separable from ordinary stars")
    lp = rec["by_type"]["LPV"]["testable"]
    t_ = fp["test_half_T"]
    if lp["n"] < 15 or t_["n"] < 60:
        verdict = "INCONCLUSIVE"
    elif t_["ci95"][0] > 0.01 or lp["ci95"][1] < 0.5:
        verdict = "FAILED"
    elif lp["ci95"][0] > 0.5 and t_["ci95"][0] <= 0.01:
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"
    out = dict(experiment="E10", protocol="docs/PROTOCOLS.md PHASE 2 E10 + docs/protocols/E10.md", index=INDEX,
               n_index_frames=S["n_index_frames"], seeds=dict(sample=S["seed"], split=S["split_seed"]),
               sample=dict(extension=S.get("extension"), n_groups_drawn=len(S["groups"]), n_groups_used=len(th["groups_used"]), quota=S["quota"],
                           pool_log=S["pool_log"], n_rejected_draws=len(S["rejected"]),
                           n_V=int(V.sum()), n_V_testable=int((V & ok).sum()), n_C=int(C.sum()),
                           n_C_calibration=int(cal.sum()), n_C_test=int(test.sum()),
                           n_C_calibration_testable=int((cal & ok).sum()), n_C_test_testable=int((test & ok).sum()),
                           V_by_type={t: int((V & (R.type == t)).sum()) for t in QUOTA},
                           V_testable_by_type={t: int((V & ok & (R.type == t)).sum()) for t in QUOTA},
                           status_counts_V=R[V].status.value_counts().to_dict(),
                           status_counts_C=R[C].status.value_counts().to_dict(),
                           median_usable_pictures=float(R[ok].n_usable.median()),
                           median_saturated_fraction_V=float(R[V].sat_frac.median()),
                           median_saturated_fraction_C=float(R[C].sat_frac.median()),
                           definition="docs/protocols/E10.md; data/phase2/e10/sample.json"),
               thresholds={k: th[k] for k in ("T_cut_1pct", "M_cut_1pct", "T_cut_candidate", "det_term_cut",
                                              "n_calibration_testable", "floor_f_d", "gate0_s_d", "frozen_local")},
               thresholds_file=THR, recall=rec, false_positive=fp, floor=floor, verdict=verdict,
               verdict_rule="docs/protocols/E10.md 'Question and verdict criterion'",
               stars_file=STARS, generated_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               prior_work=["Huang, Marengo et al. 2026, ApJ accepted, arXiv:2609.30551 — SPHEREx spectra of LMC AGB "
                           "variables (per-phase brightness factors with an external period)",
                           "Karambelkar, De et al. 2026, ApJL, arXiv:2602.07129 — SPHEREx spectra of Galactic transients",
                           "Borlaff et al. 2026, AJ, arXiv:2605.27501 — satellite trails in ~73% of SPHEREx images "
                           "(~2.2 per exposure); motivates the single-picture check on candidates"],
               reread_rule="groups with read errors on > 10% of their pictures (network timeouts) were re-read once after "
                           "the first pass (docs/protocols/E10.md D-E10-4); the re-read replaces the first pass")
    if os.path.exists(OUT):                                   # keep sections written by later stages
        old = json.load(open(OUT))
        for k in ("candidates", "deviations", "independent_light_curves"):
            if k in old:
                out[k] = old[k]
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    figure(R, rec)
    print(json.dumps(dict(verdict=verdict, recall_all=rec["all_variables"], recall_testable=rec["testable_variables"],
                          by_type={t: v["testable"] for t, v in rec["by_type"].items()},
                          by_amp={a: v["testable"] for a, v in rec["by_amplitude"].items()},
                          fp=fp, floor=floor, sample=out["sample"]), indent=1, default=float))


def figure(R, rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(figsize=(10, 7.6), facecolor="white")
    labels = ["0.3 to 0.5", "0.5 to 1", "more than 1"]
    series = [("All measurable variable stars", "#222222", "o", lambda a: rec["by_amplitude"][a]["testable"]),
              ("Slow variables (long-period)", "#D55E00", "s", lambda a: rec["by_type_and_amplitude"]["LPV"][a]),
              ("Fast variables (RR Lyrae, Cepheids, eclipsing)", "#0072B2", "^", lambda a: _sum(
                  [rec["by_type_and_amplitude"][t][a] for t in ("RR", "CEP", "ECL")]))]
    for j, (name, col, mk, get) in enumerate(series):
        for i, (lo, hi) in enumerate(AMP_BINS):
            r = get(f"{lo}-{hi}")
            if not r["n"]:
                continue
            x = i + (j - 1) * 0.27
            ax.errorbar(x, r["rate"], [[r["rate"] - r["ci95"][0]], [r["ci95"][1] - r["rate"]]], fmt=mk, color=col, ms=11,
                        capsize=6, lw=2, label=name if i == min(k for k, (l2, h2) in enumerate(AMP_BINS) if get(f"{l2}-{h2}")["n"]) else None)
            ax.annotate(f"{r['k']} of {r['n']}", (x, r["ci95"][1]), textcoords="offset points", xytext=(0, 7), ha="center",
                        fontsize=12, color=col)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("Catalogue brightness swing in visible light (magnitudes, Gaia G)")
    ax.set_ylabel("Fraction flagged as changed")
    ax.set_title("Known variable stars recovered by the change test\n(bars: 95% intervals; cut set for 1 false alarm per 100 ordinary stars)", fontsize=15)
    ax.grid(axis="y", color="0.88")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=13, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    nsat = int(((R.cls == "V") & (R.status != "ok")).sum())
    fig.text(0.5, 0.01, f"Not shown: {nsat} variable stars too bright for SPHEREx (saturated in most pictures), all long-period.",
             ha="center", fontsize=12, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150, facecolor="white")
    plt.close(fig)


def _sum(rs):
    k, n = sum(r["k"] for r in rs), sum(r["n"] for r in rs)
    return dict(k=k, n=n, rate=k / n if n else None, ci95=wilson(k, n))


# ---------------------------------------------------------------- unlisted candidates
CATS = {"VSX": ("B/vsx/vsx", "OID,Name,Type,max,min,Period"),
        "ASAS-SN variables (II/366)": ("II/366/catalog", "ID,Type,Vmag,Amp,Per"),
        "ZTF periodic variables (J/ApJS/249/18)": ("J/ApJS/249/18/table2", "ID,Type,Per,rmag"),
        "ATLAS variables (J/AJ/156/241)": ("J/AJ/156/241/table4", "ATOID,Class,gmag")}
CANDFIG = "site/data/figs/e10_candidates.png"


def ratio_series(df, s_d, f_d):
    """Per usable picture: flux / static model and its error, for plots."""
    d = prep(df, s_d)
    if d["status"] != "ok":
        return None
    e = errors(d, f_d)
    m = e7.static_fit(d["flux"], e, d["lam"], d["det"])[0]
    return pd.DataFrame(dict(mjd=d["mjd"], det=d["det"], win=d["win"], r=d["flux"] / m, re=e / np.abs(m)))


def T_only(d, f_d):
    e = errors(d, f_d)
    return e7.static_fit(d["flux"], e, d["lam"], d["det"])[2] - e7.change_fit(d["flux"], e, d["lam"], d["det"], d["win"])["chi2"]


def loo_check(df, s_d, f_d, cut):
    """A-E10-3: leave-one-picture-out and leave-the-two-most-influential-out T; per detector-window medians."""
    d = prep(df, s_d)
    keys = ("flux", "e0", "lam", "det", "win", "mjd")
    n = len(d["flux"])

    def drop(idx):
        m = np.ones(n, bool)
        m[list(idx)] = False
        return {k: d[k][m] for k in keys}

    T0 = T_only(d, f_d)
    Tl = np.array([T_only(drop([i]), f_d) for i in range(n)])
    two = np.argsort(Tl)[:2]
    T2 = T_only(drop(two), f_d)
    e = errors(d, f_d)
    m = e7.static_fit(d["flux"], e, d["lam"], d["det"])[0]
    q = pd.DataFrame(dict(det=d["det"], win=d["win"], r=d["flux"] / m))
    med = [dict(det=int(a), window=int(b), median_ratio_percent=round(100 * (float(g.r.median()) - 1), 2), n=int(len(g)))
           for (a, b), g in q.groupby(["det", "win"])]
    return dict(T_all=float(T0), min_T_leave_one_out=float(Tl.min()), T_without_two_most_influential_pictures=float(T2),
                most_influential_pictures_mjd=[float(d["mjd"][i]) for i in two],
                carried_by_one_or_two_pictures=bool(T2 <= cut), per_detector_window_median=med)


def candidates():
    S = load_sample()
    th = json.load(open(THR))
    R = pd.read_parquet(STARS)
    G = {g["gid"]: g for g in S["groups"]}
    C = R[(R.cls == "C") & (R.status == "ok")]
    above = C[C["T"] > th["T_cut_candidate"]]
    n_cal, n_test = int((C.half == "calibration").sum()), int((C.half == "test").sum())
    listed, ph_cache = [], {}
    for r in above.itertuples():
        ga = pd.read_parquet(f"{D}/groups/{r.gid}_gaia.parquet")
        s = sep_deg(ga.ra.values, ga.dec.values, r.ra, r.dec) * 3600
        nb = ga[(s < 12) & (ga.source_id != r.sid) & (ga.phot_g_mean_mag < r.G + 2)]
        ph = ph_cache.setdefault(r.gid, pd.read_parquet(f"{D}/phot/{r.gid}.parquet"))
        df = ph[ph.sid == r.sid]
        terms = json.loads(r.det_terms)
        cuts = dict(changed_in_2_detectors=bool(r.n_det_changed >= 2),
                    T_from_pictures_without_masked_pixels_within_2px=None if not np.isfinite(r.T_clean) else float(r.T_clean),
                    clean_T_above_cut=bool(np.isfinite(r.T_clean) and r.T_clean > th["T_cut_candidate"]),
                    no_bright_neighbour_within_12arcsec=bool(len(nb) == 0))
        rec = dict(gaia_dr3_source_id=str(r.sid), ra=r.ra, dec=r.dec, G=r.G, half=r.half, group_variable=r.gid,
                   T=float(r.T), M=float(r.M), a_w=json.loads(r.a_w), a_err=json.loads(r.a_err),
                   per_detector=[dict(det=t["det"], window=t["window"], a_percent=round(100 * t["a"], 2),
                                      err_percent=round(100 * t["err"], 2), n_pictures=t["n"]) for t in terms],
                   n_usable=int(r.n_usable), saturated_fraction=float(r.sat_frac),
                   fraction_pictures_with_masked_pixel_within_2px=float(r.frac_nmask2),
                   min_distance_to_detector_edge_px=float(np.minimum(np.minimum(df.x, 2039 - df.x), np.minimum(df.y, 2039 - df.y)).min()),
                   median_sources_fitted_in_stamp=float(df.n_cols.median()),
                   neighbours_within_12arcsec=[dict(source_id=str(q.source_id), G=float(q.phot_g_mean_mag)) for q in nb.itertuples()],
                   brightest_gaia_within_30arcsec=float(ga[(s < 30) & (ga.source_id != r.sid)].phot_g_mean_mag.min()) if ((s < 30) & (ga.source_id != r.sid)).any() else None,
                   cuts=cuts, passes_frozen_rule=bool(cuts["changed_in_2_detectors"] and cuts["clean_T_above_cut"] and
                                                      cuts["no_bright_neighbour_within_12arcsec"]))
        if rec["passes_frozen_rule"]:
            rec["single_picture_check"] = loo_check(df, th["gate0_s_d"], th["floor_f_d"], th["T_cut_candidate"])
            chk = {}
            for name, (src, out) in CATS.items():
                t = vizier_cone(src, r.ra, r.dec, 10 / 3600, out)
                chk[name] = "unreachable" if t is None else ("no entry within 10 arcsec" if len(t) == 0 else t.to_dict("records"))
            try:
                gv = tap(f"SELECT source_id, phot_variable_flag FROM gaiadr3.gaia_source WHERE source_id = {r.sid}")
                vs = tap(f"SELECT source_id, range_mag_g_fov FROM gaiadr3.vari_summary WHERE source_id = {r.sid}")
                chk["Gaia DR3"] = dict(phot_variable_flag=str(gv.phot_variable_flag.iloc[0]), in_vari_summary=bool(len(vs)))
            except Exception as e:
                chk["Gaia DR3"] = "unreachable: " + repr(e)[:80]
            rec["catalogue_checks"] = chk
        listed.append(rec)
    surv = [c for c in listed if c["passes_frozen_rule"]]
    out = json.load(open(OUT))
    out["candidates"] = dict(
        rule="docs/protocols/E10.md 'Candidate rule'", T_cut_candidate=th["T_cut_candidate"], det_term_cut=th["det_term_cut"],
        n_comparison_testable=dict(calibration=n_cal, test=n_test),
        expected_by_chance_T_only=round(n_test / (n_cal + 1), 2),
        expected_note="a test-half star exceeds the calibration maximum of T by chance with probability 1/(n_cal+1); "
                      "the further cuts lower this by an unmeasured factor",
        n_above_T_cut=int(len(listed)), n_passing_frozen_rule=len(surv), stars_above_T_cut=listed,
        wording="candidate variables not found in the catalogues we checked" if surv else "no candidates")
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(json.dumps(out["candidates"], indent=1, default=float))
    show = surv if surv else []
    if show:
        candidate_figure(show, R, th, ph_cache)


def candidate_figure(show, R, th, ph_cache, label="Candidate"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14})
    colors = {1: "#0072B2", 2: "#56B4E9", 3: "#009E73", 4: "#E69F00", 5: "#D55E00", 6: "#CC79A7"}
    bands = {1: "0.75-1.1", 2: "1.1-1.6", 3: "1.6-2.4", 4: "2.4-3.8", 5: "3.8-4.4", 6: "4.4-5.0"}
    fig, axs = plt.subplots(len(show), 2, figsize=(13, 4.6 * len(show)), squeeze=False, facecolor="white", sharey="row")
    for i, c in enumerate(show):
        sid = int(c["gaia_dr3_source_id"])
        ph = ph_cache[c["group_variable"]]
        peers = R[(R.gid == c["group_variable"]) & (R.cls == "C") & (R.status == "ok") & (R.sid != sid)]
        comp = peers.iloc[(peers["T"] - peers["T"].median()).abs().argsort()].iloc[0] if len(peers) else None
        for j, (who, s_, G_) in enumerate([(label, sid, c["G"])] + ([("Comparison star", int(comp.sid), comp.G)] if comp is not None else [])):
            rs = ratio_series(ph[ph.sid == s_], th["gate0_s_d"], th["floor_f_d"])
            ax = axs[i, j]
            t0 = rs.mjd.min()
            for d_ in sorted(rs.det.unique()):
                q = rs[rs.det == d_]
                ax.errorbar(q.mjd - t0, 100 * (q.r - 1), 100 * q.re, fmt="o", ms=4, lw=0.8, color=colors[int(d_)],
                            label=f"{bands[int(d_)]} µm" if (i == 0 and j == 0) else None)
            ax.axhline(0, color="0.5", lw=1)
            ax.set_title(f"{who}: Gaia DR3 {s_} (G = {G_:.1f})", fontsize=13)
            ax.set_xlabel(f"Days since first picture (MJD {t0:.0f})")
            if j == 0:
                ax.set_ylabel("Brightness vs. the star's\nown average (%)")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="0.9")
    axs[0, 0].legend(fontsize=10, ncol=2, frameon=False, title="wavelength band")
    fig.tight_layout()
    fig.savefig(CANDFIG, dpi=140, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- independent light curves (A-E10-2)
CONFFIG = "site/data/figs/e10_confirmed.png"
ZTF = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"


def lc_test(t, mag, err):
    """Frozen A-E10-2 criterion on one light curve."""
    t, mag, err = (np.asarray(x, float) for x in (t, mag, err))
    ok = np.isfinite(t) & np.isfinite(mag) & np.isfinite(err) & (err > 0)
    t, mag, err = t[ok], mag[ok], err[ok]
    out = dict(n_epochs=int(len(t)))
    if len(t) < 20:
        return dict(out, confirmed=False, reason="fewer than 20 good epochs")
    amp = float(np.percentile(mag, 95) - np.percentile(mag, 5))
    me = float(np.median(err))
    out.update(amplitude_p5_p95_mag=round(amp, 4), median_error_mag=round(me, 4), amplitude_over_error=round(amp / me, 2),
               time_span_d=round(float(t.max() - t.min()), 1), confirmed=bool(amp > 5 * me and amp > 0.1))
    try:
        from astropy.timeseries import LombScargle
        fr, pw = LombScargle(t, mag, err).autopower(minimum_frequency=1 / 1000, maximum_frequency=20, samples_per_peak=5)
        k = int(np.argmax(pw))
        out.update(lomb_scargle_peak_period_d=round(float(1 / fr[k]), 5), lomb_scargle_peak_power=round(float(pw[k]), 3),
                   lomb_scargle_note="descriptive only; daily aliases not removed")
    except Exception as e:
        out["lomb_scargle_note"] = "failed: " + repr(e)[:80]
    return out


def ztf_lc(ra, dec, sid=None):
    """IRSA ZTF light curve in a 1.1 arcsec cone, cached as data/phase2/e10/ztf/<sid>.csv (raw service output)."""
    cp = f"{D}/ztf/{sid}.csv" if sid is not None else None
    try:
        if cp and os.path.exists(cp):
            text = open(cp).read()
        elif os.environ.get("E10_ZTF_CACHE_ONLY"):
            return None, "unreachable: IRSA gave no answer within ~30 min on 2026-11-15 (D-E10-9)"
        else:
            r = requests.get(ZTF, params={"POS": f"CIRCLE {ra:.6f} {dec:.6f} 0.0003", "FORMAT": "csv", "BAD_CATFLAGS_MASK": 32768},
                             headers=UA, timeout=(15, 900))
            r.raise_for_status()
            text = r.text
            if cp:
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                open(cp, "w").write(text)
        df = pd.read_csv(io.StringIO(text))
    except Exception as e:
        return None, "unreachable: " + repr(e)[:100]
    return df[df.catflags == 0] if len(df) else df, "ok"


def independent(ra, dec, sid):
    """All independent light curves we can reach for one star. Returns (records, raw curves for plotting)."""
    recs, curves = [], {}
    z, st = ztf_lc(ra, dec, sid)
    if z is None:
        recs.append(dict(survey="ZTF (IRSA)", status=st, confirmed=False))
    elif len(z) == 0:
        recs.append(dict(survey="ZTF (IRSA)", status="no good epochs (not covered or saturated)", confirmed=False))
    else:
        for fc, q in z.groupby("filtercode"):
            q = q[q.oid == q.oid.value_counts().index[0]]            # one ZTF object id (field/CCD) per filter
            recs.append(dict(survey="ZTF (IRSA)", filter=str(fc), status="ok", median_mag=round(float(q.mag.median()), 3),
                             **lc_test(q.mjd, q.mag, q.magerr)))
            curves[f"ZTF {fc}"] = (q.mjd.values, q.mag.values, q.magerr.values)
    try:
        has = tap(f"SELECT has_epoch_photometry FROM gaiadr3.gaia_source WHERE source_id = {sid}")
        g = has.iloc[:0]
        if len(has) and str(has.iloc[0, 0]).lower() in ("true", "1", "t"):
            g = tap(f"SELECT g_transit_time, g_transit_mag, g_transit_flux_over_error FROM gaiadr3.epoch_photometry WHERE source_id = {sid}")
        if len(g) == 0:
            recs.append(dict(survey="Gaia DR3 epoch photometry (AIP mirror)", status="no published epoch photometry for this source (gaia_source.has_epoch_photometry is false)", confirmed=False))
        else:
            def arr(v):
                return np.array(str(v).strip("[]{}()").replace(",", " ").split(), float) if isinstance(v, str) else np.asarray(v, float)
            t_, m_, soe = arr(g.g_transit_time.iloc[0]), arr(g.g_transit_mag.iloc[0]), arr(g.g_transit_flux_over_error.iloc[0])
            e_ = 1.0857 / soe
            recs.append(dict(survey="Gaia DR3 epoch photometry (AIP mirror)", filter="G", status="ok", **lc_test(t_, m_, e_)))
            curves["Gaia G"] = (t_ + 55197.5, m_, e_)
    except Exception as e:
        recs.append(dict(survey="Gaia DR3 epoch photometry (AIP mirror)", status="unreachable: " + repr(e)[:100], confirmed=False))
    recs.append(dict(survey="ASAS-SN Sky Patrol light curve", status="not attempted: needs the pyasassn client, which is not installed; "
                     "the ASAS-SN variable catalogue (VizieR II/366) is checked instead", confirmed=False))
    return recs, curves


def confirm():
    S = load_sample()
    th = json.load(open(THR))
    R = pd.read_parquet(STARS)
    out = json.load(open(OUT))
    cands = [c for c in out["candidates"]["stars_above_T_cut"] if c["passes_frozen_rule"]]
    res, curves = [], {}
    order = [c["source_id"] for g in S["groups"] for c in g["C"] if c["half"] == "test"]
    Rt = R.set_index("sid")
    ctrl_ids = [i for i in order if i in Rt.index and Rt.loc[i, "status"] == "ok" and not bool(Rt.loc[i, "above_candidate_cut"])][:10]
    extra = []                                                  # D-E10-5: above the T cut but not candidates
    for c in out["candidates"]["stars_above_T_cut"]:
        if c["passes_frozen_rule"]:
            continue
        ga = pd.read_parquet(f"{D}/groups/{c['group_variable']}_gaia.parquet").set_index("source_id")
        extra.append(("star", c["gaia_dr3_source_id"], c["ra"], c["dec"], c["G"]))
        for nb in c["neighbours_within_12arcsec"]:
            q = ga.loc[int(nb["source_id"])]
            extra.append((f"neighbour of {c['gaia_dr3_source_id']}", nb["source_id"], float(q.ra), float(q.dec), nb["G"]))
    jobs = [(c["ra"], c["dec"], c["gaia_dr3_source_id"]) for c in cands] + \
           [(float(Rt.loc[i, "ra"]), float(Rt.loc[i, "dec"]), i) for i in ctrl_ids] + [(e[2], e[3], e[1]) for e in extra]
    with ThreadPoolExecutor(2) as tp:                           # the ZTF service takes ~3 min per position
        fetched = list(tp.map(lambda a: independent(*a), jobs))
    ext_res = [dict(role=e[0], gaia_dr3_source_id=str(e[1]), ra=e[2], dec=e[3], G=e[4], light_curves=f[0],
                    passes_A_E10_2=bool(any(r.get("confirmed") for r in f[0])),
                    catalogue_checks={n: (lambda t: "unreachable" if t is None else ("no entry within 10 arcsec" if len(t) == 0
                                                                                      else t.to_dict("records")))(
                                          vizier_cone(src, e[2], e[3], 10 / 3600, cols)) for n, (src, cols) in CATS.items()})
               for e, f in zip(extra, fetched[len(cands) + len(ctrl_ids):])]
    extra_curves = {str(e[1]): f[1] for e, f in zip(extra, fetched[len(cands) + len(ctrl_ids):])}
    for c, (recs, cv) in zip(cands, fetched[:len(cands)]):
        ok = [r for r in recs if r.get("confirmed")]
        res.append(dict(gaia_dr3_source_id=c["gaia_dr3_source_id"], ra=c["ra"], dec=c["dec"], G=c["G"], light_curves=recs,
                        independently_confirmed=bool(ok), confirmed_in=[f"{r['survey']} {r.get('filter', '')}".strip() for r in ok],
                        wording=("variable star not found in the catalogues we checked, variability confirmed in "
                                 + " and ".join(sorted({r["survey"] for r in ok})) + " data") if ok else
                                "candidate variable not found in the catalogues we checked; not independently confirmed"))
        curves[c["gaia_dr3_source_id"]] = cv
    ctrl = []
    for i, (recs, _) in zip(ctrl_ids, fetched[len(cands):]):
        ctrl.append(dict(gaia_dr3_source_id=str(i), G=float(Rt.loc[i, "G"]), light_curves=recs,
                         fires=bool(any(r.get("confirmed") for r in recs)),
                         n_curves_tested=sum(1 for r in recs if r.get("n_epochs", 0) >= 20)))
    out["independent_light_curves"] = dict(
        criterion="docs/protocols/E10.md A-E10-2: >= 20 good epochs AND (p95 - p5) amplitude > 5 x median error AND > 0.1 mag",
        candidates=res, n_candidates=len(res), n_confirmed=sum(r["independently_confirmed"] for r in res),
        control_comparison_stars=ctrl, n_control=len(ctrl), n_control_with_a_testable_curve=sum(c["n_curves_tested"] > 0 for c in ctrl),
        n_control_firing=sum(c["fires"] for c in ctrl),
        control_rule="first 10 testable test-half comparison stars in sample order that are not above the candidate cut",
        descriptive_above_cut_not_candidates=ext_res,
        descriptive_note="docs/protocols/E10.md D-E10-5: stars above the candidate T cut that fail the frozen rule, and "
                         "their bright neighbours, checked descriptively; never reported as candidates")
    json.dump({k: {n: [list(map(float, a)) for a in v] for n, v in cv.items()} for k, cv in extra_curves.items()},
              open(f"{D}/extra_curves.json", "w"))
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(json.dumps(out["independent_light_curves"], indent=1, default=float))
    conf = [r for r in res if r["independently_confirmed"]]
    if conf:
        confirmed_figure(conf, curves, th)


def confirmed_figure(conf, curves, th):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14})
    R = pd.read_parquet(STARS).set_index("sid")
    colors = {1: "#0072B2", 2: "#56B4E9", 3: "#009E73", 4: "#E69F00", 5: "#D55E00", 6: "#CC79A7"}
    bands = {1: "0.75-1.1", 2: "1.1-1.6", 3: "1.6-2.4", 4: "2.4-3.8", 5: "3.8-4.4", 6: "4.4-5.0"}
    fig, axs = plt.subplots(len(conf), 2, figsize=(14, 4.8 * len(conf)), squeeze=False, facecolor="white")
    for i, c in enumerate(conf):
        sid = int(c["gaia_dr3_source_id"])
        ph = pd.read_parquet(f"{D}/phot/{R.loc[sid, 'gid']}.parquet")
        rs = ratio_series(ph[ph.sid == sid], th["gate0_s_d"], th["floor_f_d"])
        ax = axs[i, 0]
        for d_ in sorted(rs.det.unique()):
            q = rs[rs.det == d_]
            ax.errorbar(q.mjd, 100 * (q.r - 1), 100 * q.re, fmt="o", ms=4, lw=0.8, color=colors[int(d_)], label=f"{bands[int(d_)]} µm")
        ax.axhline(0, color="0.5", lw=1)
        ax.set_title(f"SPHEREx: Gaia DR3 {sid} (G = {c['G']:.1f})", fontsize=13)
        ax.set_xlabel("Date (MJD)")
        ax.set_ylabel("Brightness vs. the star's\nown average (%)")
        ax.legend(fontsize=9, ncol=2, frameon=False, title="wavelength band")
        bx = axs[i, 1]
        for (name, (t_, m_, e_)), col in zip(curves[c["gaia_dr3_source_id"]].items(), ("#009E73", "#D55E00", "#0072B2", "#555555")):
            bx.errorbar(t_, m_ - np.median(m_), e_, fmt=".", ms=4, lw=0.5, color=col, label=name)
        bx.invert_yaxis()
        bx.set_title("Independent survey light curve", fontsize=13)
        bx.set_xlabel("Date (MJD)")
        bx.set_ylabel("Magnitude minus median\n(brighter is up)")
        bx.legend(fontsize=10, frameon=False)
        for a_ in (ax, bx):
            a_.spines[["top", "right"]].set_visible(False)
            a_.grid(axis="y", color="0.9")
    fig.tight_layout()
    fig.savefig(CONFFIG, dpi=140, facecolor="white")
    plt.close(fig)



def reread():
    """D-E10-4: re-read once every group whose first pass lost > 10% of its pictures to read errors."""
    import shutil
    S = load_sample()
    os.makedirs(f"{D}/phot_firstpass", exist_ok=True)
    todo = []
    for g in S["groups"]:
        lp = f"{D}/phot/{g['gid']}.log.json"
        if not os.path.exists(lp) or os.path.exists(f"{D}/phot_firstpass/{g['gid']}.log.json"):
            continue
        r = json.load(open(lp))
        if r["n_errors"] > 0.1 * r["n_frames"]:
            for suf in (".parquet", ".log.json", "_stamps.npz", "_frames.parquet"):
                if os.path.exists(f"{D}/phot/{g['gid']}{suf}"):
                    shutil.move(f"{D}/phot/{g['gid']}{suf}", f"{D}/phot_firstpass/{g['gid']}{suf}")
            todo.append(g)
    print("re-reading", len(todo), "groups", flush=True)
    ex = ProcessPoolExecutor(PROCS)
    for g in todo:
        try:
            r = phot_group(g, ex, f"{D}/phot")
        except Exception as e:
            print(g["gid"], "FAILED", repr(e)[:200], flush=True)
            continue
        if r and r["n_chunks_timed_out"]:
            ex.shutdown(wait=False, cancel_futures=True)
            ex = ProcessPoolExecutor(PROCS)
    ex.shutdown(wait=False, cancel_futures=True)
    print("reread done", time.strftime("%H:%M:%S"), flush=True)



def finalize():
    """Merge deviations (from docs/protocols/E10.md), the descriptive above-cut checks and limitations into e10.json."""
    import re
    P = "data/findings/e10.json"
    o = json.load(open(P))
    prot = open("docs/protocols/E10.md").read()
    dev = re.split(r"\n(?=- (?:D|A)-E10-\d)", prot.split("## Deviations", 1)[1])
    o["deviations"] = [re.sub(r"\s+", " ", d.strip().lstrip("- ")) for d in dev if re.match(r"\s*- ?(D|A)-E10-\d", "- " + d.strip().lstrip("- "))]
    o["deviations"].append("Freeze record: the protocol was committed as "
                           "docs/protocols/E10.md as of 12:34 PST (commit 022fff0). Hashes taken at 12:25:59 PST before any "
                           "sample picture was read: data/phase2/e10/freeze.json; byte copies data/phase2/e10/E10_protocol_as_frozen.md, "
                           "run_e10_as_frozen.py. Cap/stamp change (120 -> 96 pictures, 21 -> 17 px) was made after the throughput "
                           "test on 5 non-sample stars and before the freeze.")
    o["descriptive_above_cut_star"] = dict(
        gaia_dr3_source_id="4349942499431700992", note="test-half comparison star above the candidate T cut; excluded by the "
        "frozen neighbour criterion (Gaia DR3 4349942503734601728, G 13.30, within 12 arcsec); 0.22 deg from the RR Lyrae V0681 Oph",
        single_picture_check=json.load(open("data/phase2/e10/loo_4349942499431700992.json")),
        ztf_blend_check=json.load(open("data/phase2/e10/ztf_blend_check_4349942499431700992.json")),
        ztf_reading="ZTF passes the A-E10-2 amplitude test on all epochs, but all 39 epochs > 0.1 mag brighter than median are "
                    "exposures where ZTF did not separate the 12-arcsec neighbour (0 of 672 resolved epochs); resolved epochs "
                    "only: 0.045 mag (g), 0.038 mag (r), test not passed. Seasonal medians constant to 0.02 mag 2018-2025. ZTF "
                    "ends 2025-09-15, before SPHEREx window 2, so it cannot test the SPHEREx change. Not independently confirmed.",
        figure="data/phase2/e10/above_cut_excluded_4349942499431700992.png (descriptive, not a site figure)",
        reading="brightening of ~+8% between windows (MJD 60874-60884 and 61087-61111) seen in 5 of 6 detectors with "
                "per-window medians agreeing; removing the two most influential pictures leaves T = 103 (still above the "
                "candidate cut), so it is not carried by one or two pictures; which star of the 12-arcsec pair changed cannot "
                "be told from SPHEREx pixels")
    o["verdict_explanation"] = (
        "Frozen rule: PASS needs the 95% lower bound of LPV recall (testable) > 0.5 and the test-half false-positive interval "
        "to contain or lie below 1%. False positives pass (2/246). LPV recall is 16/24 = 0.67 with lower bound 0.467 < 0.5 and "
        "upper bound > 0.5, so the result is INCONCLUSIVE, not FAILED.")
    o["limitations"] = [
        "28 of 52 long-period variables (54%) are saturated in most pictures and could not be tested; the floor and thresholds "
        "were set on comparison stars of the same G, which are much fainter in the infrared.",
        "A window-level brightness factor averages over phase for RR Lyrae, Cepheids and eclipsing binaries; their recall "
        "reflects chance sampling of phase, not the stars' amplitudes.",
        "Comparison stars share pictures with their variable and each other (4 per group); not fully independent.",
        "Catalogue amplitude is Gaia G (visible) trimmed range; the infrared amplitude of LPVs is typically smaller.",
        "Catalogue services: CDS VizieR (Strasbourg) unreachable all day; VSX and other VizieR tables were read from the CfA "
        "mirror. ASAS-SN Sky Patrol light curves not fetched (client not installed). Gaia DR3 epoch photometry exists for "
        "none of the stars checked. ZTF light curves from IRSA where reachable (3 of the 10 control stars unreachable, "
        "4 south of ZTF coverage or without good epochs, 3 with curves, none firing).",
        "About 8 pictures per detector and window (cap 96) and 17-px stamps.",
    ]
    o["finalized_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(o, open(P, "w"), indent=1, default=float)
    print(len(o["deviations"]), [d[:40] for d in o["deviations"]])


if __name__ == "__main__":
    {"sample": sample, "sample_ext": lambda: sample(True), "throughput": throughput, "tp_pseudo": tp_pseudo, "phot": phot, "calibrate": calibrate, "analyze": analyze, "candidates": candidates, "confirm": confirm, "reread": reread, "finalize": finalize}[sys.argv[1]]()
