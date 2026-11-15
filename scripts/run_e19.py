"""E19: does the VY CMa SPHEREx "shell" follow the detector or the sky? Replication of Karambelkar, Sharma & De 2026
(arXiv:2609.28612) on DR1 detector-1 pictures. Frozen choices: docs/protocols/E19.md.

  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e19.py select    # catalogue draw + picture lists (headers read)
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e19.py measure   # 181x181 cutouts + frozen ring statistic S
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e19.py blank     # Deviation 3 control: same pixels, no bright star
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e19.py report    # verdicts -> data/findings/e19.json, site/data/figs/e19_shield.png
Resumable: cutouts are cached in data/phase2/e19/cut/.
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shield import (SPOT, ring_stat, spike_stat, halo_subtract, pattern, corr, detector_xy, cutout, d1_index,  # noqa: E402
                    pictures_containing)  # noqa: E402

OUT = "data/phase2/e19"
SEED = 2026111519
VIZ = "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv"
NAMED = {"VY CMa": (110.743026, -25.767554, 2.82), "eps Oct": (335.007003, -80.439747, -0.08),
         "GY UMa": (161.902978, 65.616640, 2.60)}
J_RANGE = (-0.6, 3.3)
K26 = ["2025W46_1A_0770_1D1", "2025W43_2A_0490_1D1", "2025W43_1A_0318_2D1", "2026W02_1A_0645_4D1",
       "2025W43_2A_0319_1D1", "2025W44_1A_0590_2D1", "2025W37_1A_0297_1D1", "2025W52_2A_0587_1D1",
       "2025W50_1A_0678_2D1", "2026W24_1A_0015_1D1", "2026W29_1A_0585_1D1", "2026W16_1A_0558_1D1"]
WORKERS = 3


def tmass_j(ra, dec):
    url = f"{VIZ}?-source=II/246/out&-c={ra:.6f}{dec:+.6f}&-c.rs=10&-out=Jmag&-out.max=20"
    for i in range(4):
        try:
            t = requests.get(url, timeout=30).text
            rows = [ln.strip() for ln in t.splitlines() if ln and not ln.startswith("#")][3:]
            js = [float(r) for r in rows if r]
            return min(js) if js else None
        except Exception:
            time.sleep(2 * (i + 1))
    return None


def bsc_pool():
    url = f"{VIZ}?-source=V/50/catalog&-out=HR,Name,RAJ2000,DEJ2000,Vmag,B-V&-out.max=20000"
    rows = [ln.split("\t") for ln in requests.get(url, timeout=200).text.splitlines()
            if ln and not ln.startswith("#")][3:]
    out = []
    for r in rows:
        try:
            hr, name, ras, des, v, bv = r[0].strip(), r[1].strip(), r[2], r[3], float(r[4]), float(r[5])
        except (ValueError, IndexError):
            continue
        h, m, s = map(float, ras.split())
        sg = -1 if des.strip().startswith("-") else 1
        d, dm, ds = map(float, des.replace("-", " ").replace("+", " ").split())
        ra, dec = 15 * (h + m / 60 + s / 3600), sg * (d + dm / 60 + ds / 3600)
        if v < 7.5 and bv >= 1.3:
            out.append(dict(hr=int(hr), name=name or f"HR {hr}", ra=ra, dec=dec, V=v, BV=bv))
    pool = pd.DataFrame(out)
    for ra0, de0, _ in NAMED.values():
        sep = np.degrees(np.arccos(np.clip(np.sin(np.radians(pool.dec)) * np.sin(np.radians(de0)) + np.cos(
            np.radians(pool.dec)) * np.cos(np.radians(de0)) * np.cos(np.radians(pool.ra - ra0)), -1, 1)))
        pool = pool[sep > 3]
    order = np.random.default_rng(SEED).permutation(len(pool))
    return pool.iloc[order].reset_index(drop=True)


def confirm(rows, ra, dec, ex):
    """Header-WCS detector x,y for index rows -> DataFrame (key, obs_id, mjd, x, y, d)."""
    res = list(ex.map(lambda k: detector_xy(k, ra, dec), rows.key))
    df = rows[["key", "obs_id", "mjd"]].copy()
    df["x"], df["y"] = [r[0] for r in res], [r[1] for r in res]
    df["d"] = np.hypot(df.x - SPOT[0], df.y - SPOT[1])
    return df


def inside(df):
    return (df.x >= 91) & (df.x <= 2039 - 91) & (df.y >= 91) & (df.y <= 2039 - 91)


def select():
    os.makedirs(OUT, exist_ok=True)
    idx = d1_index()
    print(f"detector-1 index rows: {len(idx)}", flush=True)
    t0 = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    print("first header read:", t0, flush=True)
    sel = dict(first_header_read=t0, named={}, c_else=[], c_spot=[], k26=[])
    rng = np.random.default_rng(SEED)
    with ThreadPoolExecutor(WORKERS) as ex:
        for name, (ra, dec, J) in NAMED.items():
            c = pictures_containing(idx, ra, dec)
            near = c[c.d_approx < 200]
            far = c[c.d_approx >= 200]
            same_row = far[(abs(far.y_approx - SPOT[1]) < 70) & (abs(far.x_approx - SPOT[0]) > 140)]
            rest = far.drop(same_row.index)
            rest = rest.iloc[rng.permutation(len(rest))[:20]]
            conf = confirm(pd.concat([near, same_row, rest]), ra, dec, ex)
            conf = conf[inside(conf)]
            sel["named"][name] = dict(ra=ra, dec=dec, J=J, n_d1_pictures=len(c), pictures=conf.to_dict("records"))
            print(f"{name}: {len(c)} D1 pictures; spot {int((conf.d < 60).sum())}, near "
                  f"{int(((conf.d >= 60) & (conf.d <= 150)).sum())}, measured total {len(conf)}", flush=True)
        pool = bsc_pool()
        print(f"BSC pool: {len(pool)} stars", flush=True)
        n_else = 0
        for r in pool.itertuples():
            if len(sel["c_spot"]) >= 15 and n_else >= 8:
                break
            c = pictures_containing(idx, r.ra, r.dec)
            if not len(c):
                continue
            want_else = n_else < 8
            spot_cand = c[c.d_approx < 200]
            if not want_else and (not len(spot_cand) or len(sel["c_spot"]) >= 15):
                continue
            J = tmass_j(r.ra, r.dec)
            if J is None or not (J_RANGE[0] <= J <= J_RANGE[1]):
                continue
            rec = dict(hr=r.hr, name=r.name, ra=r.ra, dec=r.dec, V=r.V, BV=r.BV, J=J)
            if len(spot_cand) and len(sel["c_spot"]) < 15:
                conf = confirm(spot_cand, r.ra, r.dec, ex)
                conf = conf[inside(conf) & (conf.d < 60)]
                if len(conf):
                    sel["c_spot"].append(dict(rec, pictures=conf.to_dict("records")))
                    print(f"C-spot {r.name} J={J:.2f}: {len(conf)} spot pictures", flush=True)
            if want_else:
                far = c[c.d_approx >= 250]
                far = far.iloc[rng.permutation(len(far))[:8]]
                conf = confirm(far, r.ra, r.dec, ex)
                conf = conf[inside(conf) & (conf.d > 150)].iloc[:6]
                if len(conf):
                    sel["c_else"].append(dict(rec, pictures=conf.to_dict("records")))
                    n_else += 1
                    print(f"C-else {r.name} J={J:.2f}: {len(conf)} pictures", flush=True)
        for oid in K26:
            m = idx[(idx.obs_id == oid[:-2])]
            sel["k26"].append(dict(id=oid, in_index=bool(len(m)), key=m.key.iloc[0] if len(m) else None,
                                   mjd=float(m.mjd.iloc[0]) if len(m) else None))
        print("K26 ids in DR1 index:", sum(k["in_index"] for k in sel["k26"]), "/", len(K26), flush=True)
    json.dump(sel, open(f"{OUT}/selection.json", "w"), indent=1, default=float)


def measure():
    sel = json.load(open(f"{OUT}/selection.json"))
    jobs = []
    for name, s in sel["named"].items():
        for p in s["pictures"]:
            jobs.append(dict(star=name, set="named", ra=s["ra"], dec=s["dec"], J=s["J"], **p))
    for setname in ("c_else", "c_spot"):
        for s in sel[setname]:
            for p in s["pictures"]:
                jobs.append(dict(star=s["name"], set=setname, ra=s["ra"], dec=s["dec"], J=s["J"], **p))
    for k in sel["k26"]:
        if k["in_index"]:
            jobs.append(dict(star=f"K26 {k['id']}", set="k26", ra=None, dec=None, J=None, key=k["key"],
                             obs_id=k["id"][:-2], mjd=k["mjd"], x=float(SPOT[0]), y=float(SPOT[1]),
                             d=0.0))
    os.makedirs(f"{OUT}/cut", exist_ok=True)

    def one(j):
        path = f"{OUT}/cut/{j['obs_id']}_D1_{j['star'].replace(' ', '_')}.npy"
        if os.path.exists(path):
            img = np.load(path)
        else:
            img = cutout(j["key"], j["x"], j["y"])
            np.save(path + ".tmp.npy", img)
            os.replace(path + ".tmp.npy", path)
        return dict(j, cutout=path, **ring_stat(img, j["x"] - round(j["x"]), j["y"] - round(j["y"])))

    with ThreadPoolExecutor(WORKERS) as ex:
        rows = []
        for i, r in enumerate(ex.map(one, jobs)):
            rows.append(r)
            if i % 20 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    res = pd.DataFrame(rows)
    res.to_parquet(f"{OUT}/pictures.parquet", index=False)
    print(f"measured {len(res)} pictures", flush=True)


def blank():
    """Deviation 3: the C-else pictures cut at the fixed spot (no bright star there); appended to pictures.parquet."""
    res = pd.read_parquet(f"{OUT}/pictures.parquet")
    res = res[res.set != "blank"]
    ce = res[res.set == "c_else"]

    def one(r):
        path = f"{OUT}/cut/{r.obs_id}_D1_blankspot.npy"
        if not os.path.exists(path):
            img = cutout(r.key, *SPOT)
            np.save(path + ".tmp.npy", img)
            os.replace(path + ".tmp.npy", path)
        img = np.load(path)
        return dict(star=f"sky at the spot ({r.star} elsewhere)", set="blank", ra=None, dec=None, J=None, key=r.key,
                    obs_id=r.obs_id, mjd=r.mjd, x=SPOT[0], y=SPOT[1], d=0.0, cutout=path, **ring_stat(img))
    with ThreadPoolExecutor(WORKERS) as ex:
        b = pd.DataFrame(list(ex.map(one, ce.itertuples())))
    pd.concat([res, b], ignore_index=True).to_parquet(f"{OUT}/pictures.parquet", index=False)
    print(f"blank-spot cutouts: {len(b)}", flush=True)


def cls(d):
    return "spot" if d < 60 else ("near" if d <= 150 else "elsewhere")


def evaluate(res, col):
    """Threshold on C-else and the frozen criteria (a)-(c) for statistic `col` (S frozen; Z post hoc)."""
    ce = res[res.set == "c_else"][col]
    med, sig = float(ce.median()), float(1.4826 * np.median(np.abs(ce - ce.median())))
    T = med + 5 * sig
    present = res[col] > T
    per = {}
    for name in NAMED:
        r = res[(res.set == "named") & (res.star == name)]
        p = present[r.index]
        sp, nr, el = r["class"] == "spot", r["class"] == "near", r["class"] == "elsewhere"
        per[name] = dict(n_spot=int(sp.sum()), n_spot_ring=int(p[sp].sum()), n_near=int(nr.sum()), n_near_ring=int(p[nr].sum()),
                         n_elsewhere=int(el.sum()), n_elsewhere_ring=int(p[el].sum()),
                         n_same_row=int((el & r.same_row).sum()), n_same_row_ring=int(p[el & r.same_row].sum()),
                         spot=[dict(obs_id=o, value=float(v)) for o, v in zip(r[sp].obs_id, r[sp][col])],
                         elsewhere_median=float(r[el][col].median()),
                         elsewhere_robust_sigma=float(1.4826 * np.median(np.abs(r[el][col] - r[el][col].median()))))
    cs, k = res[res.set == "c_spot"], res[res.set == "k26"]
    v, e, g = per["VY CMa"], per["eps Oct"], per["GY UMa"]
    a_ = v["n_spot_ring"] >= 1
    b_ = e["n_spot_ring"] >= 1 and g["n_spot_ring"] >= 1
    c_ = v["n_elsewhere"] >= 5 and v["n_elsewhere_ring"] <= 0.1 * v["n_elsewhere"] and v["n_same_row_ring"] == 0
    astro = v["n_elsewhere_ring"] >= 0.5 * v["n_elsewhere"] and not e["n_spot_ring"] and not g["n_spot_ring"]
    if v["n_spot"] == 0 or not a_:
        verdict = "FAILED (ring not seen at the spot)"
    elif a_ and b_ and c_:
        verdict = "REPLICATED: the feature follows the detector position (artefact), as Karambelkar et al. 2026 found"
    elif astro:
        verdict = "ASTROPHYSICAL-favoured"
    else:
        verdict = "INCONCLUSIVE"
    return T, present, dict(
        threshold=dict(T=T, c_else_median=med, c_else_robust_sigma=sig, n_c_else=int(len(ce)), c_else_exceed=int((ce > T).sum())),
        named=per,
        c_spot=dict(n_stars=int(cs.star.nunique()), n_pictures=int(len(cs)), n_ring=int(present[cs.index].sum()),
                    n_stars_ring=int(cs[present[cs.index]].star.nunique())),
        k26_listed=dict(n_in_index=int(len(k)), n_ring=int(present[k.index].sum()),
                        ring=[o for o, p in zip(k.obs_id, present[k.index]) if p]),
        criteria=dict(a_vycma_spot_ring=bool(a_), b_epsoct_and_gyuma_spot_ring=bool(b_), c_vycma_elsewhere_clean=bool(c_)),
        verdict=verdict)


def pattern_test(res):
    """Deviation 2: do different stars at the spot share the same detector-frame spike pattern?"""
    from scipy.stats import mannwhitneyu
    P = {i: pattern(np.load(r.cutout), r.x - round(r.x), r.y - round(r.y)) for i, r in res.iterrows()}
    k26 = {i: r for i, r in res[res.set == "k26"].iterrows()}
    named_k26 = {}
    for s_, oid in [("VY CMa", K26[0]), ("eps Oct", K26[1]), ("GY UMa", K26[2])]:
        named_k26[s_] = res[(res.set == "named") & (res.star == s_) & (res.obs_id == oid[:-2])].index[0]
    A = list(res[(res.set == "named") & (res.d < 15)].index) + [i for i, r in k26.items() if r.obs_id not in
                                                                [res.obs_id[j] for j in named_k26.values()]]
    star_of = {i: (res.star[i] if res.set[i] != "k26" else res.obs_id[i]) for i in res.index}
    per = {}
    for s_, a in named_k26.items():
        others = [named_k26[t] for t in named_k26 if t != s_]
        c_spot = float(np.mean([corr(P[a], P[o]) for o in others]))
        el = res[(res.set == "named") & (res.star == s_) & (res.d > 150)]
        c_el = {res.obs_id[e]: float(np.mean([corr(P[e], P[o]) for o in others])) for e in el.index}
        v = np.array(list(c_el.values()))
        per[s_] = dict(spot_picture=res.obs_id[a], c_spot=c_spot, n_elsewhere=len(v), c_else_max=float(v.max()),
                       c_else_median=float(np.median(v)), c_else_robust_sigma=float(1.4826 * np.median(np.abs(v - np.median(v)))),
                       rank=int((v >= c_spot).sum()) + 1, follows_detector=bool(c_spot > v.max()), c_else=c_el)
    aa = [corr(P[i], P[j]) for n, i in enumerate(A) for j in A[n + 1:] if star_of[i] != star_of[j]]
    ce = list(res[res.set == "c_else"].index)
    ab = [corr(P[i], P[j]) for i in A for j in ce]
    bb = [corr(P[i], P[j]) for n, i in enumerate(ce) for j in ce[n + 1:] if star_of[i] != star_of[j]]
    res["similarity"] = [float(np.mean([corr(P[i], P[a]) for a in named_k26.values()
                                     if star_of[a] != star_of[i] and res.obs_id[a] != res.obs_id[i]]))
                         for i in res.index]
    bl = res[res.set == "blank"].similarity
    blank_summary = dict(n=int(len(bl)), median=float(bl.median()) if len(bl) else None,
                         max=float(bl.max()) if len(bl) else None)
    ok = all(v["follows_detector"] for v in per.values())
    return dict(note="NOT pre-registered; protocol Deviation 2", named=per, n_spot_set=len(A),
                blank_spot_control=blank_summary,
                spot_spot=dict(n=len(aa), median=float(np.median(aa)), p16=float(np.percentile(aa, 16)), p84=float(np.percentile(aa, 84))),
                spot_vs_c_else=dict(n=len(ab), median=float(np.median(ab)), p16=float(np.percentile(ab, 16)), p84=float(np.percentile(ab, 84))),
                c_else_pairs=dict(n=len(bb), median=float(np.median(bb)), p16=float(np.percentile(bb, 16)), p84=float(np.percentile(bb, 84))),
                mannwhitney_p_spot_spot_gt_spot_else=float(mannwhitneyu(aa, ab, alternative="greater").pvalue),
                verdict=("REPLICATED (post hoc): each named star's spot picture shares the detector-frame pattern of the "
                         "other stars at the spot better than any of its own pictures elsewhere" if ok else "NOT REPLICATED (post hoc)"))


def report():
    res = pd.read_parquet(f"{OUT}/pictures.parquet")
    res["class"] = [cls(d) for d in res.d]
    res["same_row"] = (abs(res.y - SPOT[1]) < 60) & (abs(res.x - SPOT[0]) > 150)
    z = [spike_stat(np.load(r.cutout), r.x - round(r.x), r.y - round(r.y)) for r in res.itertuples()]
    res["Z"], res["Z_err"], res["sigma_R"] = [q["Z"] for q in z], [q["Z_err"] for q in z], [q["sigma_R"] for q in z]
    T_S, pS, frozen = evaluate(res, "S")
    T_Z, pZ, post = evaluate(res, "Z")
    res["ring_present_frozen_S"], res["ring_present_posthoc_Z"] = pS, pZ
    out = dict(experiment="E19", protocol="docs/protocols/E19.md",
               replicates="Karambelkar, Sharma & De 2026, arXiv:2609.28612 (the VY CMa SPHEREx shell is likely an artifact)",
               withdrawn_claim="Banerjee et al. 2026, arXiv:2609.18895 (withdrawn 22 Sep 2026)",
               spot_xy=SPOT, classes="spot d<60 px; near 60-150; elsewhere >150; same_row |y-1752|<60 & |x-1648|>150",
               frozen=dict(statistic="S = (mean of 16 sector medians r 30-50 px - median r 60-80 px) / (1.4826 MAD r 60-80 px)",
                           **frozen),
               post_hoc=dict(note="NOT pre-registered; defined after seeing the frozen result (protocol Deviation 1)",
                             statistic="Z = azimuthal structure (P90-P10 of 120 radial-median azimuth bins, r 20-70 px) of the "
                                       "halo-subtracted image, in units of its expected noise", **post),
               verdict_frozen=frozen["verdict"], verdict_post_hoc_Z=post["verdict"])
    out["post_hoc_pattern"] = pattern_test(res)
    out["verdict_post_hoc_pattern"] = out["post_hoc_pattern"]["verdict"]
    out["pictures"] = res.drop(columns=["key"]).assign(
        url=[f"https://nasa-irsa-spherex.s3.amazonaws.com/{k}" for k in res.key]).to_dict("records")
    os.makedirs("data/findings", exist_ok=True)
    json.dump(out, open("data/findings/e19.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: out[k] for k in ("frozen", "post_hoc", "post_hoc_pattern")}, indent=1))
    figure(res)


def figure(res, T=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 15})
    nm = res[res.set == "named"]
    vy_el = nm[(nm.star == "VY CMa") & (nm.d > 150)]
    vy_el = vy_el.iloc[(vy_el.y - SPOT[1]).abs().argsort()]                # closest wavelength to the spot
    cs, ce = res[res.set == "c_spot"], res[res.set == "c_else"]
    panels = [(nm[nm.obs_id == oid[:-2]].iloc[0], f"{s_} at the spot") for s_, oid in zip(NAMED, K26[:3])]
    near = cs.loc[cs.d.idxmin()]
    panels += [(vy_el.iloc[0], "VY CMa elsewhere (same star)"),
               (near, f"Another bright star {near.d:.0f} px from the spot"),
               (ce.iloc[(ce.similarity - ce.similarity.median()).abs().argsort().iloc[0]], "Another bright star elsewhere")]
    fig = plt.figure(figsize=(22, 11), facecolor="white")
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.5], wspace=0.08, hspace=0.3)
    gs.update(right=0.98)
    th = np.linspace(0, 2 * np.pi, 200)
    for i, (r, lab) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.set_xticks([]), ax.set_yticks([])
        R = np.load(r.cutout)
        rr = np.hypot(*np.mgrid[:R.shape[0], :R.shape[1]] - (R.shape[0] - 1) / 2)
        D = (R - np.nanmedian(R[rr > 80])) / r.sigma_R
        ax.imshow(np.arcsinh(D / 2), origin="lower", cmap="gray", vmin=-1.5, vmax=np.arcsinh(40), interpolation="nearest")
        c = (R.shape[0] - 1) / 2
        ax.plot(c + 40 * np.cos(th), c + 40 * np.sin(th), ":", color="tab:orange", lw=1.3)
        ax.set_title(f"{lab}\n{r.star}  ({r.obs_id})", fontsize=14)
        ax.text(0.02, 0.02, f"detector x,y = {r.x:.0f}, {r.y:.0f}\nmatch to 'shell' pattern = {r.similarity:+.2f}",
                color="yellow", fontsize=13, transform=ax.transAxes, va="bottom", bbox=dict(facecolor="black", alpha=0.6, lw=0))
    ax = fig.add_subplot(gs[:, 3])
    ax.set_position(np.array(ax.get_position().bounds) + [0.045, 0, -0.045, 0])
    for s_, lab, col, mk in [("c_else", "Other bright stars", "#999999", "o"),
                             ("c_spot", "Other bright stars, 36-60 px from spot", "#7a5195", "s"),
                             ("k26", "Pictures listed by Karambelkar+", "#2ca02c", "^"),
                             ("blank", "Same spot, no bright star there", "#17becf", "P")]:
        d = res[res.set == s_]
        ax.scatter(d.d + 1, d.similarity, s=55, c=col, marker=mk, label=lab, alpha=0.85, edgecolor="none")
    for s_, col in zip(NAMED, ["#d62728", "#1f77b4", "#ff7f0e"]):
        d = nm[nm.star == s_]
        ax.scatter(d.d + 1, d.similarity, s=70, marker="D", color=col, label=s_, alpha=0.9, edgecolor="k", lw=0.5)
    ax.axvspan(0, 16, color="tab:orange", alpha=0.18)
    ax.axhline(0, color="k", lw=0.6)
    ax.text(1.1, 0.97, "the spot", transform=ax.get_xaxis_transform(), va="top", fontsize=13, color="tab:orange")
    ax.set_xscale("log")
    ax.set_xlabel("Star's distance from detector spot (1648, 1752) + 1 [pixels]")
    ax.set_ylabel("Match to the 'shell' pattern (correlation)")
    ax.legend(fontsize=12, loc="lower right", frameon=True, framealpha=0.9)
    ax.set_title("The 'shell' sits on the detector, not on the sky", fontsize=16)
    fig.suptitle("Same detector spot → same 'shell'. Different spot → no shell.", fontsize=22, y=1.0)
    fig.text(0.5, 0.945, "SPHEREx DR1 Band 1 (detector 1), ~0.78 µm. A replication of Karambelkar, Sharma & De 2026 "
             "(arXiv:2609.28612), the check that led to the withdrawal of a claimed shell around VY CMa.",
             ha="center", fontsize=14)
    fig.text(0.01, 0.0, "Images: SPHEREx pixels, sky-subtracted, asinh stretch; dotted circle = 40 px ≈ 4′. Match = pixel "
             "correlation of the glow-subtracted 20–70 px pattern with the three top-row pictures (own star excluded).\n"
             "Post-hoc test (protocol E19, Deviations 2–3); the pre-registered smooth-ring statistic did not detect "
             "the spiky shell (FAILED; see data/findings/e19.json).", fontsize=13)
    os.makedirs("site/data/figs", exist_ok=True)
    fig.savefig("site/data/figs/e19_shield.png", dpi=100, bbox_inches="tight", facecolor="white")
    print("wrote site/data/figs/e19_shield.png")


if __name__ == "__main__":
    {"select": select, "measure": measure, "blank": blank, "report": report}[sys.argv[1]]()
