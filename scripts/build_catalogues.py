"""Downloadable catalogues for the site (data/products/): one CSV + README per result.

nep_variability: every source measured in the north-ecliptic-pole field, with the variability statistics after the
learned systematics correction (data/phase2/e26/stats_after, computed by scripts/run_e26.py) and whether it passes the
thresholds set on the calibration half of non-variable stars (same rule as run_e26.evaluate / run_e18b.above).
emission_lines: the hydrogen-line census (data/findings/e20.json).
Usage: .venv/bin/python scripts/build_catalogues.py
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = "data/products"


def nep_variability():
    import run_e26 as E
    B = E.B
    pop = B.populations()
    real, shuf = E._read("after", "real"), E._read("after", "shuf")
    real = real.merge(pop[["idx", "id", "kind", "half", "catalogued_var", "kim", "mq"]].rename(columns={"idx": "src"}), on="src")
    norms, thr, _ = B.calibrate(real, shuf, pop)
    real = B.above(B.zscores(real, norms), thr)
    src = pd.read_parquet("data/phase2/e18/sources.parquet")
    cols = ["idx", "ra", "dec", "G", "bp_rp", "W1mag", "W2mag", "vari_class", "kim2026_name", "milliquas_name"]
    t = real.merge(src[cols].rename(columns={"idx": "src"}), on="src", how="left")
    keep = ["id", "ra", "dec", "m_ab", "G", "bp_rp", "W1mag", "W2mag", "n", "n_bins", "chi2r", "chi2r_det", "s2", "ls",
            "amp", "P_best"] + [f"z_{s}" for s in B.STATS if f"z_{s}" in t] + [f"hit_{s}" for s in B.STATS if f"hit_{s}" in t] + \
           ["above", "half", "catalogued_var", "vari_class", "kim2026_name", "milliquas_name"]
    t = t[[c for c in keep if c in t]].rename(columns={"id": "gaia_or_allwise_id", "n": "n_pictures", "above": "variable_candidate",
                                                       "half": "role", "m_ab": "spherex_mean_m_ab"})
    t = t.sort_values("ra").reset_index(drop=True)
    for c in t.select_dtypes("float").columns:
        t[c] = t[c].round(6)
    t.to_csv(f"{OUT}/epoch_nep_variability.csv", index=False)
    return dict(n=len(t), n_candidates=int(t.variable_candidate.sum()), thresholds={k: round(v["Z"], 3) for k, v in thr.items()},
                columns=list(t.columns))


def emission_lines():
    """Hydrogen-line census: every candidate and control star measured (data/findings/e20.json, positions from the sample)."""
    d = json.load(open("data/findings/e20.json"))
    smp = json.load(open("data/phase2/e20/sample.json"))["sample"]
    pos = {}
    for c in smp:
        pos[c["source_id"]] = (c["ra"], c["dec"])
        for k in c.get("controls", []):
            pos[k["source_id"]] = (k["ra"], k["dec"])
    rows = []
    for s in d["stars"]:
        ra, dec = pos.get(s["source_id"], (np.nan, np.nan))
        r = dict(gaia_dr3_source_id=s["source_id"], ra=ra, dec=dec, role=s["role"], gaia_espels_class=s.get("cls"), G=s.get("G"),
                 n_pictures=s.get("n_pictures"), n_used=s.get("n_used"), testable=s.get("testable_main"), line_excess=s.get("excess_main"))
        for ln, L in s["lines"].items():
            r.update({f"{ln}_excess": L.get("E"), f"{ln}_excess_err": L.get("sE"), f"{ln}_Z": L.get("Z"), f"{ln}_detected": s["detected"].get(ln)})
        r.update(simbad_main_id=s.get("simbad_main_id"), simbad_type=s.get("simbad_otype"), simbad_n_refs=s.get("simbad_nbref"))
        rows.append(r)
    t = pd.DataFrame(rows).sort_values(["role", "ra"]).reset_index(drop=True)
    for c in t.select_dtypes("float").columns:
        t[c] = t[c].round(6)
    t.to_csv(f"{OUT}/epoch_emission_lines.csv", index=False)
    return dict(n=len(t), n_candidates=int((t.role == "candidate").sum()), n_with_excess=int(t.line_excess.fillna(False).astype(bool).sum()),
                n_missing_position=int(t.ra.isna().sum()), columns=list(t.columns))

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    info = dict(nep_variability=nep_variability(), emission_lines=emission_lines())
    json.dump(info, open(f"{OUT}/manifest.json", "w"), indent=1, default=str)
    print(json.dumps(info, indent=1, default=str)[:2000])
