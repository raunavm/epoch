"""Science-page figures, all regenerated from repository products with one command:
.venv/bin/python scripts/make_figures.py   ->  site/data/figs/fig_*.png
Colours: Okabe–Ito (colour-blind safe).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = "site/data"
OUT = f"{S}/figs"
OI = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def fig_sizes():
    objs = json.load(open(f"{S}/objects.json"))
    v = [o for o in objs if o.get("D_km") and o.get("D_truth_km")]
    if not v:
        return
    kinds = sorted(set(o["kind"] for o in v))
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for i, k in enumerate(kinds):
        s = [o for o in v if o["kind"] == k]
        ax.errorbar([o["D_truth_km"] for o in s], [o["D_km"] for o in s], yerr=[o["D_err_km"] for o in s],
                    fmt="o", ms=5, color=OI[i % 8], label=k, capsize=2, lw=1)
    lo, hi = min(o["D_truth_km"] for o in v) * 0.6, max(o["D_truth_km"] for o in v) * 1.6
    x = np.geomspace(lo, hi, 10)
    ax.plot(x, x, "k-", lw=1)
    ax.fill_between(x, x / 1.2, x * 1.2, color="0.85", zorder=0, label="±20%")
    ax.set(xscale="log", yscale="log", xlabel="Independent diameter (NEOWISE / AKARI / IRAS, km)",
           ylabel="SPHEREx thermal diameter (this work, km)", title="Sizes: SPHEREx vs independent (DERIVED)")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_sizes.png", dpi=150)
    plt.close(fig)


def fig_gaia():
    objs = json.load(open(f"{S}/objects.json"))
    rows = []
    for o in objs:
        d = json.load(open(f"{S}/objects/{o['id']}.json"))
        g = d["spectrum"].get("gaia")
        if g and d["spectrum"].get("gaia_check"):
            rows.append(d)
    if not rows:
        return
    rows = sorted(rows, key=lambda d: -d["spectrum"]["gaia_check"]["n"])[:6]
    fig, axs = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
    for ax, d in zip(axs.ravel(), rows):
        sp = d["spectrum"]
        g = sp["gaia"]
        k = sp["gaia_check"]["k"]
        ax.errorbar(g["wl_um"], np.array(g["refl"]) * k, yerr=np.array(g["err"]) * k, fmt="s", ms=3, color=OI[1], label="Gaia DR3 (scaled)")
        wl, r, e = map(np.array, (sp["wl_um"], sp["refl"], sp["refl_err"]))
        m = wl < 1.1
        ax.errorbar(wl[m], r[m], yerr=e[m], fmt="o", ms=3, color=OI[0], label="SPHEREx (this work)")
        c = sp["gaia_check"]
        ax.set_title(f"{d['name'][:22]}\nχ²/dof {c['chi2'] / max(c['dof'], 1):.2f} ({c['dof']} dof)", fontsize=9)
        ax.set_xlim(0.35, 1.12)
    axs[0, 0].legend(fontsize=8, frameon=False)
    for ax in axs[1]:
        ax.set_xlabel("Wavelength (µm)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Relative reflectance")
    fig.suptitle("Colours vs ESA Gaia, compared within each visit (DATA vs DATA)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_gaia.png", dpi=150)
    plt.close(fig)


def fig_reveal():
    p = f"{S}/reveal.json"
    if not os.path.exists(p):
        return
    r = json.load(open(p))
    st = r["stack_pngs"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    axs[0].plot([s["n"] for s in st], [s["snr"] for s in st], "o-", color=OI[0], label="along the orbit")
    n = np.array([s["n"] for s in st])
    axs[0].plot(n, st[-1]["snr"] * np.sqrt(n / n.max()), "--", color="0.5", label="√N expectation")
    axs[0].axhline(5, color=OI[5], lw=1)
    axs[0].text(n.max() * 0.02, 5.4, "5σ detection", color=OI[5], fontsize=9)
    axs[0].set(xscale="log", xlabel="Pictures combined", ylabel="Signal / noise", title=f"Asteroid {r['object']}")
    axs[0].legend(frameon=False, fontsize=9)
    sn = [f["snr"] for f in r["frames"]]
    axs[1].hist(sn, bins=30, color=OI[4], label="single pictures")
    axs[1].axvline(5, color=OI[5], lw=1)
    for v in r.get("null_snrs", []):
        axs[1].axvline(v, color="0.3", lw=0.8, ls=":")
    axs[1].set(xlabel="Signal / noise", ylabel="Pictures", title="Single pictures (bars), off-orbit nulls (dotted)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_reveal.png", dpi=150)
    plt.close(fig)


def fig_change():
    p = "data/findings/f1.json"
    if not os.path.exists(p):
        return
    f = json.load(open(p))
    pf = f["per_field"]
    short = {"ecliptic_control": "Control field (ecliptic)"}
    names = [short.get(x["field"], x["name"].split("(")[0].strip()[:24]) for x in pf]
    y = np.arange(len(pf))
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.barh(y + 0.2, [x["n_naive_all"] for x in pf], 0.4, color=OI[1], label="plain blink")
    ax.barh(y - 0.2, [x["n_corrected"] for x in pf], 0.4, color=OI[0], label="colour-corrected")
    ax.set_yticks(y, names)
    ax.set_xlabel("Apparent changes (groups of ≥2 pixels beyond 5σ)")
    ax.set_title("Which change is real?  (DERIVED)", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_change.png", dpi=150)
    plt.close(fig)


def fig_injections():
    p = f"{S}/injections.json"
    if not os.path.exists(p):
        return
    j = json.load(open(p))
    c = j["cells"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, pv in enumerate(j["grid"]["pV"]):
        s = sorted([x for x in c if x["pV"] == pv], key=lambda x: x["V"])
        ax.plot([x["V"] for x in s], [x["recovered_frac"] for x in s], "o", color=OI[i], label=f"albedo {pv}", alpha=0.8)
    ax.axvline(j["V50_stack"], color="0.3", ls="--")
    ax.set(xlabel="Visual magnitude of planted object (MODEL brightness)", ylabel="Fraction recovered (S/N ≥ 5)",
           title=f"Planted objects in real pictures ({j['n_frames']} pictures)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_injections.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_sizes, fig_gaia, fig_reveal, fig_change, fig_injections):
        f()
    print(sorted(os.listdir(OUT)))
