""""Results at a glance" table -> site/data/results.json, from docs/CLAIMS.md (verdicts) and data/findings/*.json (numbers).

The page (site/results.html) reads this file plus site/data/targets/<id>.json and site/data/findings.json. Here:
which claims and which numbers belong to each row (the mapping below), each claim's verdict exactly as the ledger
gives it, numbers that are on no card yet (read from the findings files), and the Science / What-failed anchors
(resolved from the headings of site/data/science.md and failed.md with the same rule as site/js/markdown.js).
Usage: .venv/bin/python scripts/build_results.py
"""
import json, math, re

OUT = "site/data/results.json"
VERDICTS = ["NOT CONFIRMED", "SUPPORTED", "FAILED"]  # the three shown; checked in this order
# Words in a ledger row that mark a check made afterwards, a secondary check, or a descriptive (not pass/fail) result.
LATE = ["post hoc", "post-hoc", "afterwards", "after the result", "secondary", "not blind", "not pre-registered", "descriptive"]


def load(p):
    with open(p) as f:
        return json.load(f)


def claims():
    out = {}
    for line in open("docs/CLAIMS.md"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6 or not re.fullmatch(r"[A-Z]+\d+[a-z]?", cells[0]):
            continue
        status = cells[-1]
        bold = re.search(r"\*\*(.+?)\*\*", status)
        short = bold.group(1) if bold else re.split(r" \(|;", status)[0]
        verdict = next((v for v in VERDICTS if short.upper().startswith(v)), None)
        out[cells[0]] = dict(id=cells[0], verdict=verdict, status=short.strip(), late=any(w in line.lower() for w in LATE))
    return out


def anchors(md):
    """Heading text -> id, as site/js/markdown.js assigns them (explicit {#id} first, then slugs)."""
    src = re.sub(r"^```.*?^```", "", open(md).read(), flags=re.M | re.S)  # no headings inside code blocks
    taken = set(re.findall(r"\{#([\w-]+)\}\s*$", src, re.M))
    out = {}
    for m in re.finditer(r"^(#{1,6}) (.+?)\s*$", src, re.M):
        text = m.group(2)
        idm = re.search(r"\s*\{#([\w-]+)\}$", text)
        if idm:
            out[text[:idm.start()]] = idm.group(1)
            continue
        b = re.sub(r"^-|-$", "", re.sub(r"[^\wÀ-ɏ]+", "-", re.sub(r"<[^>]+>", "", text.lower()), flags=re.A)) or "section"
        k, n = b, 1
        while k in taken:
            n += 1
            k = f"{b}-{n}"
        taken.add(k)
        out[text] = k
    return out


def find(heads, prefix):
    hits = [v for k, v in heads.items() if k.startswith(prefix)]
    assert len(hits) == 1, f"heading {prefix!r}: {hits}"
    return hits[0]


def target_numbers(tid, labels):
    nums = load(f"site/data/targets/{tid}.json")["numbers"]
    idx = []
    for lab in labels:
        hit = [i for i, n in enumerate(nums) if n["label"].startswith(lab)]
        assert len(hit) == 1, f"{tid}: number {lab!r}: {hit}"
        idx += hit
    return idx


def extras():
    """Numbers for rows that have no card, read from the findings files."""
    e9 = load("data/findings/e9.json")
    rule = float(re.search(r"\|a_w\| > (\d+(?:\.\d+)?) sigma", e9["verdict_rule"]).group(1))
    zs = [w["z"] for w in e9["targets"]["rdra"]["a_w"]]
    zmax = max(zs, key=abs)
    e1 = load("data/findings/e1.json")
    censored = all(m.get("snr50_censored") == ">max" for m in e1["per_method"].values())
    e3 = load("data/findings/e3.json")["all"]
    cal = load("data/findings/e15.json")["calibration"]
    return {
        "rdra": [dict(key="rdra_z", value=round(zmax, 1), rule=rule, source="data/findings/e9.json")],
        "e1": [dict(key="e1_gain", text=e1["gain_criterion"]["statement"], source="data/findings/e1.json")]
        + ([dict(key="e1_half", value=max(e1["levels_snr_frame"]), source="data/findings/e1.json")] if censored else []),
        "e3": [dict(key=f"e3_{m}", hits=e3[m]["hits"], n=e3["n_pairs"], lo=round(100 * e3[m]["wilson95"][0], 1),
                    hi=round(100 * e3[m]["wilson95"][1], 1), source="data/findings/e3.json") for m in ("M4", "M2")],
        "e15": [dict(key="e15_loo", hits=round(cal["loo_accuracy"] * cal["n_decisive"]), n=cal["n_decisive"],
                     source="data/findings/e15.json")],
    }


def main():
    C, sci, fail, ex = claims(), anchors("site/data/science.md"), anchors("site/data/failed.md"), extras()
    failed = lambda ns: [dict(n=n, href="failed.html#" + find(fail, f"{n}. ")) for n in ns]
    science = lambda p: "science.html#" + find(sci, p)
    T = lambda tid, cl, labels, sp, fl: dict(id=tid, group="target", target=tid, claims=[C[c] for c in cl],
                                             numbers=target_numbers(tid, labels), extra=ex.get(tid, []),
                                             card=f"timemachine.html?target={tid}", science=science(sp), failed=failed(fl))
    O = lambda rid, cl, sp, fl, fig, **kw: dict(id=rid, claims=[C[c] for c in cl], science=science(sp), failed=failed(fl),
                                                figure=fig, extra=ex.get(rid, []), **kw)
    rows = [
        T("barnard", ["C17", "C15"], ["Motion north–south (SPHEREx)", "Motion north–south (Gaia DR3)", "Pre-set check on faint neighbours"],
          "Barnard's Star: the motion", [13, 17]),
        T("epsindib", ["C10", "C16", "C14"], ["Motion east–west (SPHEREx)", "Motion east–west (Gaia DR3)", "Parallax from SPHEREx positions", "Distance"],
          "A distance from one year", []),
        T("wise0855", ["C9", "C13"], ["Signal, Nov–Dec 2025", "Brightness at 4.4–5", "Spectrum 3.8–5"], "WISE 0855", [10]),
        T("rwaur", ["C18"], ["Largest single-band change", "Change at 4.12"], "RW Aurigae and R Draconis", [16]),
        T("rdra", ["C19"], ["Colour bands usable"], "RW Aurigae and R Draconis", [16]),
        O("f1", ["C6a", "C6d"], "Which change is real", [2], "data/figs/fig_e2.png", group="finding", finding="f1", card="start.html"),
        O("f2", ["C5b", "C22"], "Stacking along an orbit", [6, 18], "data/figs/fig_reveal.png", group="finding", finding="f2", card="start.html"),
        O("e1", ["C6b"], "Faint changes in single pictures", [11], "data/figs/fig_e1.png", group="other"),
        O("e3", ["C6c", "C6e"], "Faint changes in single pictures", [14], None, group="other"),
        O("e15", ["C20"], "Using 102 colours", [], "data/figs/e15_methane.png", group="other"),
        O("e10", ["C27"], "Recovering known variable stars", [], "data/figs/e10_recall.png", group="other"),
        O("e20", ["C26"], "Hydrogen lines in emission-line stars", [], "data/figs/e20_lines.png", group="other"),
        O("e17", ["C25"], "Searching blind for moving objects", [], "data/figs/e17_recovery.png", group="other"),
        O("e19", ["C23", "C24"], "A check that would have stopped", [19], "data/figs/e19_shield.png", group="other"),
    ]
    for r in rows:
        for c in r["claims"]:
            assert c["verdict"], f"{c['id']}: status {c['status']!r} is not one of {VERDICTS}"
    out = dict(generated_with="scripts/build_results.py", sources=["docs/CLAIMS.md", "data/findings/*.json"], rows=rows)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, allow_nan=False)
    for r in rows:
        print(r["id"], [(c["id"], c["status"], c["late"]) for c in r["claims"]], r.get("numbers"), r["extra"])


if __name__ == "__main__":
    main()
