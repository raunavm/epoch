"""Fill [[file.key]] number tokens in the submission texts from the measured findings.

Sources: <name> = data/findings/<name>.json (every file there, e.g. f1, e2, e4, motion_epsindib),
meta = site/data/index/dr1/meta.json, headline = site/data/headline.json.
A token is [[file.dotted.path]] or [[file.dotted.path|fmt]], fmt in {pct, int, date, raw, f0-f3, k0-k3}
(fN: N decimals; kN: value/1000 with N decimals). List items are addressed by index (passes.1.x).
Default format: key containing "frac", "fraction" or "pct" -> percentage; int -> 1,234; float -> 2 significant
figures; string -> as is. `date` turns an MJD into YYYY-MM-DD.
Any sentence holding a token that cannot be filled (file or key missing, null, NaN) is deleted; a table row
or caption line is deleted whole. Keep each numbered sentence on one line and self-contained.

The first run copies each target to <target>.tmpl; every run fills from the .tmpl, so edit the .tmpl and re-run.
Usage: python3 scripts/fill_tokens.py
"""
import glob, json, math, os, re, shutil, sys
from datetime import datetime, timedelta

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SOURCES = {os.path.basename(p)[:-5]: os.path.relpath(p, ROOT) for p in sorted(glob.glob(os.path.join(ROOT, "data/findings/*.json")))} \
    | {"meta": "site/data/index/dr1/meta.json", "headline": "site/data/headline.json"}
TARGETS = ["docs/READY_TO_SEND.md", "docs/video_script.md", "docs/slides.md", "docs/video_captions.srt",
           "site/data/science.md", "site/data/about.json"]
TOKEN = re.compile(r"\[\[([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+)(?:\|(pct|int|date|raw|[fk][0-3]))?\]\]")
ABBREV = re.compile(r"(?:\b(?:al|e\.g|i\.e|vs|No|St|cf|approx)\.)$")


def load_sources():
    data = {}
    for name, rel in SOURCES.items():
        try:
            data[name] = json.load(open(os.path.join(ROOT, rel)))
        except (OSError, ValueError):
            pass
    return data


def sig2(x):
    if x == 0:
        return "0"
    d = 1 - int(math.floor(math.log10(abs(x))))
    r = round(x, d)
    return f"{r:,.{max(d, 0)}f}"


def fmt(v, key, how):
    if v is None or isinstance(v, (dict, list)) or (isinstance(v, float) and not math.isfinite(v)):
        return None
    if how == "raw" or isinstance(v, str):
        return str(v)
    if how and how[0] in "fk":
        return f"{float(v) / (1000 if how[0] == 'k' else 1):.{int(how[1])}f}"
    if how == "date":
        return (datetime(1858, 11, 17) + timedelta(days=float(v))).strftime("%Y-%m-%d")
    if how == "pct" or (how is None and re.search(r"frac|pct", key)):
        p = 100 * float(v)
        return (f"{round(p):d}" if abs(p) >= 10 else sig2(p)) + "%"
    if how == "int" or isinstance(v, int):
        return f"{int(round(v)):,}"
    return sig2(float(v))


def resolve(data, path, how):
    head, *keys = path.split(".")
    v = data.get(head)
    for k in keys:
        if isinstance(v, dict) and k in v:
            v = v[k]
        elif isinstance(v, list) and k.isdigit() and int(k) < len(v):
            v = v[int(k)]
        else:
            return None
    return fmt(v, keys[-1], how)


def fill_text(text, data, deleted, where):
    """Fill one block of text (a line or a JSON string). Returns None if nothing is left."""
    if not TOKEN.search(text):
        return text
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[*\"“¿¡])", text)
    sentences = []
    for p in pieces:  # re-join splits made after abbreviations
        if sentences and ABBREV.search(sentences[-1]):
            sentences[-1] += " " + p
        else:
            sentences.append(p)
    kept = []
    for s in sentences:
        out = TOKEN.sub(lambda m: resolve(data, m.group(1), m.group(2)) or "\0", s)
        if "\0" in out:
            deleted.append((where, s.strip()))
        else:
            kept.append(out)
    return " ".join(kept) if kept else None


def fill_lines(src, data, deleted, name):
    out = []
    for i, line in enumerate(src.split("\n"), 1):
        if not TOKEN.search(line):
            out.append(line)
            continue
        where = f"{name}:{i}"
        whole = line.lstrip().startswith("|") or name.endswith(".srt")
        if whole:
            filled = TOKEN.sub(lambda m: resolve(data, m.group(1), m.group(2)) or "\0", line)
            if "\0" in filled:
                deleted.append((where, line.strip()))
                continue
            out.append(filled)
            continue
        m = re.match(r"^(\s*(?:[-*+]|\d+\.|#{1,6}|>)?\s*)(.*)$", line)
        body = fill_text(m.group(2), data, deleted, where)
        if body is not None:
            out.append(m.group(1) + body)
    return "\n".join(out)


def fill_json(obj, data, deleted, where):
    if isinstance(obj, str):
        return fill_text(obj, data, deleted, where)
    if isinstance(obj, list):
        vals = [fill_json(x, data, deleted, f"{where}[{i}]") for i, x in enumerate(obj)]
        return [v for v in vals if v is not None]
    if isinstance(obj, dict):
        return {k: (v if (v := fill_json(x, data, deleted, f"{where}.{k}")) is not None else "") for k, x in obj.items()}
    return obj


def main():
    data = load_sources()
    print("sources found:", ", ".join(sorted(data)) or "none", "| missing:",
          ", ".join(sorted(set(SOURCES) - set(data))) or "none")
    deleted = []
    for rel in TARGETS:
        path = os.path.join(ROOT, rel)
        tmpl = path + ".tmpl"
        if not os.path.exists(tmpl):
            if not os.path.exists(path):
                print(f"skip {rel}: not found")
                continue
            shutil.copyfile(path, tmpl)
        src = open(tmpl, encoding="utf-8").read()
        if rel.endswith(".json"):
            out = json.dumps(fill_json(json.loads(src), data, deleted, rel), indent=1, ensure_ascii=False) + "\n"
        else:
            out = fill_lines(src, data, deleted, rel)
        open(path, "w", encoding="utf-8").write(out)
        print(f"filled {rel}")
    print(f"\n{len(deleted)} sentence(s) deleted (unfillable token):")
    for where, s in deleted:
        print(f"  {where}: {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
