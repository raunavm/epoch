"""Add Spanish `<key>_es` fields to the generated site data files.

Translations come from site/data/i18n/data_es.json, an exact English -> Spanish map. A field whose English
text has an entry gets `<key>_es`; one without keeps only English (the site then shows the English text).
Run after the build scripts (they rewrite these files); running it again changes nothing.
  python3 scripts/add_es.py            # write the _es fields, print counts
  python3 scripts/add_es.py --missing  # also print the untranslated strings as a JSON list
"""
import glob, json, sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "site" / "data"
TM = json.loads((DATA / "i18n" / "data_es.json").read_text())


def each(objs):
    return [o for o in objs or [] if isinstance(o, dict)]


def fields(name, d):
    """(object, key) pairs to translate in one data file (path relative to site/data)."""
    if name == "headline.json":
        return [(d, k) for k in ("question", "sentence", "number_label", "err_label", "spread_note", "evidence_note")] + \
               [(e, "text") for e in each(d.get("evidence"))]
    if name == "findings.json":
        return [(f, k) for f in d.values() if isinstance(f, dict) for k in ("title", "sentence")]
    if name.startswith("targets/"):
        out = [(d, k) for k in ("title", "plain_result", "status", "method_note", "slider_caption")]
        out += [(n, k) for n in each(d.get("numbers")) for k in ("label", "nulls_note", "value")]
        out += [(e, "text") for e in each(d.get("evidence_links"))]
        return out + [(f, k) for f in each(d.get("figures")) for k in ("alt", "caption")]
    if name in ("reveal.json", "e4/reveal.json"):
        return [(d, "hedge")]
    if name == "timemachine/index.json":
        return [(f, "name") for f in each(d)]
    if name.startswith("timemachine/"):
        return [(d, "name")]
    return []


def main():
    names = ["headline.json", "findings.json", "reveal.json", "e4/reveal.json"] + \
            sorted(str(Path(p).relative_to(DATA)) for p in glob.glob(str(DATA / "targets" / "*.json"))) + \
            sorted(str(Path(p).relative_to(DATA)) for p in glob.glob(str(DATA / "timemachine" / "*.json")))
    done, missing = 0, []
    for name in names:
        path = DATA / name
        if not path.exists():
            continue
        raw = path.read_text()
        d = json.loads(raw)
        for obj, key in fields(name, d):
            en = obj.get(key)
            if not isinstance(en, str) or not en.strip():
                continue
            if en in TM:
                obj[f"{key}_es"] = TM[en]
                done += 1
            else:
                obj.pop(f"{key}_es", None)  # never keep a translation of older English text
                missing.append(en)
        # Keep each builder's layout: compact or indent=1, escaped or raw non-ASCII.
        compact = "\n" not in raw.strip()
        out = json.dumps(d, indent=None if compact else 1, separators=(",", ":") if compact else None,
                         ensure_ascii=raw.isascii()) + ("\n" if raw.endswith("\n") else "")
        if out != raw:
            path.write_text(out)
    print(f"add_es: {done} strings translated, {len(missing)} missing ({len(set(missing))} distinct)")
    if "--missing" in sys.argv:
        print(json.dumps(sorted(set(missing)), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
