"""Plain statements of what SPHEREx stacks can and cannot see, from MEASURED recovery depth only.
Reads site/data/injections.json (V50_stack) and evaluates the reflected-light brightness MODEL for a few
illustrative bodies (orrery.physics.v_magnitude; opposition geometry Δ = r − 1 au, phase 0).
Writes site/data/limits.json {sentences[], examples[]}.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from orrery.physics import v_magnitude


def V(D, pV, r):
    H = 5 * np.log10(1329.0 / (D * np.sqrt(pV)))
    return float(v_magnitude(H, 0.15, r, r - 1.0, 0.0))


j = json.load(open("site/data/injections.json"))
V50 = j["V50_stack"]
N = j["n_frames"]
ex = [("A Pluto-sized world (2,400 km), bright icy surface, at 100 au", 2400, 0.6, 100),
      ("An Earth-sized planet (12,742 km), albedo 0.5, at 250 au", 12742, 0.5, 250),
      ("A planet four times Earth's width (51,000 km), albedo 0.5, at 500 au", 51000, 0.5, 500),
      ("The same planet at 700 au", 51000, 0.5, 700)]
examples = []
for name, D, pV, r in ex:
    v = V(D, pV, r)
    examples.append(dict(name=name, V=round(v, 1), detectable=bool(v < V50), label="MODEL brightness vs MEASURED depth"))
s = [f"Combining {N} SPHEREx pictures along a known path recovers planted test objects half the time down to visual magnitude {V50:.1f} (measured).",
     "This assumes we already know where the object moves. Searching for an unknown object means testing many possible paths, which raises the bar for a believable detection; that search is not done here.",
     "Planted objects are grey (equally reflective at all colours). Most real asteroids are redder, so they look brighter to SPHEREx than their visual magnitude suggests: two real asteroids near magnitude 20.9 and 21.5 were detected by stacking (see the reveal).",
     "Reflected sunlight only: a planet that glows from its own internal heat could be brighter in SPHEREx's longest colours than shown."]
for e in examples:
    s.append(f"{e['name']}: about magnitude {e['V']} — " + ("bright enough for a SPHEREx stack, if its path were known." if e["detectable"] else "too faint for a SPHEREx stack."))
s.append("Very bright planted objects are not a test of SPHEREx: real detectors saturate near magnitude 10–11.5, which the planted objects do not model.")
json.dump(dict(sentences=s, examples=examples, V50_stack=V50, source="site/data/injections.json"), open("site/data/limits.json", "w"), indent=1)
print("\n".join(s))
