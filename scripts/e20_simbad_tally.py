"""Post-hoc tally (after the E20 result): SIMBAD object types of the E20 detections -> data/findings/e20s.json."""
import collections, json

d = json.load(open("data/findings/e20.json"))
det = [s for s in d["stars"] if s["role"] == "candidate" and s.get("excess_main")]
EMISSION = {"Em*", "Be*", "Ae*", "Ae?", "Be?", "Or*", "TT*", "Y*O", "Y*?", "Ce*"}  # emission-line or young-star types
known = [s for s in det if s.get("simbad_otype") in EMISSION]
out = dict(source="data/findings/e20.json", note="post hoc (after the result); SIMBAD via the CfA mirror, queried by E20",
           n_detected=len(det), n_simbad_emission_type=len(known), n_no_emission_type=len(det) - len(known),
           n_not_in_simbad=sum(1 for s in det if not s.get("simbad_otype")),
           otypes=dict(collections.Counter(str(s.get("simbad_otype")) for s in det)),
           no_emission_type=[dict(source_id=s["source_id"], cls=s["cls"], G=round(s["G"], 2), simbad=s.get("simbad_main_id"),
                                  otype=s.get("simbad_otype"), midir_flag=s.get("midir_excess_flag")) for s in det if s not in known])
json.dump(out, open("data/findings/e20s.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "no_emission_type"})
