"""Flatten data/findings/e18b.json into data/findings/e18bs.json (token-friendly keys) for the site text."""
import json

d = json.load(open("data/findings/e18b.json"))
fl, kc = d["injections"]["floor_smallest_A_with_recall_ge_50pct"], d["known_variables"]["classes"]
agn = d["post_hoc"]["population_test_agn_vs_allwise_only"]["results"]["Kim or Milliquas"]
fa = d["false_alarm_test_half"]["any"]
out = dict(source="data/findings/e18b.json", n_pictures=d["prep"]["n_pictures"], n_rows_kept=d["prep"]["kept"],
           n_null=d["n_null"]["test"], fa_k=fa["k"], fa_n=fa["n"], fa_rate=fa["rate"], fa_hi=fa["wilson95"][1],
           floor_sin_12_17=fl["sin"]["12-13"], floor_sin_17_18=fl["sin"]["17-18"], floor_sin_18_19=fl["sin"]["18-19"],
           floor_ecl_12_15=fl["ecl"]["12-13"], floor_ecl_16_17=fl["ecl"]["16-17"],
           n_injections=d["injections"]["n"],
           kim=dict((k, kc["Kim+2026 NEOWISE NEP variable"][k]) for k in ("n", "n_above")),
           allvar=dict((k, kc["all catalogued variables"][k]) for k in ("n", "n_above")),
           n_cand_above=d["candidates"]["n_above_noncatalogued"], n_cand_survivors=d["candidates"]["n_survivors"],
           agn_n=agn["n"], agn_n_control=agn["n_control"], agn_amp_median_z=agn["amp"]["median_z"],
           agn_amp_control_median_z=agn["amp"]["control_median_z"], agn_amp_p=agn["amp"]["mann_whitney_p_greater"])
import statistics
P = d["known_variables"]["periods"]
out.update(kim_median_m_ab=kc["Kim+2026 NEOWISE NEP variable"]["median_m_ab"], n_with_period=len(P),
           median_amp_with_period=statistics.median(abs(p["sine_amp_at_P_cat"][0]) for p in P if p.get("sine_amp_at_P_cat")),  # [amplitude, error]
           n_cand_nebula_veto=sum(1 for c in d["candidates"]["list"] if c["checks"].get("V8_nebula", {}).get("passed") is False),
           n_common_mode=d["post_hoc"]["common_mode_ls_periods"]["n_sources_P_best_79p36_or_95p06"],
           expected_false_realised=d["candidates"]["expected_false"]["realised_any_rate_x_n"],
           floor_ecl_17_18=fl["ecl"]["17-18"], agn_chi2_p=agn["chi2r"]["mann_whitney_p_greater"])
json.dump(out, open("data/findings/e18bs.json", "w"), indent=1)
print(out)
