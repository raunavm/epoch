# Epoch data products

Tables measured from NASA SPHEREx DR1 Level-2 spectral images (doi:10.26131/IRSA678) by Epoch. Free to use with credit to Epoch (NASA Space Apps 2026) and to the SPHEREx mission. Methods: the Methods page of the Epoch site. Code that rebuilds these files: `scripts/build_catalogues.py`.

## epoch_nep_variability.csv — variability at the north ecliptic pole

Every catalogued source measured within 0.4° of the north ecliptic pole (RA 270°, Dec +66.56°), one row per source. Brightness measurements were corrected for errors shared by all stars (per-picture calibration offsets, position on the detector, slow drifts), using a correction learned from half of the non-variable stars. Thresholds were set on the calibration half and applied unchanged to everything else.

| Column | Meaning |
|---|---|
| gaia_or_allwise_id | Gaia DR3 source_id, or AllWISE designation for sources without a Gaia match |
| ra, dec | ICRS degrees (Gaia DR3 epoch 2016.0 or AllWISE) |
| spherex_mean_m_ab | mean SPHEREx AB magnitude over all pictures |
| G, bp_rp, W1mag, W2mag | Gaia DR3 and AllWISE photometry |
| n_pictures, n_bins | pictures measured; time bins used by the binned statistics |
| chi2r, chi2r_det, s2, ls | variability statistics: reduced χ² about the source's own static spectrum (all pictures; per detector), binned excess variance, Lomb–Scargle peak power |
| amp, P_best | fitted fractional amplitude and best period (days); descriptive only |
| z_* | each statistic as a z-score against non-variable stars of the same brightness |
| hit_* | statistic above its threshold (false-alarm rate 0.1% per statistic, set on calibration stars) |
| variable_candidate | any statistic above threshold |
| role | cal = used to learn the correction and set thresholds; test = held out; other = everything else |
| catalogued_var, vari_class, kim2026_name, milliquas_name | known variables: any catalogue, Gaia DR3 variability class, NEOWISE variable (Kim et al. 2026), Milliquas AGN |

Caveats: half of planted 1% variations are recovered at AB 12–16 (2% at 16–17, 5% at 17–19); smaller variations are not detectable. A slow 80–95-day pattern shared by many sources is reduced but not removed by the correction; check `P_best` against 79–95 days before trusting a period. The false-alarm rate on held-out stars is 0.3%, so about 8 of 2,701 sources are expected to pass by chance.

## epoch_emission_lines.csv — hydrogen emission lines

Stars that Gaia DR3 classifies as Herbig Ae/Be or Be candidates (role = candidate) and ordinary comparison stars of similar brightness in the same pictures (role = control). For each of Paschen-α (Paa, 1.875 µm), Brackett-γ (Brg, 2.166 µm) and Brackett-α (Bra, 4.05 µm): the fractional excess over the local continuum, its error, the excess statistic Z (calibrated on the controls) and whether it passes the detection rule. `line_excess` is the overall verdict. SIMBAD identifiers and types are given where they exist.

Caveat: at SPHEREx's spectral resolution, emission from the star cannot be separated from emission by surrounding gas.
- Only testable rows (column `testable` = True) count in the public totals: a star is testable when enough clean pictures fall on and around the lines.

## epoch_nep_variable_candidates.csv — vetting of the uncatalogued variability candidates (E33)

The 16 sources in `epoch_nep_variability.csv` that pass the variability thresholds but are not in the catalogues checked there (VSX, Gaia DR3 variability, Kim et al. 2026 NEOWISE variables, Milliquas), one row per source, with the result of each vetting rule (docs/protocols/E33.md). Each rule was applied in the same way to non-variable comparison stars, so the rate at which it rejects ordinary stars is known.

| Column | Meaning |
|---|---|
| gaia_or_allwise_id, ra, dec, spherex_mean_m_ab, G, W1mag, n_pictures | as in epoch_nep_variability.csv |
| chi2r, chi2r_det, s2, ls, z_*, amp, P_best | variability statistics and z-scores (corrected photometry), as in epoch_nep_variability.csv |
| pass_a_period | not at the shared 79.36 / 95.06-day periods, and not a Lomb–Scargle-only detection at 60–120 days |
| pass_b_extended | not within 3.5′ of NGC 6543, within a catalogued galaxy's size + 18″ (2MASS XSC, HyperLEDA), or within 60″ of a star with W1 < 8 or G < 9 |
| pass_c_neighbour | no brighter passing source within 6.2″ (one pixel), and not one of a group of ≥ 3 passing sources within 1′ |
| pass_d_halves | passes again in both halves of the data (detectors 1–3 vs 4–6, or first vs second half of the year) |
| pass_f_artefact | passes the E18-B crosstalk, persistence, airglow and leave-one-out checks |
| vetted_candidate | passes every rule |
| veto_reasons | the rules that rejected it, in words |
| literature | for vetted candidates: "known" or "we found no published variability"; otherwise "not checked (vetoed)" |

Result: none of the 16 pass every rule, so this table holds no vetted candidates. Most are rejected by the shared 79/95-day pattern, by the NGC 6543 nebula, or by a group of passing sources around one bright star near 18:03:16 +66:20:50. The 5 sources above threshold that are already catalogued are not in this table.

## epoch_methane_candidates.csv — methane brown-dwarf candidates (E22)

Brown-dwarf candidates from published lists (Kirkpatrick et al. 2021 and 2024, Meisner et al. 2020a/b; Dal Ponte et al. 2023 DES photometric L/T candidates) that SPHEREx shows with the methane absorption at 3.3 µm typical of T and Y dwarfs (below roughly 1300 K), and for which we found no published spectral type. One row per candidate. Rule, fixed beforehand on brown dwarfs of known type: methane index = mean flux at 3.2–3.4 µm divided by the mean of the fluxes at 2.9–3.1 and 3.9–4.1 µm; "T or later" when the index is at least 3σ below 0.356, with errors from 12 empty positions around each object. Of 985 objects measured, 687 were bright enough for a clear call: 20 "T or later", 667 "L-like". None of 63 spectroscopically typed L dwarfs measured the same way was called "T or later".

| Column | Meaning |
|---|---|
| name, simbad_ids | catalogue designation; SIMBAD identifiers where they exist |
| tier, input_catalogue, catalogue_photometric_type | tier1 = Kirkpatrick/Meisner lists, tier2 = Dal Ponte 2023 DES; the photometric type given there, if any |
| ra_deg, dec_deg, position_epoch_mjd | position used, and its epoch |
| W1, W2 | WISE magnitudes |
| n_pictures | SPHEREx pictures measured |
| methane_index, methane_index_err, sigma_below_threshold | the index, its error, and how many σ it lies below 0.356 |
| call | "T or later" for every row |
| literature_status, search_date | result of the search for a published spectral type (SIMBAD, VizieR tables, arXiv), and its date |

Caveats: these are candidates; each needs a spectrum to confirm its type. WISE 0628−8057 already has a methane-imaging photometric type (Tinney et al. 2018). Most candidates lie 3–4.7σ below the threshold. Tier-2 positions assume epoch 2016.0.

## epoch_emission_line_candidates.csv — hydrogen-line census (E23)

Stars that Gaia DR3 flags as likely Be or T Tauri stars, measured with the same hydrogen-line rule as `epoch_emission_lines.csv` (Paschen-α 1.875 µm, Brackett-α 4.05 µm; a line counts when Z ≥ 5 overall and Z_w ≥ 2 in at least two visits, each with its own continuum). The rule passed 10 of 1,719 ordinary comparison stars measured in the same pictures (0.58%). One row for each of the 88 detected stars that has no emission-line or young-star type in SIMBAD, with the results of a wider literature check. The sample is the first 691 of 816 selected fields, in a fixed random order.

| Column | Meaning |
|---|---|
| gaia_dr3_source_id, ra, dec, G | Gaia DR3 identifier, position (degrees) and G magnitude |
| gaia_espels_class | Gaia DR3 emission-line class (beStar or TTauri) |
| Paa_Z, Bra_Z, Brg_Z; *_detected; *_excess | line statistic, detection by the rule, and fractional excess over the continuum |
| simbad_main_id, simbad_type, simbad_other_emission_types | SIMBAD identifier and types |
| vsx_entries, vizier_emission_catalogues | VSX entries within 5″; emission-line, Be, Herbig, T Tauri, YSO or Hα catalogues matched within 2″ |
| only_yso_catalogue_match | matched only a YSO-candidate list |
| no_published_emission_classification_found | True for the 7 stars with no match in any catalogue checked |
| midir_excess_flag, W1_W3 | AllWISE mid-infrared excess flag and colour |
| catalogues_checked, search_date | what was searched, and when |

Caveats: at SPHEREx's spectral resolution, emission from the star cannot be separated from emission by surrounding gas. Six of the 7 unclassified stars are detected in Brackett-α only, most near the threshold. The literature check is broad but not complete.

Vetting of the methane candidates (columns `vet_verdict`, `vet_reason`, `prior_photometric_type`, `published_type_found`; checked 2026-11-15): KEEP = no published spectral type found and no significant neighbouring source; KNOWN = a published type exists (WISE 0628−8057, methane imaging); DOUBTFUL = a brighter WISE source within one to two SPHEREx pixels could bias the index. All 20 already had a T type estimated from their colours or magnitudes; SPHEREx measures their methane absorption directly. Sources: SIMBAD, VizieR brown-dwarf catalogues, CatWISE2020, AllWISE, Gaia DR3, arXiv. Not checked: the UltracoolSheet, the SpeX Prism Library and full-text searches, so "no published type" means none found in these sources.
