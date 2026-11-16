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
