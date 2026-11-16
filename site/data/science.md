# Results

NASA's SPHEREx telescope maps the whole sky in 102 infrared wavelength bands between 0.75 and 5 µm, photographing each position many times over a year. Epoch reads the 1,179,406 pictures of the first data release (taken 2025-04-24 to 2026-05-14) directly from NASA's public archive and shows any position in time order. The results below use the same pictures.

## Removing false changes {#which-change-is-real}

SPHEREx takes each picture at a different wavelength with one of six detectors, so a plain blink between consecutive pictures shows many changes where the sky has not changed. Comparing each picture with the previous picture from the same detector removes a median 93% of these changes (95% interval 90–96%) across 20 randomly chosen patches (4,906 pictures). This removes false changes but does not make faint real changes easier to detect: planted sources at 20 times the single-picture noise are recovered at similar rates by one detector at a time and by the plain blink (37% and 34%), at a shared threshold where the plain blink also flags 12,686 false changes against 599. For finding faint new sources, comparing each picture with the median of all the other pictures from the same detector works better. On 20 new patches, at the same false-alarm rate (about 3 false changes per 100 pictures), it finds 72% of planted sources at 20 times the single-picture noise, against 28% for one detector at a time. The learned detector in the live view starts from this median reference. The change maps also allow every source an uncertainty of 15% of its own brightness, so they flag a bright star only if it changes by several times that; smaller variations of bright stars are measured from their brightness instead (see the north ecliptic pole and R Draconis below).

[Methods, section 2](methods.html#change)

![Changes per patch in a plain blink and one detector at a time](figs/v2/fig_e2.svg "Changes flagged in each patch by a plain blink (crosses), one detector at a time (circles) and the experimental colour-aware view (triangles), ordered by Galactic latitude. Counts are summed over all pictures of a patch; the vertical axis is logarithmic.")

## A learned change detector {#detector}

Without the archive's quality flags, which the live view does not read, a change map is dominated by cosmic-ray hits and bad pixels. The live view therefore ranks possible changes with a small neural network (about 7,000 parameters) that runs in the browser. It was trained on simulated sources placed into real SPHEREx pictures. Tested on 20 sky fields it never saw, at a setting with about 3 false marks per 100 pictures, it finds 41% of the simulated sources (more of the brighter ones), where change statistics without quality flags find almost none. With the archive's quality flags, the median reference finds 18%; on the small default view the network does no better than that. In a test on real asteroids it found none of 32 sightings, where the flagged median reference found 12, so it is not yet shown to work on real moving objects.

[Methods, section 3](methods.html#detector)

![Recovery of injected sources by the learned detector and the baselines](figs/v2/e30_recall.svg "Fraction of injected sources recovered on the 20 test fields against their S/N per picture, at 0.03 false candidates per picture, for the network (no quality flags), the median reference with the archive's flags (a reference, not part of the test rule), and the median reference and one-detector comparison without flags. Shaded bands are 95% intervals.")

## WISE 0855−0714 {#e4}

WISE 0855−0714, the coldest known brown dwarf, is barely above the noise in a single picture. Combining the 22 pictures from the November–December 2025 visits along its predicted path detects it at 12.1σ: its signal exceeds that of the same pictures combined along 200 empty paths by about 12 times their scatter, and none of the paths is as bright. Its flux density at 4.4–5.0 µm is 0.50 ± 0.04 mJy (the error calibrated on the empty paths), consistent with Spitzer's 0.51 mJy at 4.5 µm (Luhman & Esplin 2016) and within the 0.41–0.50 mJy spanned by published WISE measurements. Over the 10 wavelength steps where it is detected, the SPHEREx spectrum is 0.97 ± 0.03 times the JWST NIRSpec spectrum (Luhman et al. 2024). The two reddest steps, at 4.8–5.0 µm, fall below it. Passing the JWST spectrum through each SPHEREx picture's own spectral response, rather than averaging it in 0.1 µm steps, gives the same ratio, 0.97 ± 0.03, and the two reddest steps stay low. A SPHEREx spectrum of WISE 0855 has been published (Rustamkulov et al. 2026); ours is an independent comparison with JWST.

[Methods, section 4](methods.html#stacking)

![SPHEREx spectrum of WISE 0855−0714 next to JWST](figs/v2/e11_wise0855.svg "SPHEREx spectrum in 0.1 µm steps (circles; open circles are below 3σ; bars are 1σ), the JWST NIRSpec spectrum (thin line) averaged over the same steps (squares), and the Spitzer 4.5 µm measurement (diamond).")

## Distance to Epsilon Indi B {#e12}

Epsilon Indi B is a pair of brown dwarfs that SPHEREx sees as one source. Fitting its SPHEREx positions over a year for position, proper motion and parallax gives a parallax of 243 ± 53 mas, a distance of 13.4 light-years (68% interval 11.0–17.2). The archive's astrometry is tied to Gaia DR3 reference stars, but no Gaia measurement of Epsilon Indi B itself is used; Gaia measures 270.7 mas. Ordinary stars in the same pictures show a small shared offset that mimics a parallax of -50 mas; it is not subtracted, and the quoted uncertainty is the root-mean-square of those 8 stars' parallaxes about zero, which includes their shared offset.

[Methods, section 5](methods.html#astrometry)

![Epsilon Indi B's path and parallax](figs/v2/e12_parallax.svg "a, positions over a year: single pictures (small points), visit averages (circles), and fits with proper motion only (dotted) and with parallax (dash-dot). b, visit averages with the proper motion removed, and the fitted parallax ellipse. Bars are the standard error of each visit mean.")

## Brown dwarfs classified by methane {#e15}

Methane absorption at 3.3 µm distinguishes T and Y dwarfs from L dwarfs. An index formed from the flux in the 3.2–3.4 µm window relative to the flux on either side classifies 91% of 23 brown dwarfs with published spectral types correctly when each is left out of the calibration in turn.

[Methods, section 6](methods.html#spectra)

![Methane index against colour](figs/v2/e15_methane.svg "The 4.4–5.0 µm to 1.0–1.3 µm flux ratio against the methane index, for brown dwarfs of known spectral type and for candidates without a published spectrum. The dashed line is the dividing value; bars are 1σ.")

## Candidate cold brown dwarfs {#methane-candidates}

We applied the same methane test, unchanged, to 985 objects that earlier surveys listed as possible brown dwarfs. 687 were bright enough for a clear answer: 20 show the methane dip of cold brown dwarfs (T type or later, below roughly 1300 K), and 667 do not. None of 53 known warmer L dwarfs measured the same way was wrongly given the dip. We found no published spectral type for any of the 20; they are candidates for cold, nearby brown dwarfs, and each needs a spectrum to confirm it.

[Download the table (CSV)](products/epoch_methane_candidates.csv) · [Column descriptions](products/README.md) · [Methods, section 6](methods.html#spectra)

![Methane index of brown-dwarf candidates](figs/v2/e22_methane_census.svg "Methane index (flux at 3.2–3.4 µm relative to the neighbouring wavelengths; lower means deeper methane absorption) against WISE W2 magnitude, for the 985 objects and the known L dwarfs. Stars are the candidates; the dashed line is the dividing value. Objects with no defined index are not shown, and the axis is cut at −1.5 and 2. Bars are 1σ.")

## Variability at the north ecliptic pole {#e18}

SPHEREx observes the north ecliptic pole on almost every orbit. Within 0.4° of the pole, 2,701 catalogued sources were measured in 17,832 pictures. The measurements of all stars share small errors: calibration offsets that differ from picture to picture, a dependence on position on the detector, and slow drifts over the year. We fit a correction for these to half of the non-variable stars and test it on the other half. With the correction, half of planted sinusoidal variations are recovered at an amplitude of 1% for sources of AB magnitude 12–16, 2% at 16–17 and 5% at 17–19, at a false-alarm rate of 0.30%. Most catalogued variables in the field vary by less than this; 5 of 86 pass the threshold.

[Download the table (CSV)](products/epoch_nep_variability.csv) · [Column descriptions](products/README.md) · [Methods, section 7](methods.html#variability)

![Recovery of planted brightness variations](figs/v2/e26_floor.svg "Fraction of planted sinusoidal variations recovered against source brightness, without (a) and with (b) the correction for shared errors, for amplitudes of 1% to 20%. Shaded bands are 95% intervals; the dotted line marks half recovered.")

## Pulsation of R Draconis {#rdra}

R Draconis is a Mira variable with a catalogued period of 246 days. It saturates the detector, so it is measured in an annulus around the saturated core, with errors set by comparison stars. A sine fit to the pictures measured first predicted a maximum at 4.4–5.0 µm on 2026-02-02. In 512 pictures not used for that fit (461 of them at 4.4–5.0 µm), the star varies by 0.34 ± 0.03 mag either side of its mean, with a fitted maximum on 2026-02-04 ± 2 days, 2.5 days after the prediction. The light curve departs from a sine: with errors scaled to the scatter about the fit, the amplitude is 6σ. At 3.8–4.4 µm the new pictures cover too few dates for a fit.

[Methods, section 7](methods.html#variability)

![Light curve of R Draconis](figs/v2/e28_rdra.svg "Brightness relative to the mean at 4.4–5.0 µm (a) and 3.8–4.4 µm (b), in 30-day bins: pictures not used for the prediction (filled circles) and pictures used for it (open circles), with the predicted curve (dash-dot), the fitted sine (solid; a only) and the scatter of comparison stars (grey band). Bars are 1σ.")

## Hydrogen emission lines {#e20}

SPHEREx covers the Paschen-α (1.875 µm) and Brackett-α (4.05 µm) hydrogen lines. Of 54 stars that Gaia classifies as Herbig Ae/Be or Be stars, 37 show an excess in these lines, against 2 of 155 ordinary stars of similar brightness in the same pictures. At SPHEREx's spectral resolution, emission from the star cannot be separated from emission by surrounding gas.

[Download the table (CSV)](products/epoch_emission_lines.csv) · [Column descriptions](products/README.md) · [Methods, section 6](methods.html#spectra)

![Hydrogen-line excess](figs/v2/e20_lines.svg "a, b, flux relative to the local continuum around Paschen-α (a) and Brackett-α (b) for one candidate (stars) and a control star in the same pictures (crosses); bars are 1σ. c, distribution of the line excess for candidates and controls; the dashed line is the detection threshold.")

## More emission-line stars {#emission-candidates}

We ran the same hydrogen-line test, unchanged, on 691 more stars that Gaia flags as likely Be or T Tauri stars. 241 of 651 that could be measured show line emission, against 10 of 1,719 ordinary stars. For 7 we found no published emission-line classification; they are candidates that need a spectrum to confirm.

[Download the table (CSV)](products/epoch_emission_line_candidates.csv) (88 rows; the column `no_published_emission_classification_found` marks the 7) · [Column descriptions](products/README.md) · [Methods, section 6](methods.html#spectra)

## Faint asteroids {#stacking-along-an-orbit}

Combining pictures along a known orbit recovers asteroids that no single picture detects: of 7 asteroids with no single detection above 5σ, 3 exceed 5σ when combined. In a blind sample of new asteroids, only 3 of the 5 expected to reach 5σ did so: the noise in the combined pictures is about 2.5 times the formal errors. Their measured brightness matched the prediction (median ratio 0.92).

[Methods, section 4](methods.html#stacking)

![Signal-to-noise of an asteroid as pictures are combined](figs/v2/fig_reveal.svg "a, signal-to-noise ratio of asteroid 98397 against the number of pictures combined along its orbit, with the √N expectation. b, signal-to-noise ratio of the single pictures and of empty paths beside the orbit.")

## A search for distant bodies {#distant-bodies}

A body hundreds of AU from the Sun moves on the sky mainly through parallax, tracing an ellipse as the Earth orbits. Measuring that ellipse needs pictures spread over a year, and the north ecliptic pole is where SPHEREx has them: it observes the pole on almost every orbit. This search is a test of the method there. We searched 0.044 deg², observed in 30 visits over 384 days, for sources whose positions follow a parallax ellipse, and found no candidate. A body injected at 500 AU is recovered at a fitted distance of 499.7 ± 1.0 AU, and at that distance half of injected bodies with a radius of 14 Earth radii, about 1.3 times Jupiter's radius (geometric albedo 0.3, reflected light), are recovered. The same search at the south ecliptic pole, with 26 visits over 93 days, also found no candidate, but it did not recover the injected 500 AU body: over so short a span, parallax cannot be separated from the body's steady motion. The field is too small, and single pictures too shallow, to exclude a planet-sized body.

[Methods, section 9](methods.html#movers)

![Recovery of injected distant bodies](figs/v2/e21_completeness.svg "Fraction of injected bodies recovered against heliocentric distance, for seven per-visit signal-to-noise ratios (48 injections per point). Bars are 68% Wilson intervals; the dotted line marks half recovered.")

## Candidates for follow-up {#candidates}

Each of these is a candidate, not a confirmed object. Each needs a spectrum or another observation.

- 20 possible cold brown dwarfs, from the methane test ([section](#methane-candidates), [CSV](products/epoch_methane_candidates.csv)).
- 7 possible emission-line stars with no published classification ([section](#emission-candidates), [CSV](products/epoch_emission_line_candidates.csv)).
- North ecliptic pole: 16 variables missing from the catalogues were checked and 0 survive ([section](#e18), [CSV](products/epoch_nep_variable_candidates.csv)).

## References

- Bock, J. et al. 2026, The SPHEREx Satellite Mission, ApJ. https://doi.org/10.3847/1538-4357/ae2be2
- Akeson, R. et al. 2026, The SPHEREx Image and Spectrophotometry Processing Pipeline, ApJS. https://doi.org/10.3847/1538-4365/ae8843
- SPHEREx DR1 Spectral Images, doi:10.26131/IRSA678. https://doi.org/10.26131/IRSA678
- Luhman, K. L. & Esplin, T. L. 2016, AJ 152, 78. https://arxiv.org/abs/1605.06655
- Luhman, K. L. et al. 2024, AJ 167, 5. https://doi.org/10.3847/1538-3881/ad0b72
- Rustamkulov, Z. et al. 2026, ApJ. https://doi.org/10.3847/1538-4357/ae9d66
- Gaia Collaboration, Vallenari, A. et al. 2023, A&A 674, A1. https://doi.org/10.1051/0004-6361/202243940
