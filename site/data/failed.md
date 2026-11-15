# What failed, and what we changed

We list every approach that did not work, because a result is only believable if you can see what was tried and dropped. Each item says what happened and what we did about it.

## 1. Our first "corrected" view made things worse
**What happened.** The first version of the colour-aware view, run on real pictures, flagged *more* changes than a plain blink in our first test fields.
**Why.** SPHEREx's pixels are large compared with its sharpest images of stars. When pictures are lined up, bright stars land slightly differently on the pixels each time, and that leaves residues far larger than the background noise. Our outlier rejection also broke when a detector had only three or four pictures of a spot.
**What we changed.** A noise allowance proportional to each star's brightness, measured on one calibration patch of sky and then frozen before any other patch was analysed; rejection of only the single worst picture at a time. Both views (plain and corrected) use the same noise rule, so the comparison stays fair.

## 2. A fair comparison shrank our headline
**What happened.** On the control patch, a time-ordered blink shows 389 changes, the colour-aware view 57 — but simply comparing pictures from the same detector already gets 52. Against that fair baseline the colour-aware view ties.
**What we changed.** We withdrew the "removes 85%" headline and ran new, pre-registered tests: 20 random patches of sky, planted test sources at equal false-alarm rates, and recall of known asteroids (see the Science page). The headline is whatever those tests support.

## 3. Asteroid colours disagreed with Gaia — because asteroids spin
**What happened.** Comparing SPHEREx colours taken on different days with ESA Gaia's spectra gave a disagreement in spectral slope well beyond the error bars for asteroid 224 Oceana (that cross-visit comparison was not saved, so we quote no number for it).
**Why.** An asteroid spins every few hours, so its brightness changes between visits; across visits that looks like colour.
**What we changed.** Colours are compared only within one visit. Then they agree (χ²/dof 0.21 for Oceana).

## 4. Sizes from heat glow came out about 15% too large
**What happened.** Diameters from SPHEREx's 3–5 µm heat glow ranged from 7% below to 64% above the independent values for the first five asteroids (four of five too large), and the data rose faster with wavelength than the standard thermal model predicts.
**Why (likely).** At 3–5 µm SPHEREx sees the steep, short-wavelength edge of an asteroid's heat glow, where the simple thermal model is least accurate at the viewing angles involved.
**What we changed.** Sizes are no longer a headline. A calibration protocol (fit on even-numbered asteroids, test on odd-numbered ones) is written but not finished; object pages show sizes as evidence for the pixel model only.

## 5. A "water band" that depended on how you draw a line
**What happened.** A dip near 2.8 µm in asteroid 84 Klio was 18.5% with a sloped continuum but 7.9% with a flat one.
**What we changed.** The difference is now part of the error (total ±11%); no water claim is made.

## 6. Our first faint asteroid was not really invisible
**What happened.** The first object picked for the "too faint for any one picture" demonstration had a few single pictures far above the detection level — light from something it passed, not the asteroid.
**What we changed.** A quality rule (drop pictures whose fit near the object is poor), applied identically to single pictures and stacks, and a strict 5-standard-deviation definition of "detected". The asteroid now shown, 98397, was then picked by hand. Of the 8 faint asteroids we could stack, 3 meet the rule and 98397 scores lowest of them (`data/findings/f2_candidates.csv`), so its significance is not corrected for the choice.

## 7. Our error bars were too small
**What happened.** Stacking along 24 empty paths beside an asteroid gave scatter 1.5–2.5 times larger than our error bars predicted (asteroid stacks).
**What we changed.** Every stacked significance on the site is quoted against those empty paths. The check that the correction itself is stable (estimate on 12 inner paths, verify on 12 outer paths) did not pass cleanly (0.77 instead of 0.9–1.1), so we use the larger, more conservative correction.

## 8. Engineering failures
- NASA/IPAC's search and cutout services returned errors when we started (15 Nov 2026), so we built our own index of all 1,179,406 DR1 pictures from the public archive copy.
- A stale index file made an early run see only one week of data; caught when a bright asteroid was found in far fewer pictures than expected.
- Downloads with no time-out froze a run; every read now times out.

## 9. Results of the tests we fixed in advance
- **Sensitivity (planted sources, 20 random patches):** no method found half of the planted sources even at 20 times the single-picture noise, so our pre-registered comparison has no winner. See item 11.
- **False changes (20 random patches):** viewing one detector at a time cut the changes a plain blink flags by a median of 93%. Our colour-aware view flagged 26% more changes than that simpler view (95% interval 12–43%).
- **Real moving objects (known asteroids, 36 patches):** see item 14 — this did not work.
- **WISE 0855 (combined along its path):** detected; see its result card.

## 10. A neighbouring star was silently dropped
**What happened.** Our fitting code treated any catalogued star within 0.7 pixels of the target as "the target itself" and left it out. Around WISE 0855 a star of magnitude 16.3 sits 0.7 pixels from the brown dwarf's predicted position in the Apr–May 2026 visits, so its light was added to the brown dwarf's (a control at short wavelengths, where WISE 0855 is invisible, showed the leak).
**What we changed.** The star was put back for WISE 0855 (the Apr–May 2026 signal drops from 13.6 to 10.6 standard deviations against empty paths); both numbers are reported. The rule itself must be replaced by matching on catalogue identity, not distance, before any further runs.

## 11. The colour-aware view did not beat a simple median
**What happened.** In a test fixed in advance (planted sources in 20 random patches of sky, each method's alarm threshold set on a separate calibration patch), no method found half of the planted sources even at 20 times the single-picture noise, so the pre-registered comparison is a tie at "too faint for all". Our colour-aware view looked ahead at the brightest level, but only because one stray spike in the calibration patch set its threshold lower; at a common threshold a plain median of other pictures from the same detector found more, and on the test patches the colour-aware view raised more false alarms (0.014 against 0.005 per picture; the 95% intervals, 0.008–0.023 and 0.001–0.011, overlap).
**What we changed.** The headline is no longer about the colour-aware view. It stays in the tool as an experimental option, labelled as such.

## 12. A clean-up rule that also removed faint real sources
**What happened.** To stop isolated bad pixels from swamping every method, we removed pixels whose neighbours were quiet. SPHEREx's images of stars are so compact that this also removed many faint real sources, lowering every method's detection rate equally.
**What we learned.** Detecting a change in a single picture is unreliable even for bright changes (at 20 times the noise no method found half of the planted sources); fainter objects need many pictures combined along their path, which is what the stacking view does.

## 13. Measuring Barnard's Star's motion did not work
**What happened.** The motion we derive for Barnard's Star (about −5.0 ± 1.9 and +5.6 ± 1.5 arcseconds per year) disagrees with Gaia's (−0.80 and +10.36). Likely cause (untested): the star saturates SPHEREx's detector, so with its centre masked the fit rests on its outer light; and only two time windows are available. The same method works for the fainter Epsilon Indi B, where it matches Gaia.
**What we did next (E8).** We chose a new method on 26 bright comparison stars first (the centre of the star's unflagged outer glow), then measured Barnard's Star once: +10.3 ± 0.4″ per year north and −0.8 ± 0.4″ east, matching Gaia. By our own pre-set rule the result still counts as failed, because a check on 8 faint stars nearby showed a small shared drift beyond their error bars (Science page, E8). The earlier-minus-later picture and the WISE 2010 comparison still show the motion directly.


## 14. Our change maps miss real asteroids
**What happened.** In 36 patches of sky we checked every known asteroid brighter than magnitude 19 against our change maps. The colour-aware view caught 2 of 79 asteroid sightings (95% interval 0.7–8.8%) and the one-detector view caught none (0–4.6%).
**Why.** Two reasons. Most of these asteroids (57 of 79) are fainter than a single SPHEREx picture can show, so no single-picture method can catch them — that is what combining pictures along a path is for. The brighter ones were erased by us: the archive's pipeline flags isolated, unusually bright pixels within each single picture as "outliers" (sigma-clipping on a median-filtered image; SPHEREx Explanatory Supplement §3.2.10). It does not compare visits, but a moving object's peak can still trip this flag, and our change maps removed those pixels. With that removal switched off, the brightest test asteroid (V = 14.6) was caught in all 8 of its pictures (a diagnostic, not a pre-registered result).
**What this means.** Today, Orrery's change maps are a tool for seeing colour flicker and bright changes, not for finding moving asteroids. For movers, use the "combine along a path" view, and the earlier-minus-later picture for nearby stars. The outlier masking must be replaced before any search.
**We then tried that, in a new pre-registered test (E3b):** with the outlier flag left in, alarm thresholds had to rise (the flag also hides many genuine artefacts), and on 10 new, never-inspected patches every view caught 0 of 17 known-asteroid sightings brighter than magnitude 19 (95% upper limit 18%). On the 36 earlier patches (not a blind test) the colour-aware view rose to 8 of 79. Unmasking alone does not fix single-picture asteroid detection.

## 15. A star-by-star scene model did not help
**What happened.** We modelled every picture star by star using ESA Gaia's catalogue and treated what was left as change. Its noise allowance looked excellent on the calibration patch (about 4% of a star's brightness), but in the fixed-in-advance test it found essentially none of the planted sources and raised about eight times the allowed false alarms on the test patches.
**Why (likely).** Gaia catalogues stars in visible light. Infrared sources that Gaia does not list stay in every SPHEREx picture as "changes". A scene model for SPHEREx needs an infrared catalogue as well.

## 16. A per-star change test could not reach a verdict
**What happened.** We tested each star on its own: fit one smooth spectrum to all its pictures, then let its brightness shift between visits, and compare the improvement with what ordinary stars show. The error calibration worked — per-picture error bars were 1.1 to 3 times too small depending on the detector, and after a correction fixed on separate control patches the scatter of constant stars matched its error bars (spread 1.01). But we had fixed in advance a comparison set of only 7 ordinary stars, so no star could ever pass our significance rule (the best possible p-value was 0.125). RW Aurigae and R Draconis also saturate the detector: R Draconis left 3 usable pictures of 250, and RW Aurigae's fit of its outer light was poor, so a change could not be told apart from a bad fit.
**What we did next (E9).** We measured both stars from the halo of light around their overloaded centres, with errors from 41 and 43 comparison stars of similar brightness. Neither change passes our pre-set 5× rule (RW Aurigae's own pictures scatter more than ordinary stars'); the colour pattern of RW Aurigae's change and R Draconis's 4.4–5 µm light curve are shown on their cards as descriptive results.

## 17. Barnard's Star, second attempt: the motion matches, the pre-set check does not
**What happened.** The new method recovers Barnard's Star's motion within 0.2σ of Gaia, but our pre-registered criterion also required 8 faint neighbouring stars to agree with Gaia within their own errors. They share a drift of about −0.17″ per year east–west, so the registered verdict is FAILED.
**Why (likely, read after the result).** The drift reverses with the detector's orientation between the two visits, which points to a tiny fixed offset in detector pixels (about 0.01 pixel). The same effect makes ordinary stars show a small false negative parallax in the distance test (E12), where it is inside the error bar.
**What we would do next.** Calibrate positions in the detector frame on many stars, then repeat both tests.

## 18. A blind asteroid test fell short of its own success level
**What happened.** We drew 12 new faint asteroids at random and, before measuring, wrote down how clearly each should show up when its pictures are combined. We had required that at least 70% of those expected to show clearly reach 5σ; 3 of 5 did (60%). The brightness part of the test passed (0.92 times the prediction), and no empty path raised a false alarm (0 of 288).
**Why (flagged before the run).** The prediction used the noise the fit reports; empty paths show the real noise is about 2.5 times larger. With the measured noise, prediction and measurement agree. For some asteroids about half of the pictures failed to download during the run; the files are on the public archive (checked afterwards), so the failures were temporary.
**What we would do next.** Predict with the empty-path noise from the start, and test on a larger blind sample.
