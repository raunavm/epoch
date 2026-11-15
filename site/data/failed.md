# What failed, and what we changed

We list every approach that did not work, because a result is only believable if you can see what was tried and dropped. Each item says what happened and what we did about it.

## 1. Our first "corrected" view made things worse
**What happened.** The first version of the colour-aware view, run on real pictures, flagged *more* changes than a plain blink (for example 53 against 46 around WISE 0855, and 591 against 277 in a crowded field).
**Why.** SPHEREx's pixels are large compared with its sharpest images of stars. When pictures are lined up, bright stars land slightly differently on the pixels each time, and that leaves residues far larger than the background noise. Our outlier rejection also broke when a detector had only three or four pictures of a spot.
**What we changed.** A noise allowance proportional to each star's brightness, measured on one calibration patch of sky and then frozen before any other patch was analysed; rejection of only the single worst picture at a time. Both views (plain and corrected) use the same noise rule, so the comparison stays fair.

## 2. A fair comparison shrank our headline
**What happened.** On the control patch, a time-ordered blink shows 389 changes, the colour-aware view 57 — but simply comparing pictures from the same detector already gets 52. Against that fair baseline the colour-aware view ties.
**What we changed.** We withdrew the "removes 85%" headline and ran new, pre-registered tests: 20 random patches of sky, planted test sources at equal false-alarm rates, and recall of known asteroids (see the Science page). The headline is whatever those tests support.

## 3. Asteroid colours disagreed with Gaia — because asteroids spin
**What happened.** Comparing SPHEREx colours taken on different days with ESA Gaia's spectra gave a slope three standard deviations off (6.8 ± 1.1 against 3.0 ± 0.1 % per 0.1 µm for asteroid 224 Oceana).
**Why.** An asteroid spins every few hours, so its brightness changes between visits; across visits that looks like colour.
**What we changed.** Colours are compared only within one visit. Then they agree (χ²/dof 0.21 for Oceana).

## 4. Sizes from heat glow came out about 15% too large
**What happened.** Diameters from SPHEREx's 3–5 µm heat glow were 14–15% larger than NEOWISE's for three of the first five asteroids, and the data rose faster with wavelength than the standard thermal model predicts.
**Why (likely).** At 3–5 µm SPHEREx sees the steep, short-wavelength edge of an asteroid's heat glow, where the simple thermal model is least accurate at the viewing angles involved.
**What we changed.** Sizes are no longer a headline. A calibration protocol (fit on even-numbered asteroids, test on odd-numbered ones) is written but not finished; object pages show sizes as evidence for the pixel model only.

## 5. A "water band" that depended on how you draw a line
**What happened.** A dip near 2.8 µm in asteroid 84 Klio was 18.5% with a sloped continuum but 7.9% with a flat one.
**What we changed.** The difference is now part of the error (total ±11%); no water claim is made.

## 6. Our first faint asteroid was not really invisible
**What happened.** The first object picked for the "too faint for any one picture" demonstration had three single pictures above 20 standard deviations — light from something it passed, not the asteroid.
**What we changed.** A quality rule (drop pictures whose fit near the object is poor), applied identically to single pictures and stacks, and a strict 5-standard-deviation definition of "detected".

## 7. Our error bars were too small
**What happened.** Stacking along 24 empty paths beside an asteroid gave scatter 1.5–3 times larger than our error bars predicted.
**What we changed.** Every stacked significance on the site is quoted against those empty paths. The check that the correction itself is stable (estimate on 12 inner paths, verify on 12 outer paths) did not pass cleanly (0.77 instead of 0.9–1.1), so we use the larger, more conservative correction.

## 8. Engineering failures
- NASA/IPAC's search services were down during the hackathon, so we built our own index of all 1,179,406 DR1 pictures from the public archive copy.
- A stale index file made an early run see only one week of data; caught when a bright asteroid was found in 18 pictures instead of 282.
- Downloads with no time-out froze a run for 40 minutes; every read now times out.

## 9. Results of the pre-registered tests
(Filled in from data/findings/e1.json–e4.json as each test finishes, whatever the outcome.)

## 10. A neighbouring star was silently dropped
**What happened.** Our fitting code treated any catalogued star within 0.7 pixels of the target as "the target itself" and left it out. Around WISE 0855 a star of magnitude 16.3 sits 0.7 pixels from the brown dwarf's predicted position in the third sky survey, so its light was added to the brown dwarf's (a control at short wavelengths, where WISE 0855 is invisible, showed the leak).
**What we changed.** The star was put back for WISE 0855 (third-survey signal drops from 13.6 to 10.6 standard deviations against empty paths); both numbers are reported. The rule itself must be replaced by matching on catalogue identity, not distance, before any further runs.

## 11. The colour-aware view did not beat a simple median
**What happened.** In a test fixed in advance (planted sources in 20 random patches of sky, each method's alarm threshold set on a separate calibration patch), no method found half of the planted sources even at 20 times the single-picture noise, so the pre-registered comparison is a tie at "too faint for all". Our colour-aware view looked ahead at the brightest level, but only because one stray spike in the calibration patch set its threshold lower; at a common threshold a plain median of other pictures from the same detector found more, and on the test patches the colour-aware view raised about three times as many false alarms.
**What we changed.** The headline is no longer about the colour-aware view. It stays in the tool as an experimental option, labelled as such.

## 12. A clean-up rule that also removed faint real sources
**What happened.** To stop isolated bad pixels from swamping every method, we removed pixels whose neighbours were quiet. SPHEREx's images of stars are so compact that this also removed many faint real sources, lowering every method's detection rate equally.
**What we learned.** Detecting a change in a single picture only works for bright changes (about 20 times the noise or more); fainter objects need many pictures combined along their path, which is what the stacking view does.

## 13. Measuring Barnard's Star's motion did not work
**What happened.** Barnard's Star is bright enough to saturate SPHEREx's detector. With its saturated centre masked, our position fits rest on the star's outer light, and the motion we derive (about −5.0 ± 1.9 and +5.6 ± 1.5 arcseconds per year) disagrees with Gaia's (−0.80 and +10.36). The same method works for the fainter Epsilon Indi B, where it matches Gaia.
**What we show instead.** The difference between the two sky surveys, where the star's light visibly shifts north as expected — as a picture, not as a measurement.
