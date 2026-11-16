"""Turn per-epoch detections into candidate moving objects.

:mod:`tracklet.search.detect` answers "what is in this frame". This answers the
question the mission actually poses: *which of those things is the same thing,
moving?* A catalogue of a thousand detections across twelve visits is not more
useful than the pictures were until something says which rows belong together.

Two stages, and the first one does most of the work.

**Reject what stood still.** A star is at the same right ascension and
declination in every frame. A body in the outer solar system is not, by a great
deal: at 500 AU the apparent position swings 825 arcsec over six months, which
is 134 SPHEREx pixels, so between two visits forty days apart it has already
moved thirty times the match radius. Cross-matching every detection against
every other epoch and discarding anything that keeps turning up in the same
place removes the fixed sky -- 89% of the catalogue on the twelve-epoch NEP
field, 77% on the four-epoch ecliptic one -- and does it without any model of
what the moving thing is doing.

**Link what is left, against the geometry.** The survivors are movers, noise,
and things that were only bright once. A pair of them from two different epochs,
plus an assumed distance, is exactly enough to fix a trajectory: invert each
apparent position to a heliocentric direction at that distance, take the
difference as the drift rate, and then the object's position at every other
epoch is predicted with no free parameters left. Count how many surviving
detections land on those predictions. That is shift-and-stack run on a
catalogue instead of on pixels, which is cheap enough to do in a browser.

The hypothesis is then thrown away and the members handed to
:func:`tracklet.physics.ephemeris.solve_astrometry`, which fits all five
parameters properly and returns a distance with an error bar and a chi-squared.
The grid's job is only to *propose* -- it never sets the answer, so the coarse
spacing of the distance grid does not quantise the result.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from tracklet.frames.wcs import angular_separation
from tracklet.physics.ephemeris import (
    AstrometricSolution,
    Observation,
    apparent_place,
    helio_direction,
    orbital_rate_arcsec_per_year,
    solve_astrometry,
)


def sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle separation in arcsec. Haversine, so it holds near the pole."""
    return angular_separation(ra1, dec1, ra2, dec2) * 3600.0


@dataclass(frozen=True)
class SkyDetection:
    """One detection, placed on the sky by its frame's own WCS."""

    epoch: int
    mjd: float
    ra_deg: float
    dec_deg: float
    snr: float
    #: Stamp array coordinates it came from, carried through so a UI can draw it.
    x: float = 0.0
    y: float = 0.0


@dataclass
class Tracklet:
    """Detections from different epochs that one moving body would explain."""

    members: list[SkyDetection]
    #: The grid distance that proposed the link. Not the answer -- see ``solution``.
    hypothesis_au: float
    #: RMS of the members about the proposing hypothesis, arcsec.
    rms_arcsec: float
    solution: AstrometricSolution | None = None
    #: Set when the fit is untrustworthy, in words, for a UI to show.
    note: str = ""

    @property
    def n_epochs(self) -> int:
        return len({m.epoch for m in self.members})

    @property
    def key(self) -> frozenset:
        return frozenset((m.epoch, round(m.ra_deg, 7), round(m.dec_deg, 7)) for m in self.members)


@dataclass
class LinkReport:
    """Everything the search did, so a null result can still be quoted."""

    n_detections: int
    n_static: int
    n_moving: int
    #: How many of the movers the linker actually considered, after the
    #: per-epoch brightness budget. Reported because a silent truncation reads
    #: as "searched everything" when it did not.
    n_linked: int = 0
    #: Visits available, and how many a candidate had to appear in.
    n_epochs: int = 0
    required_epochs: int = 0
    #: Set when the search could not run at all, in words. Distinct from
    #: "searched and found nothing", which is a result.
    blocked: str = ""
    #: Beyond this distance a body at rest sits still in enough of these visits
    #: to be cut as a star, and the search is blind (:func:`blind_beyond`); 0
    #: when that holds at every distance the grid proposes.
    max_distance_au: float = 0.0
    tracklets: list[Tracklet] = field(default_factory=list)


class SkyCells:
    """A grid on the sky in which two points within ``tol_arcsec`` of each other
    always share a cell or sit in neighbouring ones -- anywhere on the sky.

    Rows are tol high in declination, which is exact at every declination.
    Columns cannot be tol wide everywhere: a degree of right ascension is
    cos(dec) degrees on the sky, so near a pole a separation of tol spans far
    more of it than at the equator. Each column is therefore as wide as tol can
    span at the most polar of ``decs`` pushed a further tol poleward (so a
    query that does match one of them is covered too), and the columns divide
    360 degrees exactly, so the one after 359.99 is the one at 0. Where tol
    reaches the pole it spans every right ascension, and there is one column.

    This replaced one cos(dec) taken at the field's median declination, which
    held for a baked stamp a third of a degree across and nowhere else: a field
    straddling right ascension 0 split its stars into columns a whole sky
    apart, so none of them was cut as fixed, and cells were narrower than tol
    on the polar side of the median.
    """

    def __init__(self, decs: list[float], tol_arcsec: float) -> None:
        tol = tol_arcsec / 3600.0
        far = max((abs(d) for d in decs), default=0.0)
        cosd = math.cos(math.radians(min(90.0, far + tol)))
        # sin^2(sep/2) >= cos(d1) cos(d2) sin^2(dRA/2), so sep <= tol bounds dRA.
        s = math.sin(math.radians(tol) / 2.0) / cosd
        span = math.degrees(2.0 * math.asin(s)) * (1.0 + 1e-9) if s < 1.0 else 360.0
        ncol = int(math.floor(360.0 / span))
        self.ncol = ncol if ncol >= 3 else 1
        self._width = 360.0 / self.ncol
        self._row_h = tol * (1.0 + 1e-9)

    def row(self, dec: float) -> int:
        return int(math.floor(dec / self._row_h))

    def col(self, ra: float) -> int:
        return int(math.floor(ra / self._width)) % self.ncol

    def around(self, c: int) -> tuple[int, ...]:
        if self.ncol == 1:
            return (0,)
        return ((c - 1) % self.ncol, c, (c + 1) % self.ncol)


def mark_static(
    detections: list[SkyDetection], *, tol_arcsec: float = 6.2, min_epochs: int = 3
) -> list[bool]:
    """``True`` for detections that keep appearing in the same place.

    ``min_epochs`` is how many *distinct* epochs must agree before a position is
    called fixed. Two is too few -- two detections agreeing once is what a
    slow-moving object near the turning point of its parallax ellipse looks
    like -- and the third epoch is what makes it a pattern.

    Bucketed on the sky rather than compared pairwise. The honest version is
    O(n^2), and on a real twelve-epoch scan n is a few thousand, which is
    several million great-circle separations and seconds of a user's attention.
    Two detections can only match if they fall in the same cell or a
    neighbouring one, so a grid of ``tol``-sized cells turns the sweep linear
    without changing a single answer -- anywhere on the sky: see
    :class:`SkyCells`.
    """
    n = len(detections)
    static = [False] * n
    if n == 0 or tol_arcsec <= 0:
        return static

    cells = SkyCells([d.dec_deg for d in detections], tol_arcsec)
    grid: dict[tuple[int, int], list[int]] = {}
    keys = []
    for i, d in enumerate(detections):
        k = (cells.row(d.dec_deg), cells.col(d.ra_deg))
        keys.append(k)
        grid.setdefault(k, []).append(i)

    for i, a in enumerate(detections):
        hits = {a.epoch}
        ky, kx = keys[i]
        for dy in (-1, 0, 1):
            for cx in cells.around(kx):
                for j in grid.get((ky + dy, cx), ()):
                    b = detections[j]
                    if j == i or b.epoch in hits:
                        continue
                    if sep_arcsec(a.ra_deg, a.dec_deg, b.ra_deg, b.dec_deg) <= tol_arcsec:
                        hits.add(b.epoch)
        if len(hits) >= min_epochs:
            static[i] = True
    return static


def distance_grid(lo_au: float = 30.0, hi_au: float = 2000.0, n: int = 40) -> list[float]:
    """Log-spaced, because parallax goes as ``1/d``.

    A linear grid would sample 1000-1500 AU as finely as 30-530 AU while the
    observable -- the size of the wobble -- changes by a factor of 17 across the
    first and by a third across the second.
    """
    if n < 2:
        return [lo_au]
    r = math.log(hi_au / lo_au) / (n - 1)
    return [lo_au * math.exp(r * k) for k in range(n)]


#: Fewest visits a candidate may be built from, whatever the field.
#:
#: Measured, not chosen, and the measurement is the *floor's own* case: the
#: four-visit ecliptic-plane field, where :func:`required_epochs` returns
#: exactly this. That field holds no known moving object, so every candidate it
#: yields is wrong -- three visits gives 27 of them and four gives none.
#:
#: This note used to add "four returns none on either", with the pole field as
#: the other one, and that is not what the floor does there: twelve visits ask
#: for six, and it takes six. At four the pole field still returns 53, and the
#: fall is 773 / 53 / 6 / 0 across three, four, five and six.
#: ``scripts/measure_search.py --sweep`` prints it for any field.
MIN_EPOCHS_FLOOR = 4

#: Fewest visits a candidate may be built from on a field of six or more.
#:
#: Measured on the fields re-baked on 2026-10-09 with every frame the search
#: can use, none of which shows a known distant mover, so every candidate is
#: false. Against four, five and six required, at 6.5 sigma: South Ecliptic
#: Pole (25 visits) 12 / 1 / 0, latitude +60 (10) 51 / 1 / 0, mid-latitude +45
#: (6) 3 / 0 / 0, +30 (6) 2 / 0 / 0, ecliptic 150 (6) 1 / 0 / 0, ecliptic 180
#: (6) 2 / 0 / 0 -- and the twelve-visit North Ecliptic Pole's 53 / 6 / 0
#: above. Half the visits alone asks for four or five of these, which is
#: where the false candidates are; six is where every field reaches zero.
#: Fields of five or fewer keep the floor of four, which is what the
#: anticentre's blind search ran with.
ROOMY_FLOOR = 6


def required_epochs(n_epochs: int) -> int:
    """How many visits a candidate must appear in, for a field with this many.

    Half the visits, never fewer than :data:`MIN_EPOCHS_FLOOR`, nor fewer than
    :data:`ROOMY_FLOOR` once there are that many visits. The fraction is what
    carries the trials factor: more visits means more detections, which means
    more combinations to be fooled by. Both floors are measured; see them.
    """
    floor = ROOMY_FLOOR if n_epochs >= ROOMY_FLOOR else MIN_EPOCHS_FLOOR
    return max(floor, -(-n_epochs // 2))


def find_tracklets(
    detections: list[SkyDetection],
    *,
    match_arcsec: float = 12.0,
    static_tol_arcsec: float = 6.2,
    min_epochs: int | None = None,
    static_min_epochs: int = 3,
    distances: list[float] | None = None,
    max_per_epoch: int = 40,
    min_baseline_fraction: float = 0.25,
    drift_tolerance: float = 3.0,
    sigma_arcsec: float = 2.0,
    seed_slack: float = 4.0,
    max_reduced_chi2: float = 4.0,
    min_significance: float = 5.0,
) -> LinkReport:
    """Reject the fixed sky, then link the rest into candidate trajectories.

    ``min_epochs`` defaults to ``None``, which means *derive it from the field*:
    ``max(4, ceil(n_epochs / 2))``. That is one rule and it reproduces both
    measurements -- twelve epochs at the pole gives six, four in the ecliptic
    plane gives four, and both return zero false candidates on sky known to
    hold nothing. A fixed six was worse than wrong on the four-epoch field: it
    made a candidate *structurally impossible*, so the search returned an empty
    list having never run, and the panel reported that as "nothing found here",
    which is a different claim entirely.

    A candidate whose reduced chi-squared is 12 is not "a detection with a
    caveat" -- it is the statement that these detections do not lie on one
    trajectory, which is the definition of not being an object. Annotating
    those and listing them anyway produced 163 candidates on a field with
    nothing in it, which is indistinguishable from no search at all. They are
    dropped. What survives still carries a note when it is marginal.
    Order matters here and it used to be wrong. Trimming to the brightest
    ``max_per_epoch`` *before* the static cut spends the whole budget on stars:
    the fixed sky is brighter than anything moving, so on the real NEP stamp
    the top 40 of 288 detections per epoch were 40 stars, and a planted source
    at 10 sigma never reached the linker at all. Reject the fixed sky first --
    it is O(n^2) but it is a few million cheap comparisons -- and spend the
    budget on what is left.

    ``min_epochs`` is derived from the field unless you pass one; see
    :func:`required_epochs`. Whatever it works out to, it is more than the three
    the geometry needs, and the extra is not conservatism -- it is the trials
    factor. A
    five-parameter model fitted to three positions has one degree of freedom, so
    very nearly any three detections fit *something*, and the search tries every
    triple: with 45 leftover detections and a 40-point distance grid that is
    over half a million hypotheses. Run on the real NEP field at three epochs
    this returns two candidates at 389 and 358 AU, with parallax significances
    of 143 and 43 sigma. Both are noise. A human clicking three times has a
    trials factor of one and three epochs is genuinely enough for them; an
    automatic search over every triple does not get the same threshold, and
    quoting one is how a 143-sigma nothing ends up in a press release.

    Six is **measured, not chosen**. Run against the real North Ecliptic Pole
    field -- which contains no known moving object, so every candidate it
    returns is by definition false -- at 6.5 sigma:

        min_epochs   false candidates
             4              50
             5               1
             6               0
             7               0

    and a source planted at 12 sigma is recovered in ten of the twelve epochs,
    so it clears the cut with room to spare. Setting the threshold from the
    false-alarm rate of a field known to be empty is the same discipline the
    rest of the project uses: a search quotes the trials factor it paid.
    """
    grid = distances if distances is not None else distance_grid()
    n_epochs = len({d.epoch for d in detections})
    required = required_epochs(n_epochs) if min_epochs is None else min_epochs

    # Deliberately not `min_epochs`: these count different things. Three
    # sightings in one place is already a pattern and calling it a star is
    # cheap; four detections on one trajectory is a claim and costs more.
    static = mark_static(detections, tol_arcsec=static_tol_arcsec,
                         min_epochs=static_min_epochs)
    survivors = [d for d, s in zip(detections, static) if not s]

    by_epoch: dict[int, list[SkyDetection]] = {}
    for d in survivors:
        by_epoch.setdefault(d.epoch, []).append(d)
    moving: list[SkyDetection] = []
    for ep in sorted(by_epoch):
        moving.extend(sorted(by_epoch[ep], key=lambda s: -s.snr)[:max_per_epoch])

    mjds = sorted({d.mjd for d in detections})
    baseline = (mjds[-1] - mjds[0]) if len(mjds) > 1 else 0.0
    report = LinkReport(
        n_detections=len(detections),
        n_static=sum(static),
        n_moving=len(survivors),
        n_linked=len(moving),
        n_epochs=n_epochs,
        required_epochs=required,
        max_distance_au=blind_beyond(detections, tol_arcsec=static_tol_arcsec,
                                     static_min_epochs=static_min_epochs,
                                     required=required, near_au=min(grid)),
    )
    # A field with fewer visits than the cut needs cannot produce a candidate at
    # all, and "no candidates" would then be a statement about arithmetic rather
    # than about the sky. Say which it is.
    if n_epochs < required:
        report.blocked = (
            f"{n_epochs} visits is too few to search: a candidate needs "
            f"{required}, and below that the trials factor swamps any result")
        return report
    # Enough visits, made too close together: every body the grid could propose
    # sits still in enough of them to be cut as a star. Same distinction.
    if report.max_distance_au == 0.0:
        report.blocked = (
            f"{n_epochs} visits on {_nights(mjds)} nights are too close in time to "
            f"search: from {min(grid):.0f} AU out, a body sits still in enough of them "
            f"to be cut as a star, leaving fewer than the {required} a candidate needs")
        return report
    if len(moving) < required or baseline <= 0.0:
        return report

    epochs = sorted({d.epoch for d in moving})
    if len(epochs) < required:
        return report

    per_epoch: dict[int, list[SkyDetection]] = {}
    for d in moving:
        per_epoch.setdefault(d.epoch, []).append(d)
    epoch_mjd = {d.epoch: d.mjd for d in moving}

    # Note on the JavaScript mirror: `web/js/link.js` hoists the heliocentric
    # direction table out of this loop and buckets each epoch's detections on
    # the sky before matching. Both are exact -- the golden vectors check that
    # -- and both are worth about a factor of three in a browser, where someone
    # is watching. Neither is mirrored here, because this side is the reference
    # implementation and the reference should read as the algorithm rather than
    # as the algorithm plus two indexes.
    #
    # Seeding is deliberately loose. A grid node is at best a few percent off
    # the true distance, and a few percent of a 400 arcsec parallax is tens of
    # arcsec -- so a seed radius equal to the final match radius would link only
    # the epochs where the grid error happened to be small, and report a real
    # object as three disconnected fragments. Seed wide, then tighten.
    seed_arcsec = match_arcsec * seed_slack
    best: dict[frozenset, Tracklet] = {}
    min_gap = baseline * min_baseline_fraction
    for ia, ea in enumerate(epochs):
        for eb in epochs[ia + 1:]:
            if abs(epoch_mjd[eb] - epoch_mjd[ea]) < min_gap:
                continue
            for a in per_epoch[ea]:
                for b in per_epoch[eb]:
                    for d_au in grid:
                        t = _try_pair(a, b, d_au, per_epoch, epoch_mjd,
                                      seed_arcsec, required, drift_tolerance)
                        if t is None:
                            continue
                        prev = best.get(t.key)
                        if prev is None or t.rms_arcsec < prev.rms_arcsec:
                            best[t.key] = t

    grown: list[Tracklet] = []
    for t in best.values():
        g = _grow(t, per_epoch, epoch_mjd, match_arcsec, sigma_arcsec, required)
        if g is not None:
            grown.append(g)

    credible = [t for t in grown if _credible(t, max_reduced_chi2, min_significance)]
    out = _dedupe(credible)
    out.sort(key=lambda t: (-t.n_epochs, t.rms_arcsec))
    report.tracklets = out
    return report


def _grow(t, per_epoch, epoch_mjd, match_arcsec, sigma_arcsec, min_epochs):
    """Fit, re-predict from the fit, re-match. Repeat until the members settle.

    This is where the distance grid stops mattering. The seed only has to be
    close enough to associate three detections; from there the five-parameter
    solution predicts every epoch far better than any grid node can, so each
    pass pulls in epochs the coarse hypothesis missed and drops the ones it
    picked up by luck. Converges in two or three passes or not at all.
    """
    for _ in range(4):
        _fit(t, sigma_arcsec)
        if t.solution is None:
            return None
        members = []
        resid2 = 0.0
        for ep, rows in per_epoch.items():
            ra, dec = predict(t.solution, epoch_mjd[ep])
            hit, hit_sep = None, match_arcsec
            for s in rows:
                sp = sep_arcsec(ra, dec, s.ra_deg, s.dec_deg)
                if sp <= hit_sep:
                    hit, hit_sep = s, sp
            if hit is not None:
                members.append(hit)
                resid2 += hit_sep * hit_sep
        if len({m.epoch for m in members}) < min_epochs:
            return None
        settled = {(m.epoch, m.ra_deg, m.dec_deg) for m in members} == \
                  {(m.epoch, m.ra_deg, m.dec_deg) for m in t.members}
        t = Tracklet(members, t.hypothesis_au, math.sqrt(resid2 / len(members)))
        if settled:
            break
    _fit(t, sigma_arcsec)
    return t if t.solution is not None else None


def predict(sol: AstrometricSolution, mjd: float) -> tuple[float, float]:
    """Where the fitted solution says the body is, at ``mjd``.

    :meth:`AstrometricSolution.place`, the model the fit minimised and the one
    the console draws as the white curve, so the linker and the overlay cannot
    disagree about what the fit means. This used to drift ``ra0`` linearly in RA
    instead, a small circle the fit never assumed, which for a fast body put
    the prediction tens of arcsec off the fit it came from.
    """
    _, _, ra, dec = sol.place(mjd)
    return ra, dec


def _try_pair(a, b, d_au, per_epoch, epoch_mjd, match_arcsec, min_epochs, drift_tolerance):
    """One (pair, distance) hypothesis. ``None`` if it does not hold up."""
    try:
        ra1, dec1 = helio_direction(a.ra_deg, a.dec_deg, d_au, a.mjd)
        ra2, dec2 = helio_direction(b.ra_deg, b.dec_deg, d_au, b.mjd)
    except ValueError:
        return None
    dt_yr = (b.mjd - a.mjd) / 365.256363004
    if dt_yr == 0.0:
        return None

    # A physical prune, and a strong one. A bound body at d AU on a circular
    # orbit sweeps a known rate; three times that covers eccentric and
    # retrograde cases with room to spare, and anything faster is not an object
    # at this distance -- it is two unrelated detections and a wrong guess.
    sep = sep_arcsec(ra1, dec1, ra2, dec2)
    if sep / abs(dt_yr) > drift_tolerance * orbital_rate_arcsec_per_year(d_au):
        return None

    dra = ((ra2 - ra1 + 540.0) % 360.0 - 180.0) / dt_yr
    ddec = (dec2 - dec1) / dt_yr

    members = []
    resid2 = 0.0
    for ep, rows in per_epoch.items():
        yr = (epoch_mjd[ep] - a.mjd) / 365.256363004
        ap = apparent_place(ra1 + dra * yr, dec1 + ddec * yr, d_au, epoch_mjd[ep])
        hit = None
        hit_sep = match_arcsec
        for s in rows:
            sp = sep_arcsec(ap.ra_deg, ap.dec_deg, s.ra_deg, s.dec_deg)
            if sp <= hit_sep:
                hit, hit_sep = s, sp
        if hit is not None:
            members.append(hit)
            resid2 += hit_sep * hit_sep
    if len({m.epoch for m in members}) < min_epochs:
        return None
    return Tracklet(members, d_au, math.sqrt(resid2 / len(members)))


def _credible(t: Tracklet, max_chi2: float, min_sig: float) -> bool:
    """Whether this is a candidate or just an arithmetic coincidence."""
    s = t.solution
    if s is None or not s.separable:
        return False
    if not (s.reduced_chi2 == s.reduced_chi2) or s.reduced_chi2 > max_chi2:
        return False
    return s.significance >= min_sig and s.distance_au > 0


def _dedupe(tracklets: list[Tracklet], *, share: int = 2) -> list[Tracklet]:
    """One object, one candidate.

    Every pair of a body's own detections proposes that body, so a real object
    arrives here dozens of times over. A subset test alone is not enough --
    after refinement two proposals of the same object can differ by one member
    each, so neither contains the other, and the object is reported twice with
    almost the same distance. Sharing ``share`` detections with something
    already kept is the test that actually means "this is the same object",
    because a detection belongs to one body.
    """
    tracklets.sort(key=lambda t: (-len(t.members), t.rms_arcsec))
    kept: list[Tracklet] = []
    for t in tracklets:
        ids = {(m.epoch, m.ra_deg, m.dec_deg) for m in t.members}
        if any(len(ids & {(m.epoch, m.ra_deg, m.dec_deg) for m in k.members}) >= share
               for k in kept):
            continue
        kept.append(t)
    return kept


def _fit(t: Tracklet, sigma_arcsec: float) -> None:
    """Hand the members to the real solver and record what it says."""
    obs = [Observation(m.mjd, m.ra_deg, m.dec_deg, sigma_arcsec) for m in
           sorted(t.members, key=lambda m: m.mjd)]
    try:
        sol = solve_astrometry(obs)
    except Exception as exc:  # noqa: BLE001 - a bad candidate must not kill the search
        t.note = f"fit failed: {exc}"
        return
    t.solution = sol
    if not sol.separable:
        t.note = "parallax and drift are not separable at these epochs"
    elif sol.reduced_chi2 > 10.0:
        t.note = "poor fit — these detections are probably not one object"
    elif sol.significance < 3.0:
        t.note = "parallax not significant; distance is a lower bound at best"


def blind_beyond(
    detections: list[SkyDetection],
    *,
    tol_arcsec: float = 6.2,
    static_min_epochs: int = 3,
    required: int = 4,
    near_au: float = 30.0,
) -> float:
    """Distance beyond which the static cut removes a body as a star, for these visits.

    The cut removes a detection that recurs within ``tol`` of itself in
    ``static_min_epochs`` epochs. A body with no motion of its own moves only by
    parallax, which shrinks as 1/d, so past some distance enough of its
    detections sit still to leave fewer than ``required`` -- and a candidate
    needs that many. Where that happens depends on *when* the visits were
    made, not only on how far the parallax reaches: twelve visits forty days
    apart keep a body moving between each pair out to thousands of AU, while
    six visits made on three nights, three of them within three minutes, cut
    every body from 30 AU outward as a star.

    This used to be ``206265 / tol`` -- where the parallax ellipse's semi-axis
    equals ``tol`` -- whatever the visits. That is neither: two points on an
    ellipse that size can still be twice ``tol`` apart, and visits bunched in
    time never sample the ellipse at all.

    Computed on the visits' own epochs at the detections' mean position, for a
    body at rest (parallax only; orbital motion only helps a body escape the
    cut), by bisection on the distance -- the count that survives only falls as
    the parallax shrinks. Returns 0 when not even ``near_au`` survives, and
    ``inf`` when the cut can never gather ``static_min_epochs`` sightings.
    """
    if tol_arcsec <= 0.0:
        return math.inf
    mjd_of: dict[int, float] = {}
    sx = sy = sz = 0.0
    for d in detections:
        mjd_of.setdefault(d.epoch, d.mjd)
        ra, dec = math.radians(d.ra_deg), math.radians(d.dec_deg)
        sx += math.cos(dec) * math.cos(ra)
        sy += math.cos(dec) * math.sin(ra)
        sz += math.sin(dec)
    mjds = [mjd_of[e] for e in sorted(mjd_of)]
    if len(mjds) < static_min_epochs:
        return math.inf
    if math.hypot(sx, sy, sz) == 0.0:
        return math.inf
    ra0 = math.degrees(math.atan2(sy, sx)) % 360.0
    dec0 = math.degrees(math.atan2(sz, math.hypot(sx, sy)))

    def kept(d_au: float) -> int:
        places = [apparent_place(ra0, dec0, d_au, m) for m in mjds]
        n = 0
        for a in places:
            near = sum(1 for b in places
                       if sep_arcsec(a.ra_deg, a.dec_deg, b.ra_deg, b.dec_deg) <= tol_arcsec)
            if near < static_min_epochs:
                n += 1
        return n

    lo = near_au
    # Where the whole ellipse fits inside tol, every sighting has every other
    # within reach: nothing survives past this.
    hi = 2.0 * 206264.806247 / tol_arcsec
    if kept(lo) < required:
        return 0.0
    if kept(hi) >= required:
        return math.inf
    while hi / lo > 1.0 + 1e-10:
        mid = math.sqrt(lo * hi)
        if kept(mid) >= required:
            lo = mid
        else:
            hi = mid
    return lo


def _nights(mjds: list[float]) -> int:
    """How many separate nights ``mjds`` fall on: runs split by a day's gap."""
    ts = sorted(mjds)
    return 1 + sum(1 for a, b in itertools.pairwise(ts) if b - a > 1.0) if ts else 0