"""From injection counts to an exclusion at a stated confidence level.

The depth map says, cell by cell, that ``k`` of ``n`` planted bodies came back.
An exclusion needs something slightly different: the chance that a *real* body
in that cell would have come back. They are not the same number, and the gap
is largest exactly where an exclusion is claimed. A cell where all 48 bodies
were recovered has not shown that the search never misses; 48 trials cannot
tell 100% from 98%.

The chance the next body planted in the cell is recovered, given ``k`` of ``n``,
is ``(k + 1) / (n + 2)``: Laplace's rule of succession, the posterior mean of
the completeness under a uniform prior. It is also the right quantity for the
question asked, since a real body is one more draw from the same cell.
Averaging the chance of missing over everything the injections leave
possible gives ``1 - (k + 1) / (n + 2)``. So:

    a body in a cell is excluded at confidence CL  iff
    the search found nothing there, and (k + 1) / (n + 2) >= CL

With no candidate in the cell (``n_obs = 0``), CLs reduces to the same
inequality, so this is the standard null-result limit rather than a new one.

With 48 bodies per cell, 95% needs 47 recovered and 90% needs 44. The
correction is affine in ``k / n`` and fixes one half, so a "recovered half the
time" reading is unchanged by it.

``web/js/depth.js`` mirrors these functions, and ``tests/js/run-golden.mjs``
holds the two to each other.
"""

from __future__ import annotations

import math


def recovery_chance(k: int, n: int) -> float:
    """The chance a body in a cell is recovered, from ``k`` of ``n`` planted."""
    if n < 0 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    return (k + 1) / (n + 2)


def needed(n: int, cl: float) -> int | None:
    """Fewest recoveries out of ``n`` that exclude at confidence ``cl``.

    None when even ``n`` of ``n`` is not enough. That is not a rounding
    detail: a cell of fewer than ``(2 * cl - 1) / (1 - cl)`` bodies cannot
    support an exclusion at all, which is 18 bodies for 95% and 8 for 90%.
    """
    if not 0 < cl < 1:
        raise ValueError(f"cl must be in (0, 1), got {cl}")
    # The tolerance absorbs the product's last bit: 0.9 * 50 need not come out
    # as exactly 45. No level in use puts the true value a hair above an integer.
    k = max(0, math.ceil(cl * (n + 2) - 1 - 1e-9))
    return k if k <= n else None


def excluded(k: int, n: int, cl: float) -> bool:
    """Whether ``k`` of ``n`` recovered excludes a body there at ``cl``."""
    return recovery_chance(k, n) >= cl
