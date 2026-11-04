"""Heat index per city and day: 0-1, how hot the season is. Drives the AC, fan and shading policies.

The table holds one value per calendar month, read at mid-month. Days in between get a linear
interpolation between the two nearest mid-months, and the table wraps from December to January.
Pattern: highest April to June, high July to September, mid October, low December and January.
Delhi has the strongest swing, Bengaluru the flattest, and the coastal cities stay warm all year.

assumption: every value is hand-set (source: assumption). G2b replaces the table with climate normals.
"""

import calendar
import math
from datetime import date

# assumption: monthly heat index at mid-month, January first.
_TABLE: dict[str, tuple[float, ...]] = {
    "delhi":     (0.10, 0.15, 0.45, 0.80, 1.00, 1.00, 0.75, 0.70, 0.65, 0.40, 0.20, 0.10),
    "bengaluru": (0.30, 0.35, 0.50, 0.55, 0.60, 0.55, 0.45, 0.45, 0.50, 0.50, 0.40, 0.30),
    "chennai":   (0.60, 0.60, 0.70, 0.85, 0.95, 0.95, 0.90, 0.90, 0.90, 0.85, 0.75, 0.65),
    "mumbai":    (0.55, 0.55, 0.65, 0.80, 0.90, 0.85, 0.75, 0.75, 0.80, 0.85, 0.75, 0.60),
    "hyderabad": (0.25, 0.40, 0.70, 0.90, 0.95, 0.80, 0.65, 0.60, 0.60, 0.50, 0.35, 0.25),
    "kolkata":   (0.40, 0.40, 0.60, 0.75, 0.80, 0.75, 0.70, 0.70, 0.70, 0.60, 0.45, 0.40),
    "pune":      (0.30, 0.35, 0.60, 0.80, 0.90, 0.70, 0.55, 0.50, 0.55, 0.55, 0.40, 0.30),
}


def heat_index(city: str, d: date) -> float:
    """Heat index in [0, 1] for ``city`` on local date ``d``. Raises KeyError for an unknown city."""
    try:
        values = _TABLE[city]
    except KeyError:
        raise KeyError(f"no heat index table for city: {city!r}") from None
    days = calendar.monthrange(d.year, d.month)[1]
    # Position in months: each month's value sits at its middle, days in between interpolate.
    pos = (d.month - 1) + (d.day - 0.5) / days - 0.5
    lower = math.floor(pos)
    frac = pos - lower
    a = values[lower % 12]
    b = values[(lower + 1) % 12]
    return a + (b - a) * frac
