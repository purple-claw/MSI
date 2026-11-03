"""Sun elevation and clear-sky outdoor illuminance.

Weather is not modelled here (that comes in G2), so lux is clear-sky only.
"""

import math
from datetime import datetime, timezone

from astral import Observer
from astral.sun import elevation

from gaia.world.cities import City


def sun_altitude_deg(city: City, ts_utc_ms: int) -> float:
    """Return the solar elevation in degrees above the horizon at ``city``."""
    observer = Observer(city.lat, city.lon)
    dt = datetime.fromtimestamp(ts_utc_ms / 1000, tz=timezone.utc)
    return float(elevation(observer, dt))


def outdoor_lux(altitude_deg: float) -> float:
    """Clear-sky outdoor illuminance (lux) for a solar altitude.

    0 below -6 deg (civil twilight end), a linear ramp to 400 lux across
    -6..0 deg, then 400 + 110000*sin(alt) above the horizon. The pieces meet
    at 0 and 400 lux, so the function is continuous and non-decreasing.
    """
    if altitude_deg < -6:
        return 0.0
    if altitude_deg < 0:
        return 400.0 * (altitude_deg + 6) / 6
    return 400.0 + 110_000.0 * math.sin(math.radians(altitude_deg))
