"""Sun elevation and clear-sky outdoor illuminance.

Weather is not modelled here (that comes in G2), so lux is clear-sky only.
"""

import math
from datetime import datetime, timezone

import numpy as np
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


def sun_altitude_grid(city: City, start_ms: int, minutes: int) -> list[float]:
    """Solar elevation at ``city`` for each whole UTC minute from ``start_ms``.

    A numpy port of astral's ``elevation`` (NOAA solar position, refraction included), so one
    call replaces ``minutes`` calls to ``sun_altitude_deg``. Agreement with astral is pinned by
    tests/world/test_sun.py.
    """
    ms = start_ms + np.arange(minutes, dtype=np.float64) * 60_000.0
    jd = ms / 86_400_000.0 + 2440587.5
    t = (jd - 2451545.0) / 36525.0
    rad = np.radians

    l0 = np.mod(280.46646 + t * (36000.76983 + 0.0003032 * t), 360.0)
    mean_anom = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    ecc = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    mr = rad(mean_anom)
    centre = (np.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t))
              + np.sin(2 * mr) * (0.019993 - 0.000101 * t) + np.sin(3 * mr) * 0.000289)
    omega = 125.04 - 1934.136 * t
    apparent = l0 + centre - 0.00569 - 0.00478 * np.sin(rad(omega))
    seconds = 21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))
    obliquity = 23.0 + (26.0 + seconds / 60.0) / 60.0 + 0.00256 * np.cos(rad(omega))
    declination = np.degrees(np.arcsin(np.sin(rad(obliquity)) * np.sin(rad(apparent))))

    y = np.tan(rad(obliquity) / 2.0) ** 2
    lr = rad(l0)
    etime = (y * np.sin(2 * lr) - 2 * ecc * np.sin(mr)
             + 4 * ecc * y * np.sin(mr) * np.cos(2 * lr)
             - 0.5 * y * y * np.sin(4 * lr) - 1.25 * ecc * ecc * np.sin(2 * mr))
    eqtime = np.degrees(etime) * 4.0

    minute_of_day = np.mod(ms // 60_000.0, 1440.0)  # UTC; ms are whole minutes
    true_solar = np.mod(minute_of_day + eqtime + 4.0 * city.lon, 1440.0)
    hour_angle = true_solar / 4.0 - 180.0
    hour_angle = np.where(hour_angle < -180, hour_angle + 360.0, hour_angle)

    lat = min(max(city.lat, -89.8), 89.8)
    csz = np.clip(np.cos(rad(lat)) * np.cos(rad(declination)) * np.cos(rad(hour_angle))
                  + np.sin(rad(lat)) * np.sin(rad(declination)), -1.0, 1.0)
    zenith = np.degrees(np.arccos(csz))

    elev = 90.0 - zenith
    with np.errstate(divide="ignore", invalid="ignore"):
        te = np.tan(rad(elev))
        high = 58.1 / te - 0.07 / te ** 3 + 0.000086 / te ** 5
        mid = 1735.0 + elev * (-518.2 + elev * (103.4 + elev * (-12.79 + elev * 0.711)))
        low = -20.774 / te
    refraction = np.where(elev >= 85.0, 0.0,
                          np.where(elev > 5.0, high, np.where(elev > -0.575, mid, low))) / 3600.0
    return (elev + refraction).tolist()
