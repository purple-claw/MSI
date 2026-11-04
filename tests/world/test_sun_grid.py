from datetime import date

from gaia.world.cities import CITIES
from gaia.world.calendar import local_midnight_utc_ms
from gaia.world.sun import sun_altitude_deg, sun_altitude_grid


def test_grid_matches_astral_across_a_year():
    city = CITIES["hyderabad"]
    start = local_midnight_utc_ms(date(2026, 1, 1), city.tz_offset_min)
    grid = sun_altitude_grid(city, start, 365 * 1440)
    assert len(grid) == 365 * 1440
    for m in range(0, len(grid), 1013):  # sampled minutes across every season
        assert abs(grid[m] - sun_altitude_deg(city, start + m * 60_000)) < 1e-6, m


def test_grid_matches_astral_at_high_latitude_and_other_zone():
    city = CITIES["delhi"]
    start = local_midnight_utc_ms(date(2026, 6, 1), city.tz_offset_min)
    grid = sun_altitude_grid(city, start, 10 * 1440)
    for m in range(0, len(grid), 97):
        assert abs(grid[m] - sun_altitude_deg(city, start + m * 60_000)) < 1e-6, m
