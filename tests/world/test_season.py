from datetime import date, timedelta

import pytest

from gaia.world.cities import CITIES
from gaia.world.season import heat_index


def _year(city):
    d = date(2026, 1, 1)
    return [heat_index(city, d + timedelta(days=k)) for k in range(365)]


def test_every_city_has_a_table():
    for city in CITIES:
        values = _year(city)
        assert all(0.0 <= v <= 1.0 for v in values), city


def test_delhi_has_the_strongest_swing():
    delhi = _year("delhi")
    assert heat_index("delhi", date(2026, 1, 15)) < 0.2
    assert heat_index("delhi", date(2026, 5, 20)) > 0.95
    assert heat_index("delhi", date(2026, 6, 10)) > 0.95
    swing = {c: max(_year(c)) - min(_year(c)) for c in CITIES}
    assert swing["delhi"] == max(swing.values())
    assert delhi


def test_bengaluru_is_flat():
    values = _year("bengaluru")
    assert 0.25 <= min(values) and max(values) <= 0.65


def test_coastal_cities_are_warm_year_round():
    for city in ("chennai", "mumbai"):
        values = _year(city)
        assert 0.5 <= min(values) and max(values) <= 0.95, city


def test_interpolation_is_continuous():
    values = _year("delhi")
    steps = [abs(b - a) for a, b in zip(values, values[1:] + values[:1])]
    assert max(steps) < 0.05


def test_unknown_city_raises():
    with pytest.raises(KeyError, match="atlantis"):
        heat_index("atlantis", date(2026, 5, 1))
