from datetime import date, datetime, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from gaia.world.calendar import DayType, day_type, local_date, local_midnight_utc_ms

IST = 330


def test_ist_day_boundary():
    midnight = local_midnight_utc_ms(date(2026, 11, 4), IST)
    assert midnight == int(datetime(2026, 11, 3, 18, 30, tzinfo=timezone.utc).timestamp() * 1000)
    assert local_date(midnight, IST) == date(2026, 11, 4)
    assert local_date(midnight - 1, IST) == date(2026, 11, 3)


@settings(max_examples=200)
@given(
    d=st.dates(min_value=date(2000, 1, 1), max_value=date(2100, 12, 31)),
    tz=st.integers(min_value=-720, max_value=840),
)
def test_midnight_round_trip(d, tz):
    midnight = local_midnight_utc_ms(d, tz)
    assert local_date(midnight, tz) == d
    assert local_date(midnight - 1, tz) == date.fromordinal(d.toordinal() - 1)


def test_day_type():
    assert day_type(date(2026, 11, 7), frozenset()) == DayType.weekend  # Saturday
    assert day_type(date(2026, 11, 8), frozenset()) == DayType.weekend  # Sunday
    assert day_type(date(2026, 11, 4), frozenset()) == DayType.workday  # Wednesday


def test_festival_wins():
    wed = date(2026, 11, 4)
    assert day_type(wed, frozenset({wed})) == DayType.festival
    sat = date(2026, 11, 7)
    assert day_type(sat, frozenset({sat})) == DayType.festival
