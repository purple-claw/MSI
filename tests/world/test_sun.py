from datetime import datetime, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from gaia.world.cities import hyderabad
from gaia.world.sun import outdoor_lux, sun_altitude_deg


def _utc_ms(*args) -> int:
    dt = datetime(*args, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def test_noon_bright_midnight_dark():
    noon = sun_altitude_deg(hyderabad, _utc_ms(2026, 6, 21, 6, 30))  # 12:00 IST
    midnight = sun_altitude_deg(hyderabad, _utc_ms(2026, 6, 21, 18, 30))  # 00:00 IST
    assert outdoor_lux(noon) > 50_000
    assert outdoor_lux(midnight) == 0


@settings(max_examples=200)
@given(
    a=st.floats(min_value=-10, max_value=90, allow_nan=False),
    b=st.floats(min_value=-10, max_value=90, allow_nan=False),
)
def test_lux_monotonic_in_altitude(a, b):
    lo, hi = min(a, b), max(a, b)
    assert outdoor_lux(lo) <= outdoor_lux(hi)


def test_lux_continuous_at_boundaries():
    assert outdoor_lux(-6) == 0
    assert outdoor_lux(0) == 400
