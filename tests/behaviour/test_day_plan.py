from dataclasses import replace

from hypothesis import given, settings
from hypothesis import strategies as st

from gaia.behaviour.day_plan import plan_day
from gaia.household.personas import PERSONAS
from gaia.rng import rng_for
from gaia.world.calendar import DayType

PERSONA_NAMES = sorted(PERSONAS)


@settings(max_examples=300)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    persona_name=st.sampled_from(PERSONA_NAMES),
    day_type=st.sampled_from(list(DayType)),
)
def test_plan_contiguous_and_ordered(seed, persona_name, day_type):
    acts = plan_day(PERSONAS[persona_name], day_type, rng_for(seed, "day"))
    assert acts[0].start_min == 0
    assert acts[0].name == "sleep"
    assert acts[-1].name == "sleep"
    for a, b in zip(acts, acts[1:]):
        assert a.end_min == b.start_min, (a, b)
    for a in acts:
        assert a.end_min - a.start_min >= 5, a


def test_plan_deterministic_for_same_rng_state():
    for persona_name in PERSONA_NAMES:
        for day_type in DayType:
            a = plan_day(PERSONAS[persona_name], day_type, rng_for(42, "day"))
            b = plan_day(PERSONAS[persona_name], day_type, rng_for(42, "day"))
            assert a == b


def test_sleep_after_midnight_wraps():
    for base in ("working_professional", "homemaker"):
        persona = replace(PERSONAS[base], sleep_min=(1470, 0))
        for day_type in DayType:
            acts = plan_day(persona, day_type, rng_for(5, "day"))
            last = acts[-1]
            assert last.name == "sleep"
            assert last.start_min == 1470
            assert last.room == "own_bedroom"
            assert last.end_min > 1440
            assert all(a.end_min >= a.start_min for a in acts)


def test_professional_workday_has_away():
    persona = PERSONAS["working_professional"]
    acts = plan_day(persona, DayType.workday, rng_for(9, "day"))
    away = [a for a in acts if a.room is None]
    assert len(away) == 1
    assert away[0].name == "away"


def test_weekend_has_no_away():
    persona = PERSONAS["working_professional"]
    for day_type in (DayType.weekend, DayType.festival):
        acts = plan_day(persona, day_type, rng_for(9, "day"))
        assert all(a.room is not None for a in acts)


def test_homemaker_has_no_away():
    for day_type in DayType:
        acts = plan_day(PERSONAS["homemaker"], day_type, rng_for(9, "day"))
        assert all(a.room is not None for a in acts)


@settings(max_examples=300)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    day_type=st.sampled_from(list(DayType)),
)
def test_homemaker_has_lunch_in_living_and_short_cooks(seed, day_type):
    acts = plan_day(PERSONAS["homemaker"], day_type, rng_for(seed, "day"))
    lunches = [a for a in acts if a.name == "lunch"]
    assert len(lunches) == 1
    assert lunches[0].room == "living"
    for a in acts:
        if a.name == "cook":
            assert a.end_min - a.start_min <= 60 + 5, a
    rest = next(a for a in acts if a.name == "rest")
    assert rest.start_min == 840
