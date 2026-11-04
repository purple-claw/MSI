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
def test_toilet_visits_inserted(seed, persona_name, day_type):
    acts = plan_day(PERSONAS[persona_name], day_type, rng_for(seed, "day"))
    visits = [a for a in acts if a.name == "toilet"]
    assert 2 <= len(visits) <= 5, visits
    for v in visits:
        assert v.room == "own_bathroom"
        assert 5 <= v.end_min - v.start_min <= 6, v


@settings(max_examples=300)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    persona_name=st.sampled_from(PERSONA_NAMES),
    day_type=st.sampled_from(list(DayType)),
)
def test_toilet_visits_sit_in_awake_home_blocks(seed, persona_name, day_type):
    acts = plan_day(PERSONAS[persona_name], day_type, rng_for(seed, "day"))
    for i, a in enumerate(acts):
        if a.name != "toilet":
            continue
        neighbours = [acts[j] for j in (i - 1, i + 1) if 0 <= j < len(acts)]
        assert all(n.name != "sleep" for n in neighbours), (acts, i)  # sleeps are never split or bordered
        assert all(n.name != "toilet" for n in neighbours)
        assert a.start_min > 0 and a.end_min < acts[-1].start_min


def test_toilet_visits_leave_the_anchor_blocks_alone():
    for persona_name in PERSONA_NAMES:
        for day_type in DayType:
            for seed in range(40):
                acts = plan_day(PERSONAS[persona_name], day_type, rng_for(seed, "day"))
                names = [a.name for a in acts]
                assert names.count("sleep") == 2
                assert names.count("wake_up") == 1 and names.count("wind_down") <= 1
                assert names.count("bath") == 1
