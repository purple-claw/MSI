from dataclasses import replace

import pytest

from gaia.home.archetypes import ARCHETYPE_SPECS, ARCHETYPES
from gaia.household.personas import PERSONAS, make_household
from gaia.rng import rng_for

ARCHETYPE_NAMES = sorted(ARCHETYPE_SPECS)


def _home(name: str, seed: int):
    manifest = ARCHETYPES[name]("h1", "Mumbai", rng_for(seed, "home"))
    household = make_household(manifest, ARCHETYPE_SPECS[name], rng_for(seed, "household"))
    return manifest, household


def test_household_ids_stable():
    _, a = _home("apartment_3bhk", 11)
    _, b = _home("apartment_3bhk", 11)
    assert [m.person_id for m in a.members] == [m.person_id for m in b.members]
    assert [m.persona.role for m in a.members] == [m.persona.role for m in b.members]
    assert [m.person_id for m in a.members] == [f"h1:p{n}" for n in range(len(a.members))]


def test_first_member_is_working_professional():
    for seed in range(50):
        _, household = _home("apartment_2bhk", seed)
        assert household.members[0].persona.role == "working_professional"


@pytest.mark.parametrize("name", ARCHETYPE_NAMES)
def test_household_fits_home(name):
    spec = ARCHETYPE_SPECS[name]
    lo, hi = spec.household_size
    for seed in range(200):
        manifest, household = _home(name, seed)
        room_names = {r.room_id.split(":", 1)[1] for r in manifest.rooms}
        assert lo <= len(household.members) <= hi
        for m in household.members:
            assert m.bedroom in room_names
            assert m.bathroom in room_names
            if spec.bedrooms == 0:
                assert m.bedroom == "living"


def test_couple_shares_bedroom_one():
    for seed in range(50):
        _, household = _home("apartment_4bhk", seed)
        if len(household.members) >= 2:
            assert household.members[0].bedroom == "bedroom_1"
            assert household.members[1].bedroom == "bedroom_1"


def test_later_members_fill_bedrooms_in_order():
    for seed in range(200):
        _, household = _home("apartment_4bhk", seed)
        for k, m in enumerate(household.members[2:], start=2):
            assert m.bedroom == f"bedroom_{k}"


def test_bathrooms_round_robin():
    for seed in range(50):
        manifest, household = _home("apartment_3bhk", seed)
        for k, m in enumerate(household.members):
            assert m.bathroom == f"bathroom_{(k % 3) + 1}"


def test_personas_are_the_two_g1_entries():
    assert set(PERSONAS) == {"working_professional", "homemaker"}
    assert PERSONAS["working_professional"].leave_min == (525, 20)
    assert PERSONAS["homemaker"].leave_min is None
    assert replace(PERSONAS["homemaker"], role="x").role == "x"
