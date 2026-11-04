from collections import Counter

import pytest

from gaia.home.archetypes import ARCHETYPE_SPECS, ARCHETYPES
from gaia.registry.registry import REGISTRY
from gaia.rng import rng_for

ARCHETYPE_NAMES = sorted(ARCHETYPE_SPECS)


def _build(name: str, seed: int):
    return ARCHETYPES[name]("home_x", "Mumbai", rng_for(seed, "home"))


@pytest.mark.parametrize("name", ARCHETYPE_NAMES)
def test_room_counts_match_spec(name):
    spec = ARCHETYPE_SPECS[name]
    m = _build(name, 1)
    types = Counter(r.room_type for r in m.rooms)
    assert types["bedroom"] == spec.bedrooms
    assert types["bathroom"] == spec.bathrooms
    assert types["living"] == 1
    assert types["kitchen"] == 1
    assert types["bedroom"] + types["bathroom"] + len(spec.extra_rooms) + 2 == len(m.rooms)
    if spec.bedrooms == 0:
        assert {r.room_id.split(":")[-1] for r in m.rooms} == {"living", "kitchen", "bathroom_1"}
    assert {r.room_id.split(":")[-1] for r in m.rooms} == {
        "living", "kitchen", *spec.extra_rooms,
        *(f"bedroom_{i}" for i in range(1, spec.bedrooms + 1)),
        *(f"bathroom_{i}" for i in range(1, spec.bathrooms + 1)),
    }


@pytest.mark.parametrize("name", ARCHETYPE_NAMES)
def test_rooms_connected_via_living(name):
    m = _build(name, 7)
    living = "home_x:living"
    room_ids = [r.room_id for r in m.rooms]
    assert set(m.adjacency) == set(room_ids)
    assert set(m.adjacency[living]) == set(room_ids) - {living}
    for rid in room_ids:
        if rid != living:
            assert m.adjacency[rid] == (living,)
    # BFS from living reaches every room
    seen, frontier = {living}, [living]
    while frontier:
        nxt = [n for cur in frontier for n in m.adjacency[cur] if n not in seen]
        seen.update(nxt)
        frontier = nxt
    assert seen == set(room_ids)


@pytest.mark.parametrize("name", ARCHETYPE_NAMES)
def test_same_seed_same_manifest_json(name):
    a = _build(name, 42)
    b = _build(name, 42)
    assert a.to_json() == b.to_json()
    assert a == b
    assert _build(name, 43).to_json() != a.to_json()


@pytest.mark.parametrize("name", ARCHETYPE_NAMES)
def test_load_classes_registered(name):
    for seed in range(50):
        for load in _build(name, seed).loads:
            assert load.load_class in REGISTRY
