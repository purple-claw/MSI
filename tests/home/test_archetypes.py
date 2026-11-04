from collections import Counter

import pytest

from gaia.home.archetypes import ARCHETYPES, APARTMENTS
from gaia.registry.load_classes import REGISTRY
from gaia.rng import rng_for

ARCHETYPE_NAMES = sorted(APARTMENTS)


def _build(name: str, seed: int):
    return ARCHETYPES[name]("home_x", "Mumbai", rng_for(seed, "home"))


@pytest.mark.parametrize("name", ARCHETYPE_NAMES)
def test_every_room_has_one_main_light(name):
    for seed in range(200):
        m = _build(name, seed)
        for room in m.rooms:
            mains = [ld for ld in m.loads_in(room.room_id) if ld.is_main_light]
            assert len(mains) == 1, (seed, room.room_id)
            assert m.main_light(room.room_id) == mains[0]
            assert mains[0].load_class in ("light.dim", "light.onoff")


@pytest.mark.parametrize("name", ARCHETYPE_NAMES)
def test_room_counts_match_spec(name):
    spec = APARTMENTS[name]
    m = _build(name, 1)
    types = Counter(r.room_type for r in m.rooms)
    assert types["bedroom"] == spec.bedrooms
    assert types["bathroom"] == spec.bathrooms
    assert types["living"] == 1
    assert types["kitchen"] == 1
    if spec.bedrooms == 0:
        assert {r.room_id.split(":")[-1] for r in m.rooms} == {"living", "kitchen", "bathroom_1"}


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


def test_protocol_mix_roughly_matches():
    total = knx = 0
    for seed in range(500):
        for name in ARCHETYPE_NAMES:
            for load in _build(name, seed).loads:
                total += 1
                knx += load.protocol.value == "knx"
    assert 0.55 <= knx / total <= 0.65
