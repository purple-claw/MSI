import json

import pytest
import yaml

from gaia.home.archetypes import ARCHETYPE_SPECS, ARCHETYPES
from gaia.home.inventory import (
    CIRCUITS,
    INVENTORY,
    MAIN_CLASSES,
    TIER_LIGHTING,
    TIERS,
    choose_entries,
    load_inventory,
)
from gaia.registry.registry import FIXTURES, REGISTRY
from gaia.rng import rng_for
from gaia.schema.events import Protocol

ARCHETYPE_NAMES = sorted(ARCHETYPE_SPECS)
DIST_HOMES = 400
SHARE_TOL = 0.07
DALI_CLASSES = {"light.tunable", "light.rgbw", "light.rgbcct"}


def _build(name: str, seed: int, home_id: str = "h1"):
    return ARCHETYPES[name](home_id, "Mumbai", rng_for(seed, "home"))


@pytest.mark.parametrize("name", ARCHETYPE_NAMES)
def test_every_lit_room_one_main_circuit(name):
    for seed in range(100):
        m = _build(name, seed)
        for room in m.rooms:
            mains = [ld for ld in m.loads_in(room.room_id) if ld.is_main_light]
            assert len(mains) == 1, (seed, room.room_id)
            assert mains[0].circuit == "main"
            assert mains[0].load_class in MAIN_CLASSES
            assert m.main_light(room.room_id) == mains[0]


def test_luxury_living_has_layers():
    villas = [_build("villa", seed) for seed in range(200)]
    living = [m.loads_in("h1:living") for m in villas]
    with_layers = sum(any(ld.circuit in ("cove", "accent") for ld in lds) for lds in living)
    with_tunable = sum(any(ld.load_class == "light.tunable" for ld in lds) for lds in living)
    assert with_layers >= 100
    assert with_tunable >= 100


def test_loads_per_room_range():
    for seed in range(200):
        n = len(_build("villa", seed).loads_in("h1:living"))
        assert 4 <= n <= 25, (seed, n)


# Expected (system, protocol) per lighting system and class. Written out here rather than
# derived from the code under test.
_LIGHT_PROTOCOL = {
    "knx_dali": {"light.onoff": "knx", "light.dimmable": "knx", "light.led_strip": "knx",
                 "light.tunable": "dali", "light.rgbw": "dali", "light.rgbcct": "dali"},
    "lutron_ra3": "lutron",
    "lutron_caseta": "lutron",
    "zigbee_hue": "zigbee",
    "matter_wifi": "matter",
}


def _expected_system_protocol(system: str, load_class: str) -> str:
    table = _LIGHT_PROTOCOL[system]
    if isinstance(table, str):
        return table
    # Shades and other knx_dali loads use knx; only DALI-2 DT8 classes use dali.
    return table.get(load_class, "knx") if load_class not in DALI_CLASSES else table[load_class]


@pytest.mark.parametrize("name", ARCHETYPE_NAMES)
def test_protocol_matches_system(name):
    seen_wifi = seen_matter = seen_ir = False
    for seed in range(50):
        m = _build(name, seed)
        for ld in m.loads:
            cls = ld.load_class
            got = (ld.system, ld.protocol.value)
            if cls.startswith(("light.", "shade.")):
                assert got == (m.lighting_system, _expected_system_protocol(m.lighting_system, cls)), ld
            elif cls.startswith("media."):
                assert got == ("ip", "ip"), ld
            elif cls in ("climate.split_ac", "climate.cassette_ac"):
                ir = ("ir_gateway", "ir")
                system = (m.lighting_system, _expected_system_protocol(m.lighting_system, cls))
                assert got in (ir, system), ld
                seen_ir |= got == ir
            elif cls.startswith(("sensor.", "meter.")) and got == ("zigbee_hue", "zigbee"):
                pass  # sensors added by sensor richness ride zigbee or Matter (see sensor_loads)
            else:
                assert got in (("ip", "wifi"), ("matter_wifi", "matter")), ld
                seen_wifi |= got == ("ip", "wifi")
                seen_matter |= got == ("matter_wifi", "matter")
    assert all(isinstance(ld.protocol, Protocol) for ld in _build(name, 0).loads)
    if name != "apartment_1rk":
        assert seen_wifi and seen_matter and seen_ir


def test_all_classes_registered():
    for tier, rooms in INVENTORY.items():
        for room_type, entries in rooms.items():
            for e in entries:
                assert e.load_class in REGISTRY, (tier, room_type, e.load_class)
                assert e.circuit in CIRCUITS | {"none"}
                assert e.fixture is None or e.fixture in FIXTURES
    for name in ARCHETYPE_NAMES:
        for seed in range(20):
            for ld in _build(name, seed).loads:
                assert ld.load_class in REGISTRY


def test_manifest_json_stable():
    for name in ARCHETYPE_NAMES:
        first = _build(name, 42).to_json()
        _build(name, 99)  # no state leaks between builds
        assert _build(name, 42).to_json() == first
        payload = json.loads(first)
        assert payload["tier"] in TIERS
        assert payload["lighting_system"] in {s for s, _ in TIER_LIGHTING[payload["tier"]]}
        assert all({"system", "circuit", "fixture", "is_main_light"} <= set(ld) for ld in payload["loads"])
        assert json.dumps(payload, sort_keys=True, separators=(",", ":")) == first


def test_tiers_per_archetype():
    assert {tuple(spec.tier_weights) for spec in ARCHETYPE_SPECS.values()} <= {
        ("basic",), ("standard",), ("premium",), ("luxury",), ("standard", "premium")
    }
    assert ARCHETYPE_SPECS["villa"].tier_weights == {"luxury": 1.0}
    assert ARCHETYPE_SPECS["independent_house"].tier_weights == {"premium": 1.0}
    assert ARCHETYPE_SPECS["apartment_3bhk"].tier_weights == {"standard": 0.6, "premium": 0.4}


def _base_inventory() -> dict:
    main = {"class": "light.onoff", "fixture": "bulb", "circuit": "main", "count": [1, 1], "p": 1.0}
    return {tier: {"bedroom": [main]} for tier in TIERS}


def test_loader_accepts_minimal_inventory():
    text = yaml.safe_dump(_base_inventory())
    assert load_inventory(text)["luxury"]["bedroom"][0].circuit == "main"


@pytest.mark.parametrize("mutate, must_name", [
    (lambda d: d["basic"]["bedroom"].clear() or d["basic"]["bedroom"].append(
        {"class": "fan.ceiling", "fixture": None, "circuit": "none", "count": [1, 1], "p": 1.0}),
     "main circuit"),
    (lambda d: d["basic"]["bedroom"][0].update(count=[1, 2]), "count"),
    (lambda d: d["basic"]["bedroom"][0].update({"class": "light.nope"}), "light.nope"),
    (lambda d: d["basic"]["bedroom"][0].update(fixture="chandelierx"), "chandelierx"),
    (lambda d: d.pop("premium"), "premium"),
])
def test_loader_rejects_bad_inventory(mutate, must_name):
    data = _base_inventory()
    mutate(data)
    with pytest.raises(ValueError, match=must_name):
        load_inventory(yaml.safe_dump(data))


def test_loader_rejects_duplicate_room_key():
    text = yaml.safe_dump(_base_inventory()) + "basic:\n  bedroom: []\n"
    with pytest.raises(ValueError):
        load_inventory(text)


# Minimums the brief pins. Each row: (tier, room, predicate on entries, description).
_MINIMUMS = [
    ("luxury", "home_theatre", lambda es: all(_entries_has(es, c, 1.0) for c in
                                             ("media.projector", "media.screen_motorised", "media.av_receiver")),
     "projector, motorised screen and AV receiver"),
    ("luxury", "home_theatre", lambda es: _entries_has(es, "climate.split_ac", 1.0), "an AC"),
    ("basic", "bedroom", lambda es: _has_prefix(es, "fan."), "a fan"),
    ("basic", "bedroom", lambda es: _entries_has(es, "climate.split_ac"), "an AC"),
    ("standard", "bedroom", lambda es: _has_prefix(es, "fan."), "a fan"),
    ("standard", "bedroom", lambda es: _entries_has(es, "climate.split_ac"), "an AC"),
    ("premium", "bedroom", lambda es: _has_prefix(es, "fan."), "a fan"),
    ("premium", "bedroom", lambda es: _entries_has(es, "climate.split_ac"), "an AC"),
    ("luxury", "bedroom", lambda es: _has_prefix(es, "fan."), "a fan"),
    ("luxury", "bedroom", lambda es: _entries_has(es, "climate.split_ac"), "an AC"),
    ("premium", "pooja", lambda es: any(e.load_class == "light.onoff" and e.fixture == "accent"
                                        and e.circuit == "main" for e in es), "an accent lamp on main"),
    ("luxury", "pooja", lambda es: any(e.load_class == "light.onoff" and e.fixture == "accent"
                                       and e.circuit == "main" for e in es), "an accent lamp on main"),
    ("premium", "passage", lambda es: any(e.circuit == "night" for e in es), "night step lights"),
    ("premium", "staircase", lambda es: any(e.circuit == "night" for e in es), "night step lights"),
    ("luxury", "staircase", lambda es: any(e.circuit == "night" for e in es), "night step lights"),
]


def _entries_has(es, load_class, min_p=0.0):
    return any(e.load_class == load_class and e.p >= min_p for e in es)


def _has_prefix(es, prefix):
    return any(e.load_class.startswith(prefix) for e in es)


@pytest.mark.parametrize("tier, room, check, what", _MINIMUMS,
                         ids=[f"{t}-{r}-{w}" for t, r, _, w in _MINIMUMS])
def test_minimum_coverage(tier, room, check, what):
    assert check(INVENTORY[tier][room]), f"{tier}/{room} needs {what}"


@pytest.mark.parametrize("tier", ["premium", "luxury"])
@pytest.mark.parametrize("room", ["living", "bedroom"])
def test_premium_plus_layers_at_least_half(tier, room):
    es = INVENTORY[tier][room]
    assert any(e.circuit == "cove" and e.p >= 0.5 for e in es), "cove"
    assert any(e.circuit == "accent" and e.p >= 0.5 for e in es), "accent"
    assert any(e.load_class == "light.tunable" and e.p >= 0.5 for e in es), "tunable"


def test_every_source_is_assumption_or_url():
    for tier, rooms in INVENTORY.items():
        for room, entries in rooms.items():
            for e in entries:
                assert e.source == "assumption" or e.source.startswith("https://"), (tier, room, e)


def test_loader_rejects_bad_source():
    data = _base_inventory()
    data["basic"]["bedroom"][0].update(source="http://example.com")
    with pytest.raises(ValueError, match="source"):
        load_inventory(yaml.safe_dump(data))


def test_archetype_filters_name_known_archetypes():
    for tier, rooms in INVENTORY.items():
        for room, entries in rooms.items():
            for e in entries:
                for name in e.archetypes:
                    assert name in ARCHETYPE_SPECS, (tier, room, name)


@pytest.mark.parametrize("archetype", ARCHETYPE_NAMES)
def test_every_room_keeps_a_main_after_archetype_filter(archetype):
    spec = ARCHETYPE_SPECS[archetype]
    for tier in spec.tier_weights:
        names = ["living", "kitchen", *spec.extra_rooms,
                 *(f"bedroom_{i}" for i in range(1, spec.bedrooms + 1)),
                 *(f"bathroom_{i}" for i in range(1, spec.bathrooms + 1))]
        for name in names:
            chosen = choose_entries(tier, archetype, name, rng_for(3, name))
            assert sum(e.circuit == "main" for e in chosen) == 1, (tier, name)


def test_entrances_by_archetype():
    # 4BHK is premium only: living entry lock at p 0.6 (apartments), none in the independent house.
    apt = [_build("apartment_4bhk", s) for s in range(400)]
    with_lock = sum(any(ld.load_class == "lock.smart" for ld in m.loads_in("h1:living")) for m in apt)
    assert abs(with_lock / len(apt) - 0.6) <= SHARE_TOL
    ind = [_build("independent_house", s) for s in range(200)]
    assert all(not any(ld.load_class == "lock.smart" for ld in m.loads_in("h1:living")) for m in ind)
    villas = [_build("villa", s) for s in range(200)]
    assert all(not any(ld.load_class == "lock.smart" for ld in m.loads_in("h1:living")) for m in villas)


def test_distribution_of_tiers_and_lighting_systems():
    from collections import Counter

    from gaia.home.inventory import TIER_LIGHTING
    by_tier = {t: Counter() for t in TIER_LIGHTING}
    for name in ARCHETYPE_NAMES:
        spec = ARCHETYPE_SPECS[name]
        tiers = Counter()
        for seed in range(DIST_HOMES):
            m = _build(name, seed)
            tiers[m.tier] += 1
            by_tier[m.tier][m.lighting_system] += 1
        for tier, w in spec.tier_weights.items():
            share = tiers[tier] / DIST_HOMES
            assert abs(share - w) <= SHARE_TOL, (name, tier, share, w)
        assert set(tiers) <= set(spec.tier_weights), (name, tiers)
    for tier, options in TIER_LIGHTING.items():
        total = sum(by_tier[tier].values())
        assert total > 0, tier
        weight_sum = sum(w for _, w in options)
        for system, w in options:
            share = by_tier[tier][system] / total
            assert abs(share - w / weight_sum) <= SHARE_TOL, (tier, system, share, w)
