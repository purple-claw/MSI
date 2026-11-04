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
    load_inventory,
)
from gaia.registry.registry import FIXTURES, REGISTRY
from gaia.rng import rng_for
from gaia.schema.events import Protocol

ARCHETYPE_NAMES = sorted(ARCHETYPE_SPECS)
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
