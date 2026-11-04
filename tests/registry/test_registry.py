import dataclasses

import pytest

from gaia.registry.registry import (
    FIXTURES, REGISTRY, Capability, LoadClass, get_load_class, load_registry,
)
from gaia.schema.events import Protocol

EXPECTED_CLASSES = {
    # lights
    "light.onoff", "light.dimmable", "light.tunable", "light.rgbw", "light.rgbcct",
    "light.led_strip", "light.festival",
    # media
    "media.tv", "media.stb_dth", "media.streamer", "media.speaker", "media.soundbar",
    "media.av_receiver", "media.projector", "media.screen_motorised",
    # climate
    "climate.split_ac", "climate.vrf_indoor", "climate.cassette_ac", "fan.bldc_ceiling",
    "fan.ceiling", "fan.exhaust", "air.purifier", "air.dehumidifier",
    # shading
    "shade.curtain", "shade.roller", "shade.venetian", "shade.skylight",
    # water and utility
    "water.geyser", "water.pump", "water.ro_purifier", "kitchen.chimney", "kitchen.dishwasher",
    "laundry.washing_machine", "power.inverter_ups", "ev.charger", "plug.smart",
    "misc.mosquito_repellent",
    # security
    "lock.smart", "doorbell.video", "gate.motor", "cctv.camera",
    # sensors
    "sensor.motion", "sensor.presence", "sensor.contact", "sensor.temperature",
    "sensor.humidity", "sensor.lux", "sensor.co2", "sensor.water_leak", "meter.energy",
    # keypads
    "keypad.scene",
}

EXPECTED_FIXTURES = {
    "downlight", "profile", "cove", "pendant", "chandelier", "track", "wall_washer",
    "step", "landscape", "facade", "accent", "strip", "bulb",
}


def test_registry_has_exactly_the_spec_classes():
    assert set(REGISTRY) == EXPECTED_CLASSES


def test_registry_entries_are_load_classes():
    assert all(isinstance(lc, LoadClass) and lc.name == name for name, lc in REGISTRY.items())


def test_fixtures_match_spec():
    assert isinstance(FIXTURES, frozenset)
    assert FIXTURES == EXPECTED_FIXTURES


def test_every_class_has_power_source():
    for lc in REGISTRY.values():
        assert lc.power_source, lc.name
        assert lc.power_source == "assumption" or lc.power_source.startswith("https://"), lc.name


def test_ranges_valid():
    for lc in REGISTRY.values():
        for cap in lc.capabilities:
            if cap.kind == "enum":
                assert cap.values, (lc.name, cap.name)
            if cap.lo is not None and cap.hi is not None:
                assert cap.lo < cap.hi, (lc.name, cap.name)


def test_curtain_position_is_normalised():
    cap = next(c for c in get_load_class("shade.curtain").capabilities if c.name == "position")
    assert cap.kind == "position"
    assert (cap.lo, cap.hi) == (0, 1)


def test_unknown_class_raises():
    with pytest.raises(KeyError, match="toaster"):
        get_load_class("toaster")


def test_sourced_numbers():
    speaker = get_load_class("media.speaker")
    assert speaker.power_w["standby"] == 4.3
    # Mixed class (standby sourced, on assumed): the class source must be the conservative `assumption`.
    assert speaker.power_source == "assumption"

    geyser = get_load_class("water.geyser")
    assert geyser.power_w["on"] == 2000
    assert geyser.power_source.startswith("https://")


def test_protocol_enum_extended():
    values = {p.value for p in Protocol}
    assert values == {"knx", "matter", "zigbee", "wifi", "other", "lutron", "dali", "ip", "ir"}


def test_load_class_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        get_load_class("water.geyser").power_source = "assumption"


def test_capability_is_frozen_and_typed():
    cap = Capability("power", "binary", 0, 1, None, None)
    assert cap.kind == "binary"
    with pytest.raises(dataclasses.FrozenInstanceError):
        cap.kind = "level"


def test_sensors_are_flagged_and_draw_no_power():
    sensors = {lc.name for lc in REGISTRY.values() if lc.is_sensor}
    assert sensors == {
        "sensor.motion", "sensor.presence", "sensor.contact", "sensor.temperature",
        "sensor.humidity", "sensor.lux", "sensor.co2", "sensor.water_leak", "meter.energy",
    }
    assert all(max(lc.power_w.values(), default=0) == 0 for lc in REGISTRY.values() if lc.is_sensor)


def test_waste_rules_are_valid():
    for lc in REGISTRY.values():
        rule = lc.waste_rule
        if rule in ("unattended", "none"):
            continue
        kind, _, minutes = rule.partition(":")
        assert kind == "overrun" and minutes.isdigit() and int(minutes) > 0, (lc.name, rule)


def test_light_tunable_cct_range():
    cap = next(c for c in get_load_class("light.tunable").capabilities if c.name == "cct")
    assert (cap.kind, cap.lo, cap.hi, cap.unit) == ("kelvin", 2700, 6500, "K")


def test_every_position_and_tilt_capability_has_semantics():
    caps = [c for lc in REGISTRY.values() for c in lc.capabilities if c.kind in ("position", "tilt")]
    assert caps
    for cap in caps:
        assert cap.semantics, cap


def test_shade_semantics_say_one_is_open():
    for name in ("shade.curtain", "shade.roller", "shade.venetian", "shade.skylight"):
        pos = next(c for c in get_load_class(name).capabilities if c.name == "position")
        assert "1=open" in pos.semantics, name


def test_every_non_sensor_class_has_on_power():
    for lc in REGISTRY.values():
        if not lc.is_sensor:
            assert "on" in lc.power_w, lc.name


def test_power_values_non_negative():
    assert all(v >= 0 for lc in REGISTRY.values() for v in lc.power_w.values())


MINIMAL = """\
light.onoff:
  domain: lighting
  matter_device_type: on_off_light
  capabilities:
    - {name: power, kind: binary}
  power_w: {"on": 12}
  power_source: assumption
  waste_rule: unattended
  rooms: [living]
"""


def test_minimal_fixture_loads():
    assert set(load_registry(MINIMAL)) == {"light.onoff"}


@pytest.mark.parametrize("text", ["", "- a\n- b\n", "just a string\n"])
def test_empty_or_non_mapping_yaml_raises(text):
    with pytest.raises(ValueError, match="mapping"):
        load_registry(text)


@pytest.mark.parametrize("text, must_name", [
    (MINIMAL + MINIMAL, "light.onoff"),
    (MINIMAL.replace("kind: binary", "kind: sparkle"), "light.onoff"),
    (MINIMAL.replace("waste_rule: unattended", "waste_rule: sometimes"), "light.onoff"),
    (MINIMAL.replace("power_source: assumption", "power_source: 12 W"), "light.onoff"),
    (MINIMAL.replace('{"on": 12}', "{on: 12}"), "light.onoff"),
    (MINIMAL.replace("    - {name: power, kind: binary}\n",
                     "    - {name: power, kind: binary}\n"
                     "    - {name: level, kind: level, lo: 0, hi: 100}\n"), "light.onoff"),
    (MINIMAL.replace("    - {name: power, kind: binary}\n",
                     "    - {name: power, kind: binary}\n"
                     "    - {name: power, kind: binary}\n"), "light.onoff"),
    (MINIMAL.replace('{"on": 12}', '{"on": -1}'), "light.onoff"),
    (MINIMAL.replace('{"on": 12}', '{standby: 1}'), "light.onoff"),
    (MINIMAL.replace("kind: binary}", "kind: position}"), "light.onoff"),
])
def test_malformed_registry_raises_naming_class(text, must_name):
    with pytest.raises(ValueError, match=must_name):
        load_registry(text)


def test_duplicate_class_key_is_named():
    with pytest.raises(ValueError, match="duplicate key 'light.onoff'"):
        load_registry(MINIMAL + MINIMAL)
