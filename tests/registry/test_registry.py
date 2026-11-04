import dataclasses

import pytest

from gaia.registry.registry import FIXTURES, REGISTRY, Capability, LoadClass, get_load_class
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
    assert speaker.power_source.startswith("https://")

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
