import dataclasses

import pytest
from hypothesis import given, strategies as st

from gaia.devices.state import DeviceState, HomeDevices
from gaia.home.manifest import HomeManifest, Load
from gaia.registry.registry import Capability, LoadClass, get_load_class
from gaia.schema.events import Protocol


def _manifest(*loads: Load) -> HomeManifest:
    return HomeManifest(
        home_id="h1", archetype="test", city="pune", rooms=(),
        adjacency={}, loads=tuple(loads),
    )


def _load(load_id: str, load_class: str) -> Load:
    return Load(load_id=load_id, room_id="living", load_class=load_class,
                protocol=Protocol.matter, is_main_light=False)


def test_clamp_cct():
    state = DeviceState(get_load_class("light.tunable"))

    assert state.set("cct", 9000) == (6500.0, 4600.0, True)
    assert state.get("cct") == 6500.0


def test_redundant_set_returns_none():
    state = DeviceState(get_load_class("light.tunable"))

    assert state.set("cct", 4600) is None
    assert state.set("level", 0) is None
    assert state.set("level", 0.5) == (0.5, 0.0, False)
    assert state.set("level", 0.5 + 1e-12) is None


def test_enum_validation():
    state = DeviceState(get_load_class("media.tv"))

    assert state.get("source") == "stb"
    with pytest.raises(ValueError):
        state.set("source", "bogus")
    assert state.get("source") == "stb"
    assert state.set("source", "spotify") == ("spotify", "stb", False)


def test_position_semantics():
    state = DeviceState(get_load_class("shade.curtain"))

    assert state.get("position") == 0.0
    assert state.set("position", 1.0) == (1.0, 0.0, False)
    assert state.get("position") == 1.0

    devices = HomeDevices(_manifest(_load("shade1", "shade.curtain")))
    assert devices.is_on("shade1") is False
    devices.get("shade1").set("position", 0.4)
    assert devices.is_on("shade1") is True


def test_requires_power():
    tunable = DeviceState(get_load_class("light.tunable"))
    assert tunable.requires_power("cct") is True
    assert tunable.requires_power("power") is False

    shade = DeviceState(get_load_class("shade.curtain"))
    assert shade.requires_power("position") is False

    button = LoadClass(
        name="test.button", domain="lighting", matter_device_type=None,
        capabilities=(
            Capability("power", "binary", 0.0, 1.0, None, None),
            Capability("press", "event", None, None, None, None),
        ),
        power_w={"on": 1.0}, power_source="assumption", waste_rule="none",
        is_sensor=False, rooms=("living",),
    )
    powered_event = DeviceState(button)
    assert powered_event.requires_power("press") is False
    assert powered_event.set("press", 1.0) == (1.0, None, False)
    assert powered_event.get("press") is None
    assert "press" not in powered_event.snapshot()


def test_unknown_capability_raises():
    state = DeviceState(get_load_class("light.dimmable"))

    with pytest.raises(ValueError):
        state.get("cct")
    with pytest.raises(ValueError):
        state.set("cct", 3000)


def test_count_rounds_to_int_valued_float():
    state = DeviceState(get_load_class("fan.ceiling"))

    assert state.get("fan_speed") == 1.0
    assert state.set("fan_speed", 3.4) == (3.0, 1.0, False)
    assert state.set("fan_speed", 9) == (5.0, 3.0, True)


def test_power_drives_is_on_for_powered_class():
    devices = HomeDevices(_manifest(_load("lamp", "light.dimmable")))
    assert devices.is_on("lamp") is False

    devices.get("lamp").set("power", 1)
    assert devices.is_on("lamp") is True
    assert devices.get("lamp").snapshot() == {"power": 1.0, "level": 0.0}


@given(st.floats(min_value=2700, max_value=6500, allow_nan=False))
def test_set_get_roundtrip_within_range(value):
    state = DeviceState(get_load_class("light.tunable"))

    state.set("cct", value)

    assert state.get("cct") == pytest.approx(value)


def test_binary_snaps_to_zero_or_one():
    state = DeviceState(get_load_class("light.dimmable"))

    assert state.set("power", 0.5) == (1.0, 0.0, True)
    assert state.set("power", 0.2) == (0.0, 1.0, True)
    assert state.set("power", 1) == (1.0, 0.0, False)
    assert state.set("power", 0) == (0.0, 1.0, False)


def test_on_off_capability_by_class():
    assert DeviceState(get_load_class("light.onoff")).on_off_capability() == "power"
    assert DeviceState(get_load_class("shade.curtain")).on_off_capability() == "position"
    assert DeviceState(get_load_class("sensor.temperature")).on_off_capability() is None
