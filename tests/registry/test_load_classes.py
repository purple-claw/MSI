import dataclasses

import pytest

from gaia.registry.load_classes import REGISTRY, LoadClass, get_load_class


def test_all_spec_classes_present():
    expected = {
        "light.onoff", "light.dim", "light.cct", "light.rgb",
        "fan.speed", "ac", "curtain.position", "blind.tilt", "plug",
        "geyser", "water_pump", "exhaust", "tv_media", "mosquito_repellent",
        "scene_button",
        "motion", "door", "temp", "humidity", "lux", "energy_meter",
    }
    assert set(REGISTRY.keys()) == expected


def test_light_dim_caps():
    assert get_load_class("light.dim").capabilities == ("power", "level")


def test_unknown_raises():
    with pytest.raises(KeyError, match="toaster"):
        get_load_class("toaster")


def test_sensors_zero_watts():
    sensors = [lc for lc in REGISTRY.values() if lc.is_sensor]
    assert {lc.name for lc in sensors} == {
        "motion", "door", "temp", "humidity", "lux", "energy_meter"
    }
    assert all(lc.rated_watts == 0 for lc in sensors)


def test_load_class_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        get_load_class("plug").rated_watts = 5.0
