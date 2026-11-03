"""Load-class registry v0 (G0).

Each entry describes what a device class can do and its rated power draw.
Sensors draw no power in the model (rated_watts == 0).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LoadClass:
    name: str
    capabilities: tuple[str, ...]
    rated_watts: float
    is_sensor: bool = False


_ENTRIES: tuple[LoadClass, ...] = (
    LoadClass("light.onoff", ("power",), 12),
    LoadClass("light.dim", ("power", "level"), 12),
    LoadClass("light.cct", ("power", "level", "cct"), 15),
    LoadClass("light.rgb", ("power", "level", "color"), 15),
    LoadClass("fan.speed", ("power", "fan_speed"), 75),
    LoadClass("ac", ("power", "mode", "setpoint", "fan_speed"), 1500),
    LoadClass("curtain.position", ("position",), 60),
    LoadClass("blind.tilt", ("position", "tilt"), 40),
    LoadClass("plug", ("power",), 100),
    LoadClass("geyser", ("power",), 2000),
    LoadClass("water_pump", ("power",), 750),
    LoadClass("exhaust", ("power",), 40),
    LoadClass("tv_media", ("power",), 120),
    LoadClass("mosquito_repellent", ("power",), 5),
    LoadClass("scene_button", ("press",), 0),
    LoadClass("motion", ("occupancy",), 0, is_sensor=True),
    LoadClass("door", ("contact",), 0, is_sensor=True),
    LoadClass("temp", ("temperature",), 0, is_sensor=True),
    LoadClass("humidity", ("humidity",), 0, is_sensor=True),
    LoadClass("lux", ("illuminance",), 0, is_sensor=True),
    LoadClass("energy_meter", ("power_w", "energy_wh"), 0, is_sensor=True),
)

REGISTRY: dict[str, LoadClass] = {lc.name: lc for lc in _ENTRIES}


def get_load_class(name: str) -> LoadClass:
    """Return the load class for ``name``; raise KeyError naming the unknown class."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown load class: {name!r}") from None
