"""Apartment archetypes. Layouts are data (APARTMENTS rows); build_apartment does the rest.

Randomness is drawn only from the rng passed in, in a fixed order, so the same
rng state always yields the same manifest.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np

from gaia.home.manifest import HomeManifest, Load, Room
from gaia.registry.registry import get_load_class
from gaia.schema.events import Protocol


@dataclass(frozen=True)
class ApartmentSpec:
    name: str
    bedrooms: int
    bathrooms: int
    has_separate_living: bool
    household_size: tuple[int, int]  # (min, max) members


APARTMENTS: dict[str, ApartmentSpec] = {
    s.name: s
    for s in (
        ApartmentSpec("apartment_1rk", 0, 1, False, (1, 2)),
        ApartmentSpec("apartment_1bhk", 1, 1, True, (1, 2)),
        ApartmentSpec("apartment_2bhk", 2, 2, True, (2, 3)),
        ApartmentSpec("apartment_3bhk", 3, 3, True, (2, 4)),
        ApartmentSpec("apartment_4bhk", 4, 4, True, (3, 5)),
    )
}

_WINDOW = {"living": 0.03, "kitchen": 0.02, "bedroom": 0.02, "bathroom": 0.005}


def _room_names(spec: ApartmentSpec) -> list[str]:
    names = ["living", "kitchen"]
    names += [f"bedroom_{i}" for i in range(1, spec.bedrooms + 1)]
    names += [f"bathroom_{i}" for i in range(1, spec.bathrooms + 1)]
    return names


def _protocol(rng: np.random.Generator) -> Protocol:
    x = rng.random()
    if x < 0.6:
        return Protocol.knx
    if x < 0.85:
        return Protocol.zigbee
    return Protocol.matter


def _room_loads(room_name: str, is_1rk: bool) -> list[tuple[str, float | None]]:
    """Return (load_class, presence probability or None for always-present) per room.

    The main light is always first; its class is drawn separately.
    """
    if room_name == "living":
        extras: list[tuple[str, float | None]] = [("fan.ceiling", None)]
        if is_1rk:
            extras.append(("climate.split_ac", 0.5))
        extras.append(("media.tv", None))
        return extras
    if room_name == "kitchen":
        return [("fan.exhaust", None)]
    if room_name.startswith("bedroom_"):
        p_ac = 0.8 if room_name == "bedroom_1" else 0.4
        return [("fan.ceiling", None), ("climate.split_ac", p_ac)]
    return [("water.geyser", 0.5)]  # bathroom


def build_apartment(spec: ApartmentSpec, home_id: str, city: str, rng: np.random.Generator) -> HomeManifest:
    is_1rk = spec.bedrooms == 0
    names = _room_names(spec)
    rooms: list[Room] = []
    loads: list[Load] = []

    for name in names:
        room_type = name.rsplit("_", 1)[0] if name[-1].isdigit() else name
        room_id = f"{home_id}:{name}"
        rooms.append(Room(room_id, room_type, _WINDOW[room_type]))

        counter = 0

        def add(load_class: str, is_main: bool = False) -> None:
            nonlocal counter
            get_load_class(load_class)  # fail fast on unknown class
            loads.append(
                Load(f"{room_id}:{load_class}:{counter}", room_id, load_class, _protocol(rng), is_main)
            )
            counter += 1

        add("light.dimmable" if rng.random() < 0.5 else "light.onoff", is_main=True)
        for load_class, p in _room_loads(name, is_1rk):
            if p is None or rng.random() < p:
                add(load_class)

    hub = f"{home_id}:living"
    adjacency: dict[str, tuple[str, ...]] = {}
    for room in rooms:
        if room.room_id == hub:
            adjacency[room.room_id] = tuple(r.room_id for r in rooms if r.room_id != hub)
        else:
            adjacency[room.room_id] = (hub,)

    return HomeManifest(
        home_id=home_id,
        archetype=spec.name,
        city=city,
        rooms=tuple(rooms),
        adjacency=adjacency,
        loads=tuple(loads),
    )


ARCHETYPES: dict[str, Callable[[str, str, np.random.Generator], HomeManifest]] = {
    name: partial(build_apartment, spec) for name, spec in APARTMENTS.items()
}
