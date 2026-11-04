"""Home archetypes. Layouts are data (ARCHETYPE_SPECS rows); build_home does the rest.

Rooms are living, kitchen, the spec's extra single-instance rooms, then bedroom_<n> and
bathroom_<n>. Each room draws its loads from the tier's inventory (gaia/home/inventory.py).
Randomness is drawn only from the rng passed in, in a fixed order, so the same rng state always
yields the same manifest.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import numpy as np

from gaia.home.inventory import (
    TIER_LIGHTING,
    assign_protocol,
    choose_entries,
    room_type_of,
    weighted_choice,
)
from gaia.home.manifest import HomeManifest, Load, Room


@dataclass(frozen=True)
class ArchetypeSpec:
    name: str
    bedrooms: int
    bathrooms: int
    has_separate_living: bool
    household_size: tuple[int, int]  # (min, max) members
    extra_rooms: tuple[str, ...]  # single-instance room types, in layout order
    # dict is unhashable; excluded from hash, still compared for equality.
    tier_weights: dict[str, float] = field(hash=False)


ARCHETYPE_SPECS: dict[str, ArchetypeSpec] = {
    s.name: s
    for s in (
        ArchetypeSpec("apartment_1rk", 0, 1, False, (1, 2), (), {"basic": 1.0}),
        ArchetypeSpec("apartment_1bhk", 1, 1, True, (1, 2), (), {"basic": 1.0}),
        ArchetypeSpec("apartment_2bhk", 2, 2, True, (2, 3), (), {"standard": 1.0}),
        ArchetypeSpec("apartment_3bhk", 3, 3, True, (2, 4), (), {"standard": 0.6, "premium": 0.4}),
        ArchetypeSpec("apartment_4bhk", 4, 4, True, (3, 5), (), {"premium": 1.0}),
        ArchetypeSpec("independent_house", 3, 3, True, (3, 6),
                      ("dining", "pooja", "balcony", "passage", "staircase"), {"premium": 1.0}),
        ArchetypeSpec("villa", 4, 5, True, (3, 7),
                      ("dining", "pooja", "home_theatre", "study", "dressing", "terrace", "garden",
                       "lobby", "staircase", "utility"), {"luxury": 1.0}),
    )
}

# The five apartment archetypes (the default run config and the G1 set).
APARTMENTS: dict[str, ArchetypeSpec] = {
    name: spec for name, spec in ARCHETYPE_SPECS.items() if name.startswith("apartment_")
}

# window_factor: share of outdoor lux that reaches the room (assumption; no ETS data yet).
_WINDOW = {
    "living": 0.03, "kitchen": 0.02, "bedroom": 0.02, "bathroom": 0.005,
    "dining": 0.02, "pooja": 0.01, "home_theatre": 0.0, "study": 0.02, "dressing": 0.01,
    "terrace": 0.5, "garden": 1.0, "lobby": 0.01, "staircase": 0.01, "utility": 0.01,
    "passage": 0.005, "balcony": 0.3,
}


def _room_names(spec: ArchetypeSpec) -> list[str]:
    names = ["living", "kitchen", *spec.extra_rooms]
    names += [f"bedroom_{i}" for i in range(1, spec.bedrooms + 1)]
    names += [f"bathroom_{i}" for i in range(1, spec.bathrooms + 1)]
    return names


def build_home(archetype: ArchetypeSpec, home_id: str, city: str, rng: np.random.Generator) -> HomeManifest:
    tier = weighted_choice(list(archetype.tier_weights.items()), rng)
    lighting_system = weighted_choice(TIER_LIGHTING[tier], rng)
    rooms: list[Room] = []
    loads: list[Load] = []

    for name in _room_names(archetype):
        room_id = f"{home_id}:{name}"
        rooms.append(Room(room_id, room_type_of(name), _WINDOW[room_type_of(name)]))
        for counter, entry in enumerate(choose_entries(tier, archetype.name, name, rng)):
            system, protocol = assign_protocol(entry.load_class, lighting_system, rng)
            loads.append(Load(
                load_id=f"{room_id}:{entry.load_class}:{counter}", room_id=room_id,
                load_class=entry.load_class, protocol=protocol, system=system,
                circuit=entry.circuit,
                fixture=entry.fixture,
            ))

    hub = f"{home_id}:living"
    adjacency: dict[str, tuple[str, ...]] = {}
    for room in rooms:
        if room.room_id == hub:
            adjacency[room.room_id] = tuple(r.room_id for r in rooms if r.room_id != hub)
        else:
            adjacency[room.room_id] = (hub,)

    return HomeManifest(
        home_id=home_id,
        archetype=archetype.name,
        city=city,
        tier=tier,
        lighting_system=lighting_system,
        rooms=tuple(rooms),
        adjacency=adjacency,
        loads=tuple(loads),
    )


ARCHETYPES: dict[str, Callable[[str, str, np.random.Generator], HomeManifest]] = {
    name: partial(build_home, spec) for name, spec in ARCHETYPE_SPECS.items()
}
