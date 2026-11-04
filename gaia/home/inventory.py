"""Room inventory v1: loads per room by tier, lighting-system weights and protocol assignment.

inventory.yaml is validated at import. Room draws come only from the rng passed in: one weighted
choice for the room's main light, then each other entry in YAML order (presence, then count).
The main circuit is drawn first; the YAML order of the other entries is kept.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import TypeVar

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from gaia.home.manifest import Load, Room
from gaia.registry.registry import FIXTURES, REGISTRY, _StrictLoader, get_load_class
from gaia.schema.events import Protocol

TIERS = ("basic", "standard", "premium", "luxury")
CIRCUITS = frozenset({"main", "accent", "cove", "task", "night", "decor"})
MAIN_CLASSES = frozenset({"light.onoff", "light.dimmable", "light.tunable"})
DALI_CLASSES = frozenset({"light.tunable", "light.rgbw", "light.rgbcct"})
IR_CLASSES = frozenset({"climate.split_ac", "climate.cassette_ac"})
IR_SHARE = 0.6  # assumption: share of AC units on an IR gateway rather than the lighting system
WIFI_SHARE = 0.5  # assumption: share of Wi-Fi (rather than Matter) for fans, utility, security, sensors
BEDROOM_ONE_AC_BONUS = 0.2  # assumption: bedroom_1 is the main bedroom, so its AC p is raised
DEFAULT_SOURCE = "assumption"
SENSOR_RICHNESS: tuple[tuple[str, float], ...] = (("none", 0.4), ("some", 0.4), ("rich", 0.2))  # assumption
ZIGBEE_SENSOR_P = 0.5  # assumption: share of added sensors on zigbee rather than Matter
_SENSOR_SYSTEM = {Protocol.zigbee: "zigbee_hue", Protocol.matter: "matter_wifi"}

# Lighting system per tier, as (system, weight). Drawn once per home.
# assumption: weights are design choices until calibrated from ETS (Task 11).
TIER_LIGHTING: dict[str, tuple[tuple[str, float], ...]] = {
    "basic": (("zigbee_hue", 0.3), ("matter_wifi", 0.4), ("knx_dali", 0.3)),
    "standard": (("zigbee_hue", 0.3), ("matter_wifi", 0.4), ("knx_dali", 0.3)),
    "premium": (("knx_dali", 0.6), ("lutron_ra3", 0.3), ("zigbee_hue", 0.1)),
    "luxury": (("knx_dali", 0.6), ("lutron_ra3", 0.4)),
}

_SYSTEM_PROTOCOL = {
    "lutron_ra3": Protocol.lutron,
    "lutron_caseta": Protocol.lutron,
    "zigbee_hue": Protocol.zigbee,
    "matter_wifi": Protocol.matter,
}

_T = TypeVar("_T")


@dataclass(frozen=True)
class InventoryEntry:
    load_class: str
    fixture: str | None
    circuit: str
    count: tuple[int, int]
    p: float
    archetypes: tuple[str, ...]  # empty = every archetype using this tier and room
    source: str  # a https URL, or "assumption"


def room_type_of(room_name: str) -> str:
    """Room type of a room name: bedroom_2 -> bedroom, home_theatre -> home_theatre."""
    return room_name.rsplit("_", 1)[0] if room_name[-1].isdigit() else room_name


class _EntrySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    load_class: str = Field(alias="class")
    fixture: str | None
    circuit: str
    count: tuple[int, int]
    p: float = Field(ge=0, le=1)
    archetypes: tuple[str, ...] = ()
    source: str = DEFAULT_SOURCE

    @model_validator(mode="after")
    def _check(self):
        if self.load_class not in REGISTRY:
            raise ValueError(f"unknown load class {self.load_class!r}")
        if not (self.source == DEFAULT_SOURCE or self.source.startswith("https://")):
            raise ValueError(f"{self.load_class}: source must be a https URL or 'assumption': {self.source!r}")
        is_light = get_load_class(self.load_class).domain == "lighting"
        if is_light:
            if self.circuit not in CIRCUITS:
                raise ValueError(f"{self.load_class}: circuit must be one of {sorted(CIRCUITS)}")
            if self.fixture not in FIXTURES:
                raise ValueError(f"{self.load_class}: unknown fixture {self.fixture!r}")
        else:
            if self.circuit != "none" or self.fixture is not None:
                raise ValueError(f"{self.load_class}: non-lights take circuit none and fixture null")
        lo, hi = self.count
        if not 0 <= lo <= hi:
            raise ValueError(f"{self.load_class}: count must satisfy 0 <= min <= max, got {self.count}")
        if self.circuit == "main":
            if self.load_class not in MAIN_CLASSES:
                raise ValueError(f"main circuit needs one of {sorted(MAIN_CLASSES)}, got {self.load_class}")
            if self.count != (1, 1):
                raise ValueError("main circuit entries take count [1, 1]")
        return self


def _parse_tiers(raw: object) -> dict[str, dict[str, tuple[InventoryEntry, ...]]]:
    if not isinstance(raw, dict) or set(raw) != set(TIERS):
        raise ValueError(f"inventory YAML must map exactly the tiers {list(TIERS)}")
    out: dict[str, dict[str, tuple[InventoryEntry, ...]]] = {}
    for tier in TIERS:
        rooms = raw[tier]
        if not isinstance(rooms, dict) or not rooms:
            raise ValueError(f"tier {tier!r} must map room types to entry lists")
        out[tier] = {}
        for room_type, entries in rooms.items():
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"{tier}/{room_type}: needs a non-empty entry list")
            parsed = []
            for body in entries:
                try:
                    spec = _EntrySpec.model_validate(body)
                except ValidationError as exc:
                    raise ValueError(f"invalid entry in {tier}/{room_type}: {exc}") from exc
                parsed.append(InventoryEntry(spec.load_class, spec.fixture, spec.circuit, spec.count,
                                             spec.p, spec.archetypes, spec.source))
            if not any(e.circuit == "main" for e in parsed):
                raise ValueError(f"{tier}/{room_type}: every room needs one main circuit entry")
            if sum(e.p for e in parsed if e.circuit == "main") <= 0:
                raise ValueError(f"{tier}/{room_type}: main entries need a positive total weight")
            out[tier][room_type] = tuple(parsed)
    return out


def load_inventory(text: str) -> dict[str, dict[str, tuple[InventoryEntry, ...]]]:
    """Parse and validate inventory YAML text. Raises ValueError naming the offending entry."""
    try:
        raw = yaml.load(text, Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid inventory YAML: {exc}") from exc
    return _parse_tiers(raw)


INVENTORY = load_inventory(
    (files("gaia.home") / "inventory.yaml").read_text(encoding="utf-8")
)


def weighted_choice(options: Sequence[tuple[_T, float]], rng: np.random.Generator) -> _T:
    """Pick one option with probability proportional to its weight; always one rng draw."""
    total = sum(w for _, w in options)
    u = rng.random() * total
    acc = 0.0
    for value, w in options:
        acc += w
        if u < acc:
            return value
    return options[-1][0]


def choose_entries(tier: str, archetype: str, room_name: str, rng: np.random.Generator) -> list[InventoryEntry]:
    """Return the loads for one room, main circuit first, then the other entries in YAML order."""
    room_type = room_type_of(room_name)
    try:
        entries = INVENTORY[tier][room_type]
    except KeyError:
        raise KeyError(f"no inventory for tier {tier!r} room {room_type!r}") from None
    applicable = [e for e in entries if not e.archetypes or archetype in e.archetypes]
    mains = [(e, e.p) for e in applicable if e.circuit == "main"]
    chosen = [weighted_choice(mains, rng)]
    for e in applicable:
        if e.circuit == "main":
            continue
        p = e.p
        if room_name == "bedroom_1" and e.load_class in IR_CLASSES:
            p = min(1.0, p + BEDROOM_ONE_AC_BONUS)
        if rng.random() < p:
            lo, hi = e.count
            n = lo if lo == hi else int(rng.integers(lo, hi + 1))
            chosen.extend([e] * n)
    return chosen


def system_protocol(lighting_system: str, load_class: str) -> Protocol:
    """Protocol a lighting system uses for a load class (knx_dali splits DALI-2 DT8 classes)."""
    if lighting_system == "knx_dali":
        return Protocol.dali if load_class in DALI_CLASSES else Protocol.knx
    return _SYSTEM_PROTOCOL[lighting_system]


def assign_protocol(load_class: str, lighting_system: str, rng: np.random.Generator) -> tuple[str, Protocol]:
    """Return (system, protocol) for one load. Draws from rng only for the IR and Wi-Fi/Matter splits."""
    domain = get_load_class(load_class).domain
    if domain in ("lighting", "shading"):
        return lighting_system, system_protocol(lighting_system, load_class)
    if domain == "media":
        return "ip", Protocol.ip
    if load_class in IR_CLASSES:
        if rng.random() < IR_SHARE:
            return "ir_gateway", Protocol.ir
        return lighting_system, system_protocol(lighting_system, load_class)
    if load_class == "climate.vrf_indoor":
        return "knx_dali", Protocol.knx
    # Fans, utility, security and sensors: Wi-Fi or Matter.
    if rng.random() < WIFI_SHARE:
        return "ip", Protocol.wifi
    return "matter_wifi", Protocol.matter


def _wanted_sensors(richness: str, archetype: str, rooms: Sequence[Room],
                    loads: Sequence[Load]) -> list[tuple[str, str]]:
    """(room id, load class) of the sensors a richness level adds, in draw order.

    A room never gets a second sensor of a class it already has, from the inventory or from this
    list. The main door is the living room's, or the lobby's in a villa.
    """
    have = {(ld.room_id, ld.load_class) for ld in loads}
    lit = {ld.room_id for ld in loads if ld.is_main_light}
    wanted: list[tuple[str, str]] = []

    def add(room_id: str, load_class: str) -> None:
        if (room_id, load_class) not in have:
            have.add((room_id, load_class))
            wanted.append((room_id, load_class))

    def of_types(*room_types: str) -> list[str]:
        return [r.room_id for r in rooms if r.room_type in room_types]

    door = of_types("lobby" if archetype == "villa" else "living")[0]
    for room_id in of_types("passage", "staircase", "bathroom"):
        add(room_id, "sensor.motion")
    add(door, "sensor.contact")
    if richness == "rich":
        for room_id in of_types("living", "bedroom"):
            add(room_id, "sensor.presence")
        for room in rooms:
            if room.room_type in ("living", "bedroom") or room.room_id not in lit:
                continue
            if (room.room_id, "sensor.presence") not in have:
                add(room.room_id, "sensor.motion")
        add(of_types("living")[0], "sensor.lux")
        for room_id in of_types("bedroom"):
            add(room_id, "sensor.temperature")
            add(room_id, "sensor.humidity")
        utility = of_types("utility")
        add(utility[0] if utility else of_types("living")[0], "meter.energy")
    return wanted


def sensor_loads(richness: str, archetype: str, rooms: Sequence[Room], loads: Sequence[Load],
                 rng: np.random.Generator) -> list[Load]:
    """The sensor loads a richness level adds to a home whose other loads are ``loads``.

    Called after every other draw of the home, so the other loads never depend on richness.
    One protocol draw per added sensor, in the order of ``_wanted_sensors``. A load keeps the
    counter after the room's last load, so load ids stay unique within a room.
    """
    if richness not in ("none", "some", "rich"):
        raise ValueError(f"unknown sensor richness: {richness!r}")
    if richness == "none":
        return []
    counter = Counter(ld.room_id for ld in loads)
    added: list[Load] = []
    for room_id, load_class in _wanted_sensors(richness, archetype, rooms, loads):
        protocol = Protocol.zigbee if rng.random() < ZIGBEE_SENSOR_P else Protocol.matter
        k = counter[room_id]
        counter[room_id] += 1
        added.append(Load(
            load_id=f"{room_id}:{load_class}:{k}", room_id=room_id, load_class=load_class,
            protocol=protocol, system=_SENSOR_SYSTEM[protocol], circuit="none",
        ))
    return added
