"""Room inventory v1: loads per room by tier, lighting-system weights and protocol assignment.

inventory.yaml is validated at import. Room draws come only from the rng passed in: one weighted
choice for the room's main light, then each other entry in YAML order (presence, then count).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import TypeVar

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from gaia.registry.registry import FIXTURES, REGISTRY, _StrictLoader, get_load_class
from gaia.schema.events import Protocol

TIERS = ("basic", "standard", "premium", "luxury")
CIRCUITS = frozenset({"main", "accent", "cove", "task", "night", "decor"})
MAIN_CLASSES = frozenset({"light.onoff", "light.dimmable", "light.tunable"})
DALI_CLASSES = frozenset({"light.tunable", "light.rgbw", "light.rgbcct"})
IR_CLASSES = frozenset({"climate.split_ac", "climate.cassette_ac"})
IR_SHARE = 0.6
WIFI_SHARE = 0.5

# Lighting system per tier, as (system, weight). Drawn once per home.
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


class _EntrySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    load_class: str = Field(alias="class")
    fixture: str | None
    circuit: str
    count: tuple[int, int]
    p: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _check(self):
        if self.load_class not in REGISTRY:
            raise ValueError(f"unknown load class {self.load_class!r}")
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
                parsed.append(InventoryEntry(spec.load_class, spec.fixture, spec.circuit, spec.count, spec.p))
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


def choose_entries(tier: str, room_type: str, rng: np.random.Generator) -> list[InventoryEntry]:
    """Return the loads for one room, main circuit first, then the other entries in YAML order."""
    try:
        entries = INVENTORY[tier][room_type]
    except KeyError:
        raise KeyError(f"no inventory for tier {tier!r} room {room_type!r}") from None
    mains = [(e, e.p) for e in entries if e.circuit == "main"]
    chosen = [weighted_choice(mains, rng)]
    for e in entries:
        if e.circuit == "main":
            continue
        if rng.random() < e.p:
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
