"""Import anonymised home layouts from converted ETS projects (KNXdata.json).

A layout keeps only room types and (class, fixture, circuit) per load. No project, room or load
name survives: the project id is a hash of the file bytes. Rooms are typed from the ETS room
name; a room grouped by function ("Switching", "Dimming") is split by the room word found in
each load's own name.

build_ets_home turns a layout into a HomeManifest. ETS covers what a KNX integrator wires
(lighting, shading, HVAC, fans, keypads); the inventory adds everything else per room (media,
appliances), and the living room, kitchen and one bathroom when the project has none.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from gaia.home.archetypes import _WINDOW, ArchetypeSpec
from gaia.home.inventory import (
    INVENTORY,
    SENSOR_RICHNESS,
    assign_protocol,
    choose_entries,
    sensor_loads,
    system_protocol,
    weighted_choice,
)
from gaia.home.manifest import HomeManifest, Load, Room
from gaia.registry.registry import get_load_class
from gaia.schema.events import Protocol

LIGHTING_SYSTEM = "knx_dali"
ETS_TIER = "luxury"  # assumption: KNX projects are premium and luxury installs
_FALLBACK_TIERS = ("luxury", "premium", "standard")
RESIDENTIAL_TYPES = frozenset(_WINDOW)
_ETS_DOMAINS = frozenset({"lighting", "shading"})  # inventory never adds these to an ETS room


def _compile(rows: list[list[str]]) -> tuple[tuple[re.Pattern[str], str], ...]:
    return tuple((re.compile(pat, re.IGNORECASE), out) for pat, out in rows)


_MAPS = yaml.safe_load((Path(__file__).with_name("maps.yaml")).read_text(encoding="utf-8"))
_ROOMS = _compile(_MAPS["rooms"])
_SWITCH = _compile(_MAPS["switch_keywords"])
_FIXTURES = _compile(_MAPS["fixtures"])
_LOAD_TYPES: dict[str, str] = _MAPS["load_types"]
_FIXTURE_CIRCUIT: dict[str, str] = _MAPS["fixture_circuit"]
_DEFAULT_FIXTURE: dict[str, str] = _MAPS["default_fixture"]
_SWITCH_DEFAULT: str = _MAPS["switch_default"]


def _first(rules: tuple[tuple[re.Pattern[str], str], ...], text: str) -> tuple[str, str] | None:
    """(output, matched text) of the first rule matching text."""
    for pat, out in rules:
        m = pat.search(text)
        if m:
            return out, m.group(0).lower()
    return None


def room_type_for(name: str) -> str | None:
    hit = _first(_ROOMS, name)
    return hit[0] if hit else None


def classify_load(ld_type: str, name: str) -> tuple[str, str | None, str] | None:
    """(class, fixture, circuit) of an ETS load, or None when the type is not mapped."""
    if ld_type == "Switch":
        hit = _first(_SWITCH, name)
        cls = hit[0] if hit else _SWITCH_DEFAULT
    elif ld_type in _LOAD_TYPES:
        cls = _LOAD_TYPES[ld_type]
    else:
        return None
    if get_load_class(cls).domain != "lighting":
        return cls, None, "none"
    hit = _first(_FIXTURES, name)
    fixture = hit[0] if hit else _DEFAULT_FIXTURE.get(cls, "bulb")
    return cls, fixture, _FIXTURE_CIRCUIT.get(fixture, "accent")


@dataclass(frozen=True)
class EtsLayout:
    project_id: str
    rooms: tuple[tuple[str, tuple[tuple[str, str | None, str], ...]], ...]
    stats: dict = field(hash=False, compare=False)

    def to_json(self) -> str:
        payload = {
            "project_id": self.project_id,
            "rooms": [{"type": t, "loads": [list(ld) for ld in loads]} for t, loads in self.rooms],
            "stats": self.stats,
        }
        return json.dumps(payload, sort_keys=True, indent=1)

    @classmethod
    def from_json(cls, text: str) -> EtsLayout:
        d = json.loads(text)
        rooms = tuple((r["type"], tuple(tuple(ld) for ld in r["loads"])) for r in d["rooms"])
        return cls(d["project_id"], rooms, d["stats"])


def import_knxdata(path: Path, project_id: str | None = None) -> EtsLayout:
    raw = Path(path).read_bytes()
    pid = project_id or hashlib.sha256(raw).hexdigest()[:12]
    data = json.loads(raw)
    loads = data.get("loads", [])
    owner: dict[int, int] = {}
    for r_idx, room in enumerate(data.get("rooms", [])):
        for ref in room.get("loads", []):
            owner.setdefault(int(ref) - 1, r_idx)  # KNXdata load refs are 1-based
    room_names = [r.get("name", "") for r in data.get("rooms", [])]

    groups: dict[tuple[int, str, str], list[tuple[str, str | None, str]]] = defaultdict(list)
    unknown = unmapped = 0
    for i, ld in enumerate(loads):
        name = ld.get("ldNm", "")
        r_idx = owner.get(i, -1)
        hit = _first(_ROOMS, room_names[r_idx]) if r_idx >= 0 else None
        key_word = "room"
        if hit is None:  # function-grouped or unassigned: use the load's own room word
            hit = _first(_ROOMS, name)
            key_word = hit[1] if hit else ""
        kind = classify_load(ld.get("ldTyp", ""), name)
        if kind is None:
            unknown += 1
            continue
        if hit is None:
            unmapped += 1
            continue
        groups[(r_idx, hit[0], key_word)].append(kind)

    rooms = tuple((key[1], tuple(v)) for key, v in groups.items())
    stats = {
        "loads_total": len(loads),
        "loads_unknown_type": unknown,
        "loads_unmapped_room": unmapped,
        "loads_kept": sum(len(v) for _, v in rooms),
        "rooms": len(rooms),
        "room_types": dict(Counter(t for t, _ in rooms)),
        "classes": dict(Counter(ld[0] for _, v in rooms for ld in v)),
    }
    return EtsLayout(pid, rooms, stats)


def is_residential(layout: EtsLayout) -> bool:
    """>= 4 rooms of residential types, >= 60% of loads kept, no room with > 50% of the loads."""
    total = layout.stats.get("loads_total", 0)
    if total == 0:
        return False
    if sum(1 for t, _ in layout.rooms if t in RESIDENTIAL_TYPES) < 4:
        return False
    if layout.stats["loads_kept"] < 0.6 * total:
        return False
    return all(len(v) <= 0.5 * total for _, v in layout.rooms)


def inventory_stats(layouts: list[EtsLayout]) -> dict[str, dict[str, float]]:
    """Mean count of each class per room, by room type (for manual inventory calibration)."""
    n_rooms: Counter[str] = Counter()
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for lay in layouts:
        for rtype, loads in lay.rooms:
            n_rooms[rtype] += 1
            counts[rtype].update(ld[0] for ld in loads)
    return {
        t: {c: round(n / n_rooms[t], 2) for c, n in sorted(counts[t].items())}
        for t in sorted(counts)
    }


@lru_cache(maxsize=8)
def load_layouts(ets_dir: str) -> tuple[EtsLayout, ...]:
    """Residential layouts in ets_dir (<project_id>.json files), sorted by file name."""
    d = Path(ets_dir)
    if not d.is_dir():
        raise ValueError(f"ets_dir {ets_dir!r} is not a directory (set ets_dir in the run config)")
    out = []
    for p in sorted(d.glob("*.json")):
        if p.name == "summary.json":
            continue
        lay = EtsLayout.from_json(p.read_text(encoding="utf-8"))
        if is_residential(lay):
            out.append(lay)
    if not out:
        raise ValueError(f"ets_dir {ets_dir!r} holds no residential ETS layouts")
    return tuple(out)


def _inventory_entries(room_name: str, rng: np.random.Generator):
    rtype = room_name.rsplit("_", 1)[0] if room_name[-1].isdigit() else room_name
    for tier in _FALLBACK_TIERS:
        if rtype in INVENTORY[tier]:
            return choose_entries(tier, "villa", room_name, rng)
    return []


def _ets_protocol(cls: str) -> Protocol:
    if get_load_class(cls).domain in _ETS_DOMAINS:
        return system_protocol(LIGHTING_SYSTEM, cls)
    return Protocol.knx


def _name_rooms(types: list[str]) -> list[str]:
    seen: Counter[str] = Counter()
    names = []
    for t in types:
        seen[t] += 1
        if t in ("bedroom", "bathroom"):
            names.append(f"{t}_{seen[t]}")
        else:
            names.append(t if seen[t] == 1 else f"{t}_{seen[t]}")
    return names


def build_ets_home(ets_dir: str | Path | None, home_id: str, city: str,
                   rng: np.random.Generator) -> tuple[HomeManifest, ArchetypeSpec]:
    if not ets_dir:
        raise ValueError("archetype 'ets' needs ets_dir set in the run config")
    layouts = load_layouts(str(ets_dir))
    layout = layouts[int(rng.integers(len(layouts)))]

    ets_rooms = list(layout.rooms)
    present = {t for t, _ in ets_rooms}
    for must in ("living", "kitchen", "bathroom"):
        if must not in present:
            ets_rooms.append((must, ()))
    order = {"living": 0, "kitchen": 1}
    ets_rooms.sort(key=lambda r: order.get(r[0], 2))  # stable: ETS order otherwise
    names = _name_rooms([t for t, _ in ets_rooms])

    rooms: list[Room] = []
    loads: list[Load] = []
    for (rtype, ets_loads), name in zip(ets_rooms, names):
        room_id = f"{home_id}:{name}"
        rooms.append(Room(room_id, rtype, _WINDOW[rtype]))
        counter = 0

        def add(cls: str, fixture: str | None, circuit: str, system: str, protocol: Protocol) -> None:
            nonlocal counter
            loads.append(Load(f"{room_id}:{cls}:{counter}", room_id, cls, protocol, system,
                              circuit, fixture))
            counter += 1

        has_main = False
        for cls, fixture, circuit in ets_loads:
            if circuit == "main":
                if has_main:
                    circuit = "accent"  # one ambient circuit per room; other downlight groups layer
                has_main = True
            add(cls, fixture, circuit, LIGHTING_SYSTEM, _ets_protocol(cls))
        if ets_loads and not has_main:
            lights = [i for i, ld in enumerate(loads) if ld.room_id == room_id
                      and get_load_class(ld.load_class).domain == "lighting"]
            if lights:  # promote the room's first light to its ambient circuit
                i = lights[0]
                loads[i] = Load(loads[i].load_id, room_id, loads[i].load_class, loads[i].protocol,
                                loads[i].system, "main", loads[i].fixture)
            else:
                add("light.onoff", "bulb", "main", LIGHTING_SYSTEM, Protocol.knx)

        ets_classes = {c for c, _, _ in ets_loads}
        for entry in _inventory_entries(name, rng):
            domain = get_load_class(entry.load_class).domain
            if ets_loads and (domain in _ETS_DOMAINS or entry.load_class in ets_classes):
                continue
            if ets_loads and domain == "climate" and entry.load_class.startswith("climate.") \
                    and any(c.startswith("climate.") for c in ets_classes):
                continue
            system, protocol = assign_protocol(entry.load_class, LIGHTING_SYSTEM, rng)
            add(entry.load_class, entry.fixture, entry.circuit, system, protocol)

    richness = weighted_choice(SENSOR_RICHNESS, rng)
    door_arch = "villa" if any(r.room_type == "lobby" for r in rooms) else "ets"
    loads.extend(sensor_loads(richness, door_arch, rooms, loads, rng))

    hub = f"{home_id}:living"
    adjacency = {
        r.room_id: (tuple(x.room_id for x in rooms if x.room_id != hub) if r.room_id == hub else (hub,))
        for r in rooms
    }
    manifest = HomeManifest(
        home_id=home_id, archetype="ets", city=city, tier=ETS_TIER,
        lighting_system=LIGHTING_SYSTEM, rooms=tuple(rooms), adjacency=adjacency,
        loads=tuple(loads), sensor_richness=richness,
    )
    bedrooms = sum(1 for r in rooms if r.room_type == "bedroom")
    bathrooms = sum(1 for r in rooms if r.room_type == "bathroom")
    extras = tuple(n for n in names if not n.startswith(("bedroom_", "bathroom_"))
                   and n not in ("living", "kitchen"))
    spec = ArchetypeSpec("ets", bedrooms, bathrooms, True,
                         (2, max(2, min(7, bedrooms + 2))), extras, {ETS_TIER: 1.0})
    return manifest, spec
