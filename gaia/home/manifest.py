"""Home manifest: rooms, room adjacency and the loads installed in each room."""

import json
from dataclasses import asdict, dataclass, field

from gaia.schema.events import Protocol


@dataclass(frozen=True)
class Room:
    room_id: str
    room_type: str
    window_factor: float


@dataclass(frozen=True)
class Load:
    load_id: str
    room_id: str
    load_class: str
    protocol: Protocol
    is_main_light: bool


@dataclass(frozen=True)
class HomeManifest:
    home_id: str
    archetype: str
    city: str
    rooms: tuple[Room, ...]
    # dict is unhashable; excluded from hash, still compared for equality.
    adjacency: dict[str, tuple[str, ...]] = field(hash=False)
    loads: tuple[Load, ...]

    def to_json(self) -> str:
        payload = {
            "home_id": self.home_id,
            "archetype": self.archetype,
            "city": self.city,
            "rooms": [asdict(r) for r in self.rooms],
            "adjacency": {k: list(v) for k, v in self.adjacency.items()},
            "loads": [
                {
                    "load_id": ld.load_id,
                    "room_id": ld.room_id,
                    "load_class": ld.load_class,
                    "protocol": ld.protocol.value,
                    "is_main_light": ld.is_main_light,
                }
                for ld in self.loads
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def loads_in(self, room_id: str) -> tuple[Load, ...]:
        return tuple(ld for ld in self.loads if ld.room_id == room_id)

    def main_light(self, room_id: str) -> Load | None:
        for ld in self.loads:
            if ld.room_id == room_id and ld.is_main_light:
                return ld
        return None
