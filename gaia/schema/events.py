"""MSI Event Schema v0 (spec section 4.1), shared by GAIA and the on-board logger."""

from dataclasses import dataclass
from enum import Enum

import pyarrow as pa


class EventKind(str, Enum):
    command = "command"
    state = "state"
    scene_fired = "scene_fired"
    schedule_fired = "schedule_fired"
    automation_edit = "automation_edit"
    sensor = "sensor"
    system = "system"
    msi_proposal = "msi_proposal"
    msi_action = "msi_action"
    user_feedback = "user_feedback"


class Source(str, Enum):
    wall = "wall"
    app = "app"
    voice = "voice"
    homekit = "homekit"
    scene = "scene"
    schedule = "schedule"
    msi = "msi"
    system = "system"
    unknown = "unknown"


class Protocol(str, Enum):
    knx = "knx"
    matter = "matter"
    zigbee = "zigbee"
    wifi = "wifi"
    lutron = "lutron"
    dali = "dali"
    ip = "ip"
    ir = "ir"
    other = "other"


class Domain(str, Enum):
    synthetic = "synthetic"
    real = "real"
    public = "public"


@dataclass(frozen=True, slots=True)
class Event:
    ts_utc: int
    tz_offset_min: int
    home_id: str
    event_id: str
    kind: EventKind
    room_id: str | None
    room_type: str | None
    load_id: str | None
    load_class: str | None
    capability: str | None
    value: float | str | None
    prev_value: float | str | None
    source: Source
    actor: str
    protocol: Protocol | None
    ref_id: str | None
    domain: Domain


EVENT_ARROW_SCHEMA = pa.schema(
    [
        pa.field("ts_utc", pa.int64(), nullable=False),
        pa.field("tz_offset_min", pa.int16(), nullable=False),
        pa.field("home_id", pa.string(), nullable=False),
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("room_id", pa.string(), nullable=True),
        pa.field("room_type", pa.string(), nullable=True),
        pa.field("load_id", pa.string(), nullable=True),
        pa.field("load_class", pa.string(), nullable=True),
        pa.field("capability", pa.string(), nullable=True),
        pa.field("value", pa.string(), nullable=True),
        pa.field("prev_value", pa.string(), nullable=True),
        pa.field("source", pa.string(), nullable=False),
        pa.field("actor", pa.string(), nullable=False),
        pa.field("protocol", pa.string(), nullable=True),
        pa.field("ref_id", pa.string(), nullable=True),
        pa.field("domain", pa.string(), nullable=False),
    ]
)


def _value_to_str(value: float | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def _enum_to_str(member: Enum | None) -> str | None:
    return None if member is None else member.value


def events_to_table(events: list[Event]) -> pa.Table:
    """Convert events to an Arrow table matching EVENT_ARROW_SCHEMA. Built column by column."""
    columns = {
        "ts_utc": [ev.ts_utc for ev in events],
        "tz_offset_min": [ev.tz_offset_min for ev in events],
        "home_id": [ev.home_id for ev in events],
        "event_id": [ev.event_id for ev in events],
        "kind": [ev.kind.value for ev in events],
        "room_id": [ev.room_id for ev in events],
        "room_type": [ev.room_type for ev in events],
        "load_id": [ev.load_id for ev in events],
        "load_class": [ev.load_class for ev in events],
        "capability": [ev.capability for ev in events],
        "value": [_value_to_str(ev.value) for ev in events],
        "prev_value": [_value_to_str(ev.prev_value) for ev in events],
        "source": [ev.source.value for ev in events],
        "actor": [ev.actor for ev in events],
        "protocol": [_enum_to_str(ev.protocol) for ev in events],
        "ref_id": [ev.ref_id for ev in events],
        "domain": [ev.domain.value for ev in events],
    }
    arrays = [pa.array(columns[f.name], type=f.type) for f in EVENT_ARROW_SCHEMA]
    return pa.Table.from_arrays(arrays, schema=EVENT_ARROW_SCHEMA)
