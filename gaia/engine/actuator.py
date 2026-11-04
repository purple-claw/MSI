"""Actuator: applies intents to device state and emits command, state and sensor events.

Each applied command yields a command event, a state event referencing it, and one Label.
Timestamps follow the G1 engine: command = tick + engine-rng jitter, state = command +
protocol latency, and both are serialised per load so a load never goes back in time.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np

from gaia.devices.state import HomeDevices
from gaia.engine.protocols import latency_ms
from gaia.home.manifest import HomeManifest, Load
from gaia.household.personas import Persona
from gaia.interaction.choose import choose_source
from gaia.labels.model import SCHEMA_VERSION, Label, validate_label
from gaia.schema.events import Domain, Event, EventKind, Source

MS_PER_MIN = 60_000
SYSTEM_ACTOR = "system"
SENSOR_INTENT = "system"
SENSOR_TRIGGER = "presence_tick"
SENSOR_SOURCE_REF = "emit_sensor"


@dataclass(frozen=True)
class Intent:
    load_id: str
    capability: str
    value: float | str
    person_id: str | None
    activity: str
    intent: str
    trigger: str
    device_action: str
    policy: str  # source_ref of the emitted labels
    source: Source | None = None  # None: draw from the person's persona, or system without one


class Actuator:
    def __init__(
        self,
        home_id: str,
        manifest: HomeManifest,
        devices: HomeDevices,
        tz_offset_min: int,
        rng_engine: np.random.Generator,
        rng_interaction: np.random.Generator,
        rng_protocol: np.random.Generator,
        personas_by_id: Mapping[str, Persona],
        next_id: Callable[[], str],
    ) -> None:
        self._home_id = home_id
        self._loads = {ld.load_id: ld for ld in manifest.loads}
        self._rooms = {room.room_id: room for room in manifest.rooms}
        self._devices = devices
        self._tz_offset_min = tz_offset_min
        self._rng_engine = rng_engine
        self._rng_interaction = rng_interaction
        self._rng_protocol = rng_protocol
        self._personas = personas_by_id
        self._next_id = next_id
        self._last_ts: dict[str, int] = {}
        self._events: list[Event] = []
        self._labels: list[Label] = []

    @property
    def events(self) -> list[Event]:
        return self._events

    @property
    def labels(self) -> list[Label]:
        return self._labels

    def apply(self, intents: list[Intent], tick_start_ms: int) -> None:
        """Apply intents for one tick. The first intent per (load, capability) wins."""
        seen: set[tuple[str, str]] = set()
        for intent in intents:
            key = (intent.load_id, intent.capability)
            if key in seen:
                continue
            seen.add(key)
            self._apply_one(intent, tick_start_ms)

    def emit_sensor(self, load_id: str, capability: str, value: float, tick_ms: int) -> None:
        """Report a sensor reading. Sensors report even when the value is unchanged."""
        load = self._load(load_id)
        state = self._devices.get(load_id)
        prev = state.get(capability)
        new, clamped = state.normalise(capability, value)
        state.set(capability, new)
        ts = self._serial_ts(load_id, tick_ms + self._jitter())
        event_id = self._next_id()
        self._events.append(self._event(
            EventKind.sensor, load, ts, capability, new, prev, Source.system, SYSTEM_ACTOR, None, event_id,
        ))
        self._last_ts[load_id] = ts
        self._record(event_id, None, load, "none", SENSOR_INTENT, SENSOR_TRIGGER, "none",
                     SENSOR_SOURCE_REF, clamped)

    # --- internals ------------------------------------------------------------

    def _apply_one(self, intent: Intent, tick: int) -> None:
        load = self._load(intent.load_id)
        state = self._devices.get(intent.load_id)
        if not state.would_change(intent.capability, intent.value):
            return  # a redundant intent must not switch an off device on
        source = intent.source if intent.source is not None else self._draw_source(intent.person_id)
        if state.requires_power(intent.capability) and not state.is_on():
            power_on = replace(intent, capability="power", value=1.0, device_action="turn_on")
            self._command(power_on, source, load, tick)
        self._command(intent, source, load, tick)

    def _command(self, intent: Intent, source: Source, load: Load, tick: int) -> None:
        result = self._devices.get(load.load_id).set(intent.capability, intent.value)
        if result is None:
            return
        new, prev, clamped = result
        cmd_ts = self._serial_ts(load.load_id, tick + self._jitter())
        cmd_id = self._next_id()
        actor = intent.person_id if intent.person_id is not None else SYSTEM_ACTOR
        self._events.append(self._event(
            EventKind.command, load, cmd_ts, intent.capability, new, prev, source, actor, None, cmd_id,
        ))
        state_ts = self._serial_ts(load.load_id, cmd_ts + latency_ms(load.protocol, self._rng_protocol))
        self._events.append(self._event(
            EventKind.state, load, state_ts, intent.capability, new, prev, source, actor, cmd_id,
            self._next_id(),
        ))
        self._last_ts[load.load_id] = state_ts
        self._record(cmd_id, intent.person_id, load, intent.activity, intent.intent, intent.trigger,
                     intent.device_action, intent.policy, clamped)

    def _record(self, event_id: str, person_id: str | None, load: Load, activity: str, intent: str,
                trigger: str, device_action: str, source_ref: str, clamped: bool) -> None:
        label = Label(
            event_id=event_id, person_id=person_id, room_id=load.room_id, activity=activity,
            intent=intent, trigger=trigger, device_action=device_action, source_ref=source_ref,
            schema_version=SCHEMA_VERSION, clamped=clamped,
        )
        validate_label(label)
        self._labels.append(label)

    def _event(self, kind: EventKind, load: Load, ts: int, capability: str, value, prev, source: Source,
               actor: str, ref_id: str | None, event_id: str) -> Event:
        room = self._rooms[load.room_id]
        return Event(
            ts_utc=ts, tz_offset_min=self._tz_offset_min, home_id=self._home_id, event_id=event_id,
            kind=kind, room_id=room.room_id, room_type=room.room_type, load_id=load.load_id,
            load_class=load.load_class, capability=capability, value=value, prev_value=prev,
            source=source, actor=actor, protocol=load.protocol, ref_id=ref_id,
            domain=Domain.synthetic,
        )

    def _serial_ts(self, load_id: str, ts: int) -> int:
        return max(ts, self._last_ts.get(load_id, -1) + 1)

    def _jitter(self) -> int:
        return int(self._rng_engine.integers(0, MS_PER_MIN))

    def _draw_source(self, person_id: str | None) -> Source:
        if person_id is None:
            return Source.system
        return choose_source(self._personas[person_id], self._rng_interaction)

    def _load(self, load_id: str) -> Load:
        try:
            return self._loads[load_id]
        except KeyError:
            raise KeyError(f"unknown load: {load_id!r}") from None
