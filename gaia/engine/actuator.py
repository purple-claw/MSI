"""Actuator: applies intents to device state and emits command, state and sensor events.

Each applied command yields a command event, a state event referencing it, and one Label.
Timestamps follow the G1 engine: command = tick + jitter, state = command + protocol latency,
and both are serialised per load so a load never goes back in time.

Randomness is keyed by the emitting policy: each policy name gets its own jitter ("engine"),
source ("interaction") and latency ("protocol") streams, derived from the home seed. So adding
a policy that emits intents on other loads leaves every other policy's events unchanged.
Event ids are "<home>:<policy>:<counter>", counted per policy, for the same reason.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from gaia.devices.state import HomeDevices
from gaia.engine.protocols import latency_ms
from gaia.home.manifest import HomeManifest, Load
from gaia.household.personas import Persona
from gaia.interaction.choose import choose_source
from gaia.labels.model import SCHEMA_VERSION, Label, validate_label
from gaia.rng import rng_for
from gaia.schema.events import Domain, Event, EventKind, Source

MS_PER_MIN = 60_000
SYSTEM_ACTOR = "system"
SENSOR_INTENT = "system"
SENSOR_TRIGGER = "presence_tick"
SENSOR_SOURCE_REF = "emit_sensor"

# Called as (load_id, capability, state_ts_ms) whenever a state event is emitted.
StateCallback = Callable[[str, str, int], None]


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
    policy: str  # the emitting policy; also the source_ref of its labels
    source: Source | None = None  # None: draw from the person's persona, or system without one


@dataclass(frozen=True)
class SensorReading:
    """A sensor report for one capability, sent through ``Actuator.emit_sensor``."""
    load_id: str
    capability: str
    value: float


@dataclass(frozen=True)
class ActuatorStreams:
    engine: np.random.Generator
    interaction: np.random.Generator
    protocol: np.random.Generator


class Actuator:
    def __init__(
        self,
        home_id: str,
        manifest: HomeManifest,
        devices: HomeDevices,
        tz_offset_min: int,
        home_seed: int,
        personas_by_id: Mapping[str, Persona],
        on_state: StateCallback | None = None,
    ) -> None:
        self._home_id = home_id
        self._loads = {ld.load_id: ld for ld in manifest.loads}
        self._rooms = {room.room_id: room for room in manifest.rooms}
        self._devices = devices
        self._tz_offset_min = tz_offset_min
        self._home_seed = home_seed
        self._personas = personas_by_id
        self._on_state = on_state
        self._streams: dict[str, ActuatorStreams] = {}
        self._counters: dict[str, int] = {}
        self._last_ts: dict[str, int] = {}
        self._events: list[Event] = []
        self._labels: list[Label] = []
        self._checked_labels: set[tuple[str, str, str, str]] = set()  # enum tuples already validated

    @property
    def events(self) -> list[Event]:
        return self._events

    @property
    def labels(self) -> list[Label]:
        return self._labels

    def streams_for(self, policy: str) -> ActuatorStreams:
        """The policy's own jitter, source and latency streams (created on first use)."""
        streams = self._streams.get(policy)
        if streams is None:
            streams = ActuatorStreams(
                engine=rng_for(self._home_seed, f"actuator/{policy}/engine"),
                interaction=rng_for(self._home_seed, f"actuator/{policy}/interaction"),
                protocol=rng_for(self._home_seed, f"actuator/{policy}/protocol"),
            )
            self._streams[policy] = streams
        return streams

    def apply(self, items: list[Intent | SensorReading], tick_start_ms: int) -> None:
        """Apply one tick's intents and sensor readings in order.

        The first intent per (load, capability) wins, even if it turns out redundant: a later
        intent for the same pair is dropped unseen, so a policy must not propose a no-op ahead
        of the change it means. A redundant intent (the device already holds the value) emits
        nothing and draws from no stream. An intent whose ``source`` is None draws one source
        from its policy's interaction stream, or none for a system intent. A reading is never
        deduplicated: each one emits a sensor event through ``emit_sensor``.

        Unknown person ids raise ValueError before any item of the tick is applied.
        """
        intents = [item for item in items if isinstance(item, Intent)]
        for intent in intents:
            if intent.person_id is not None and intent.person_id not in self._personas:
                raise ValueError(
                    f"unknown person_id {intent.person_id!r} in intent for load {intent.load_id!r}")
        seen: set[tuple[str, str]] = set()
        for item in items:
            if isinstance(item, SensorReading):
                self.emit_sensor(item.load_id, item.capability, item.value, tick_start_ms)
                continue
            key = (item.load_id, item.capability)
            if key in seen:
                continue
            seen.add(key)
            self._apply_one(item, tick_start_ms)

    def emit_sensor(self, load_id: str, capability: str, value: float, tick_ms: int) -> None:
        """Report a sensor reading. Sensors report even when the value is unchanged."""
        load = self._load(load_id)
        state = self._devices.get(load_id)
        prev = state.get(capability)
        new, clamped = state.normalise(capability, value)
        state.set(capability, new)
        ts = self._serial_ts(load_id, tick_ms + self._jitter(SENSOR_SOURCE_REF))
        event_id = self._next_id(SENSOR_SOURCE_REF)
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
        source = intent.source if intent.source is not None else self._draw_source(intent)
        if state.requires_power(intent.capability) and not state.is_on():
            power_on = Intent(
                load_id=intent.load_id, capability="power", value=1.0, person_id=intent.person_id,
                activity=intent.activity, intent=intent.intent, trigger=intent.trigger,
                device_action="turn_on", policy=intent.policy, source=intent.source,
            )
            self._command(power_on, source, load, tick)
        self._command(intent, source, load, tick)

    def _command(self, intent: Intent, source: Source, load: Load, tick: int) -> None:
        result = self._devices.get(load.load_id).set(intent.capability, intent.value)
        if result is None:
            return
        streams = self.streams_for(intent.policy)
        new, prev, clamped = result
        cmd_ts = self._serial_ts(load.load_id, tick + self._jitter(intent.policy))
        cmd_id = self._next_id(intent.policy)
        actor = intent.person_id if intent.person_id is not None else SYSTEM_ACTOR
        self._events.append(self._event(
            EventKind.command, load, cmd_ts, intent.capability, new, prev, source, actor, None, cmd_id,
        ))
        state_ts = self._serial_ts(load.load_id, cmd_ts + latency_ms(load.protocol, streams.protocol))
        self._events.append(self._event(
            EventKind.state, load, state_ts, intent.capability, new, prev, source, actor, cmd_id,
            self._next_id(intent.policy),
        ))
        self._last_ts[load.load_id] = state_ts
        self._record(cmd_id, intent.person_id, load, intent.activity, intent.intent, intent.trigger,
                     intent.device_action, intent.policy, clamped)
        if self._on_state is not None:
            self._on_state(load.load_id, intent.capability, state_ts)

    def _record(self, event_id: str, person_id: str | None, load: Load, activity: str, intent: str,
                trigger: str, device_action: str, source_ref: str, clamped: bool) -> None:
        label = Label(
            event_id=event_id, person_id=person_id, room_id=load.room_id, activity=activity,
            intent=intent, trigger=trigger, device_action=device_action, source_ref=source_ref,
            schema_version=SCHEMA_VERSION, clamped=clamped,
        )
        # validate_label depends only on the enum fields, so each combination is checked once.
        key = (activity, intent, trigger, device_action)
        if key not in self._checked_labels:
            validate_label(label)
            self._checked_labels.add(key)
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

    def _next_id(self, policy: str) -> str:
        n = self._counters.get(policy, 0)
        self._counters[policy] = n + 1
        return f"{self._home_id}:{policy}:{n:09d}"

    def _serial_ts(self, load_id: str, ts: int) -> int:
        return max(ts, self._last_ts.get(load_id, -1) + 1)

    def _jitter(self, policy: str) -> int:
        return int(self.streams_for(policy).engine.integers(0, MS_PER_MIN))

    def _draw_source(self, intent: Intent) -> Source:
        if intent.person_id is None:
            return Source.system
        return choose_source(self._personas[intent.person_id],
                             self.streams_for(intent.policy).interaction)

    def _load(self, load_id: str) -> Load:
        try:
            return self._loads[load_id]
        except KeyError:
            raise KeyError(f"unknown load: {load_id!r}") from None
