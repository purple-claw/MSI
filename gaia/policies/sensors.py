"""Sensors policy: motion, presence, the main door contact, lux, temperature, humidity and energy.

Sensors never switch a load, so no other policy reads them. Every reading is a SensorReading the
actuator sends through ``emit_sensor``: a sensor event, and a label with person None, activity
none, intent system, trigger presence_tick. Sensor events never carry waste.

- Motion (sensor.motion, one per room): 1 when an awake person enters the room, and again every
  2-5 minutes (drawn from the policy rng, per room) while an awake person is there. 0 once the
  room has had nobody awake for 2 minutes. Sleepers never trigger it.
- Presence (sensor.presence): 1 while anyone, asleep or awake, is in the room; 0 at the first
  empty minute. Reported on change only, plus once at minute 0 for a room occupied at the start.
- Contact (sensor.contact, the main door): when someone leaves home or comes back, open (1) and
  then closed (0) one minute later. One pair at a time: an event while a pair is pending is absorbed.
- Lux (sensor.lux, in living): every 15 minutes, outdoor lux times the room's window factor, plus
  150 lx per main-circuit light on in the room (assumption).
- Temperature (sensor.temperature): every 15 minutes, a toy model. The base is 24 + 12 x heat index
  (assumption). An AC on in the room moves the reading towards its setpoint, and with no AC on the
  reading moves towards the base; both at 0.4 C per minute. The state is kept per room, and the AC
  state at a reading applies to the minutes since the room's previous reading.
- Humidity (sensor.humidity): every 15 minutes, 0.4 + 0.4 x heat index, clamped to 0-1 (assumption).
- Energy meter (meter.energy): every 5 minutes, the home's active power in W: each load's power_w
  "on" when it is on, its power_w "standby" when it is off and its class has one, else 0.

Readings run before notice in POLICY_ORDER, so they see this minute's commands but not the
switch-offs notice makes later in the same minute.
"""

from collections import deque

import numpy as np

from gaia.engine.actuator import SensorReading
from gaia.engine.context import TickContext
from gaia.home.manifest import HomeManifest, Load
from gaia.registry.registry import get_load_class
from gaia.world.season import heat_index

MOTION_HOLD_MIN = 2
MOTION_RETRIGGER_MIN = (2, 5)
CONTACT_HOLD_MIN = 1
LIGHT_LUX_PER_MAIN = 150.0  # assumption: lux added by each main-circuit light that is on
TEMP_BASE_C = 24.0
TEMP_SPAN_C = 12.0
TEMP_STEP_C = 0.4  # per minute, towards the setpoint or the base
HUMIDITY_BASE = 0.4
HUMIDITY_SPAN = 0.4
READING_EVERY_MIN = 15
ENERGY_EVERY_MIN = 5
AC_CLASSES = frozenset({"climate.split_ac", "climate.cassette_ac", "climate.vrf_indoor"})
OCCUPANCY = "occupancy"


class Sensors:
    name = "sensors"

    def __init__(self) -> None:
        self._manifest: HomeManifest | None = None
        self._windows: dict[str, float] = {}  # room id -> window factor
        self._motion: dict[str, Load] = {}
        self._presence: dict[str, Load] = {}
        self._present: dict[str, bool] = {}  # room id -> last reported occupancy
        self._motion_on: dict[str, bool] = {}
        self._last_awake: dict[str, int] = {}  # room id -> last minute an awake person was there
        self._retrigger: dict[str, int] = {}  # room id -> minute of the next re-trigger
        self._motion_due: int | None = None  # earliest minute a motion room needs a look
        self._contact: Load | None = None
        self._door: deque[tuple[int, float]] = deque()  # (minute due, contact value), in order
        self._lux: Load | None = None
        self._main_lights: dict[str, tuple[str, ...]] = {}
        self._temps: dict[str, Load] = {}
        self._humidity: dict[str, Load] = {}
        self._acs: dict[str, tuple[Load, ...]] = {}
        self._temp_state: dict[str, tuple[float, int]] = {}  # room id -> (last value, its minute)
        self._power: tuple[tuple[str, float, float], ...] = ()  # (load id, on W, standby W)
        self._meter: Load | None = None
        self._heat = 0.0
        self._heat_day = None

    def tick(self, ctx: TickContext, rng: np.random.Generator) -> list[SensorReading]:
        self._bind(ctx)
        out: list[SensorReading] = []
        m = ctx.minute
        if ctx.changed or m == 0:
            self._presence_step(ctx, out)
        self._motion_step(ctx, rng, out)
        self._contact_step(ctx, out)
        if m % READING_EVERY_MIN == 0:
            self._climate_readings(ctx, out)
        if m % ENERGY_EVERY_MIN == 0 and self._meter is not None:
            self._energy_reading(ctx, out)
        return out

    # --- occupancy ----------------------------------------------------------------

    def _presence_step(self, ctx: TickContext, out: list[SensorReading]) -> None:
        for room_id, ld in self._presence.items():
            occupied = ctx.any_in(room_id)
            if occupied != self._present[room_id]:
                self._present[room_id] = occupied
                out.append(SensorReading(ld.load_id, OCCUPANCY, 1.0 if occupied else 0.0))

    def _motion_step(self, ctx: TickContext, rng: np.random.Generator, out: list[SensorReading]) -> None:
        m = ctx.minute
        if not self._motion:
            return
        if not ctx.changed and (self._motion_due is None or m < self._motion_due):
            return  # quiet minute with nothing due: no motion room needs a look
        entered = {room_id for person, room_id in ctx.entered() if not person.asleep}
        due: int | None = None
        for room_id, ld in self._motion.items():
            on = self._motion_on.get(room_id, False)
            if ctx.any_awake_in(room_id):
                self._last_awake[room_id] = m
                if not on or room_id in entered or m >= self._retrigger.get(room_id, m):
                    out.append(SensorReading(ld.load_id, OCCUPANCY, 1.0))
                    self._motion_on[room_id] = True
                    lo, hi = MOTION_RETRIGGER_MIN
                    self._retrigger[room_id] = m + int(rng.integers(lo, hi + 1))
                due = _earlier(due, self._retrigger[room_id])
            elif on:
                hold_end = self._last_awake[room_id] + MOTION_HOLD_MIN
                if m >= hold_end:
                    out.append(SensorReading(ld.load_id, OCCUPANCY, 0.0))
                    self._motion_on[room_id] = False
                else:
                    due = _earlier(due, hold_end)
        self._motion_due = due

    def _contact_step(self, ctx: TickContext, out: list[SensorReading]) -> None:
        if self._contact is None:
            return
        if ctx.changed and not any(value == 1.0 for _, value in self._door) and any(
                (p.room_id is None) != (p.prev_room_id is None) for p in ctx.people):
            self._door.append((ctx.minute, 1.0))
            self._door.append((ctx.minute + CONTACT_HOLD_MIN, 0.0))
        if self._door and self._door[0][0] <= ctx.minute:
            _, value = self._door.popleft()
            out.append(SensorReading(self._contact.load_id, "contact", value))

    # --- climate and energy -------------------------------------------------------

    def _climate_readings(self, ctx: TickContext, out: list[SensorReading]) -> None:
        heat = self._heat_now(ctx)
        if self._lux is not None:
            room_id = self._lux.room_id
            lit = sum(1 for load_id in self._main_lights.get(room_id, ()) if ctx.devices.is_on(load_id))
            lux = self._windows[room_id] * ctx.outdoor_lux + LIGHT_LUX_PER_MAIN * lit
            out.append(SensorReading(self._lux.load_id, "illuminance", lux))
        for room_id, ld in self._temps.items():
            out.append(SensorReading(ld.load_id, "temperature", self._temperature(ctx, room_id, heat)))
        humidity = min(1.0, max(0.0, HUMIDITY_BASE + HUMIDITY_SPAN * heat))
        for ld in self._humidity.values():
            out.append(SensorReading(ld.load_id, "humidity", humidity))

    def _temperature(self, ctx: TickContext, room_id: str, heat: float) -> float:
        base = TEMP_BASE_C + TEMP_SPAN_C * heat
        m = ctx.minute
        prev, prev_m = self._temp_state.get(room_id, (base, m))
        running = [ac for ac in self._acs.get(room_id, ()) if ctx.devices.is_on(ac.load_id)]
        target = float(ctx.devices.get(running[0].load_id, "setpoint")) if running else base
        step = TEMP_STEP_C * (m - prev_m)
        value = prev + max(-step, min(step, target - prev))
        self._temp_state[room_id] = (value, m)
        return round(value, 1)

    def _energy_reading(self, ctx: TickContext, out: list[SensorReading]) -> None:
        assert self._meter is not None
        total = 0.0
        for load_id, on_w, standby_w in self._power:
            total += on_w if ctx.devices.is_on(load_id) else standby_w
        out.append(SensorReading(self._meter.load_id, "active_power", total))

    def _heat_now(self, ctx: TickContext) -> float:
        day = ctx.local_date
        if day != self._heat_day:
            self._heat_day = day
            self._heat = heat_index(ctx.manifest.city, day)
        return self._heat

    # --- setup ----------------------------------------------------------------------

    def _bind(self, ctx: TickContext) -> None:
        manifest = ctx.manifest
        if manifest is self._manifest:
            return
        self._manifest = manifest
        self._windows = {room.room_id: room.window_factor for room in manifest.rooms}
        self._motion, self._presence, self._temps, self._humidity = {}, {}, {}, {}
        self._contact = self._lux = self._meter = None
        mains: dict[str, list[str]] = {}
        acs: dict[str, list[Load]] = {}
        power: list[tuple[str, float, float]] = []
        for ld in manifest.loads:
            cls = ld.load_class
            if cls == "sensor.motion":
                self._motion.setdefault(ld.room_id, ld)
            elif cls == "sensor.presence":
                self._presence.setdefault(ld.room_id, ld)
            elif cls == "sensor.contact" and self._contact is None:
                self._contact = ld
            elif cls == "sensor.lux" and self._lux is None:
                self._lux = ld
            elif cls == "sensor.temperature":
                self._temps.setdefault(ld.room_id, ld)
            elif cls == "sensor.humidity":
                self._humidity.setdefault(ld.room_id, ld)
            elif cls == "meter.energy" and self._meter is None:
                self._meter = ld
            if ld.is_main_light:
                mains.setdefault(ld.room_id, []).append(ld.load_id)
            if cls in AC_CLASSES:
                acs.setdefault(ld.room_id, []).append(ld)
            rating = get_load_class(cls).power_w
            on_w, standby_w = rating.get("on", 0.0), rating.get("standby", 0.0)
            if on_w or standby_w:
                power.append((ld.load_id, on_w, standby_w))
        self._main_lights = {room_id: tuple(ids) for room_id, ids in mains.items()}
        self._acs = {room_id: tuple(lds) for room_id, lds in acs.items()}
        self._power = tuple(power)
        self._present = {room_id: False for room_id in self._presence}
        self._motion_on, self._last_awake, self._retrigger = {}, {}, {}
        self._motion_due = None
        self._temp_state = {}
        self._door = deque()


def _earlier(due: int | None, minute: int) -> int:
    return minute if due is None else min(due, minute)
