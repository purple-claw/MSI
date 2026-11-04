"""Utility policy: water heating, the pump, the kitchen chimney and RO purifier, and the washing
machine. Every intent has intent utility.

- Geyser (water.geyser in a bathroom): for each planned bath (the day plan, looked up once per
  run), on at start - lead minutes when the day's heat index is below GEYSER_HEAT or the persona
  has hot_water_always. The lead is drawn per bath in [15, 30], centred on persona
  geyser_lead_min. Off at the bath's end, unless the member forgets (p = forgetfulness): then it
  stays on, and the waste tracker's overrun:60 rule records it. A shared geyser stays on while
  another bath window (lead or bath) of the same geyser is pending.
  Whoever switches the geyser on or off at that minute is the first awake member at home in
  household order (their activity and their own source). With nobody awake at home, the geyser's
  built-in timer does it: no person, source schedule, trigger timer.
- Pump (the first water.pump in the home): a run in 06:00-08:00 every day and, with p 0.4, one in
  18:00-20:00. Each run lasts 20-40 minutes. Actor: the system.
- Chimney (kitchen.chimney, in the room where someone starts cooking): on at speed 2 or 3 when
  the cook starts, off 0-5 minutes after the cook ends unless someone is cooking in the room again.
- RO purifier (water.ro_purifier, in the kitchen): a 10-20 minute run after each cook ends. Actor:
  the system.
- Washing machine: 2-4 runs in each week. The week is planned on Monday 00:00, or on the first
  day of the run for the rest of its week. A run starts 09:00-10:59 on its planned day, by the
  homemaker if at home and awake, else (weekends and festivals only) by the first awake member at
  home. A start with nobody to do it is retried each minute until 13:00 the same day. Runs last
  60-90 minutes.

Each intent checks the device state first: a load already in the wanted state gets nothing, and a
load another policy switched off is switched on only by its own schedule or cook. Scheduled events
wait in one heap of (minute, sequence, kind, payload); the sequence keeps ties in creation order.
"""

import heapq

import numpy as np

from gaia.engine.actuator import Intent
from gaia.engine.context import PersonNow, TickContext
from gaia.home.manifest import HomeManifest, Load
from gaia.household.personas import Persona
from gaia.schema.events import Source
from gaia.world.calendar import DayType
from gaia.world.season import heat_index

MINUTES_PER_DAY = 1440
BATH = "bath"
COOK = "cook"
NONE_ACTIVITY = "none"
UTILITY = "utility"
HYGIENE = "hygiene"  # the geyser serves the bath
GEYSER_CLASS = "water.geyser"
PUMP_CLASS = "water.pump"
RO_CLASS = "water.ro_purifier"
CHIMNEY_CLASS = "kitchen.chimney"
WASHER_CLASS = "laundry.washing_machine"
HOMEMAKER = "homemaker"
GEYSER_HEAT = 0.7  # heat index below this heats the geyser before a bath
GEYSER_LEAD = (15, 30)  # minutes before the bath, inclusive
LEAD_SD = 4.0  # spread of the lead around the persona mean
PUMP_MORNING = (360, 479)  # start minute of day, inclusive
PUMP_EVENING = (1080, 1199)
PUMP_EVENING_P = 0.4
PUMP_RUN = (20, 40)  # minutes, inclusive
WASH_RUNS_WEEK = (2, 4)
WASH_START = (540, 659)  # 09:00-10:59
WASH_DEADLINE = 780  # a missed washer start is retried until 13:00
WASH_RUN = (60, 90)
CHIMNEY_SPEED = (2, 3)
CHIMNEY_OFF_AFTER = (0, 5)
RO_RUN = (10, 20)

_Heap = list[tuple[int, int, str, tuple]]  # (minute, sequence, kind, payload)
_Window = tuple[int, int, bool]  # a geyser window: (on minute, end minute, forgotten)


def _draw(rng: np.random.Generator, lo: int, hi: int) -> int:
    """An integer drawn uniformly from [lo, hi], both inclusive."""
    return int(rng.integers(lo, hi + 1))


class Utility:
    name = "utility"

    def __init__(self) -> None:
        self._manifest: HomeManifest | None = None
        self._city = ""
        self._loads: dict[str, Load] = {}
        self._pump: Load | None = None
        self._washer: Load | None = None
        self._chimney: dict[str, Load] = {}  # room id -> chimney
        self._ro: dict[str, Load] = {}  # room id -> RO purifier
        self._geyser: dict[str, Load] = {}  # room id -> geyser
        self._personas: dict[str, Persona] = {}
        self._prev_act: dict[str, str] = {}  # person id -> activity at the last changed minute
        self._ro_until: dict[str, int] = {}  # RO load id -> minute its current run ends
        self._windows: dict[str, list[_Window]] = {}  # geyser load id -> its bath windows
        self._week_planned = False
        self._heap: _Heap = []
        self._seq = 0
        self._emitted: set[tuple[str, str]] = set()  # (load id, capability) with an intent this tick

    def tick(self, ctx: TickContext, rng: np.random.Generator) -> list[Intent]:
        self._bind(ctx, rng)
        self._emitted = set()
        out: list[Intent] = []
        if ctx.local_minute_of_day == 0:
            self._plan_day(ctx, rng)
            if self._washer is not None and (not self._week_planned or ctx.local_date.weekday() == 0):
                self._plan_week(ctx, rng)
                self._week_planned = True
        self._fire(ctx, rng, out)
        for person, prev in self._transitions(ctx):
            if person.activity == COOK:
                self._cook_start(ctx, person, rng, out)
            elif prev == COOK:
                self._cook_end(ctx, person, rng, out)
        return out

    # --- planning -------------------------------------------------------------

    def _plan_day(self, ctx: TickContext, rng: np.random.Generator) -> None:
        if self._pump is None:
            return
        self._pump_run(ctx.minute + _draw(rng, *PUMP_MORNING), rng)
        if rng.random() < PUMP_EVENING_P:
            self._pump_run(ctx.minute + _draw(rng, *PUMP_EVENING), rng)

    def _pump_run(self, start: int, rng: np.random.Generator) -> None:
        assert self._pump is not None
        lid = self._pump.load_id
        self._push(start, "power", (lid, True, None, NONE_ACTIVITY, UTILITY, "timer"))
        self._push(start + _draw(rng, *PUMP_RUN), "power", (lid, False, None, NONE_ACTIVITY, UTILITY, "timer"))

    def _plan_week(self, ctx: TickContext, rng: np.random.Generator) -> None:
        """Pick the washing days of the rest of this week and each run's start and length."""
        assert self._washer is not None
        today = ctx.local_date.weekday()
        days = list(range(today, 7))
        runs = min(_draw(rng, *WASH_RUNS_WEEK), len(days))
        for weekday in sorted(int(d) for d in rng.choice(days, size=runs, replace=False)):
            day_start = ctx.minute + (weekday - today) * MINUTES_PER_DAY
            start = day_start + _draw(rng, *WASH_START)
            self._push(start, "wash_start",
                       (self._washer.load_id, _draw(rng, *WASH_RUN), day_start + WASH_DEADLINE))

    def _schedule_baths(self, ctx: TickContext, rng: np.random.Generator) -> None:
        """The geyser for each planned bath: on ahead of it, off after it unless forgotten."""
        for span in ctx.activity_spans(BATH):
            geyser = self._geyser.get(span.room_id) if span.room_id is not None else None
            if geyser is None:
                continue
            persona = self._personas[span.person_id]
            heat = heat_index(self._city, ctx.local_date_at(span.start))
            if not (persona.hot_water_always or heat < GEYSER_HEAT):
                continue
            lead = int(round(np.clip(rng.normal(persona.geyser_lead_min, LEAD_SD), *GEYSER_LEAD)))
            forgets = bool(rng.random() < persona.forgetfulness)
            windows = self._windows.setdefault(geyser.load_id, [])
            on_min = max(0, span.start - lead)
            windows.append((on_min, span.end, forgets))
            self._push(on_min, "geyser_on", (geyser.load_id, len(windows) - 1))
            if not forgets:
                self._push(span.end, "geyser_off", (geyser.load_id, len(windows) - 1))

    # --- events ---------------------------------------------------------------

    def _fire(self, ctx: TickContext, rng: np.random.Generator, out: list[Intent]) -> None:
        while self._heap and self._heap[0][0] <= ctx.minute:
            _, _, kind, payload = heapq.heappop(self._heap)
            if kind == "power":
                lid, on, person_id, activity, intent, trigger = payload
                self._power(ctx, self._loads[lid], on, person_id, activity, intent, trigger, out)
            elif kind == "geyser_on":
                self._geyser_switch(ctx, payload[0], True, out)
            elif kind == "geyser_off":
                lid, idx = payload
                on_min, end, _ = self._windows[lid][idx]
                # another bath window of this geyser still pending (lead or bath): it stays on
                if not any(i != idx and start <= ctx.minute < stop
                           for i, (start, stop, _) in enumerate(self._windows[lid])):
                    self._geyser_switch(ctx, lid, False, out)
            elif kind == "chimney_off":
                lid, room_id = payload
                if not self._cooking(ctx, room_id):
                    self._power(ctx, self._loads[lid], False, None, NONE_ACTIVITY, UTILITY, "timer", out)
            elif kind == "ro_off":
                lid, until = payload
                if self._ro_until.get(lid) == until:  # a later run keeps it on
                    del self._ro_until[lid]
                    self._power(ctx, self._loads[lid], False, None, NONE_ACTIVITY, UTILITY, "timer", out)
            elif kind == "wash_start":
                self._wash_start(ctx, payload, out)

    def _wash_start(self, ctx: TickContext, payload: tuple, out: list[Intent]) -> None:
        lid, run, deadline = payload
        if ctx.devices.is_on(lid):
            return
        starter = self._washer_starter(ctx)
        if starter is None:
            if ctx.minute < deadline:  # nobody to start it: try again next minute, until the deadline
                self._push(ctx.minute + 1, "wash_start", payload)
            return
        self._power(ctx, self._loads[lid], True, starter.person_id, starter.activity, UTILITY, "schedule", out)
        self._push(ctx.minute + run, "power", (lid, False, None, NONE_ACTIVITY, UTILITY, "timer"))

    def _geyser_switch(self, ctx: TickContext, lid: str, on: bool, out: list[Intent]) -> None:
        """Switch the geyser by its awake member at home (the first in household order), or by its
        own timer when nobody is awake at home."""
        load = self._loads[lid]
        actor = self._awake_actor(ctx)
        if actor is None:
            self._power(ctx, load, on, None, NONE_ACTIVITY, HYGIENE, "timer", out, source=Source.schedule)
        else:
            self._power(ctx, load, on, actor.person_id, actor.activity, HYGIENE, "schedule", out)

    def _awake_actor(self, ctx: TickContext) -> PersonNow | None:
        for person in ctx.people:
            if person.room_id is not None and not person.asleep:
                return person
        return None

    def _washer_starter(self, ctx: TickContext) -> PersonNow | None:
        awake = [p for p in ctx.people if p.room_id is not None and not p.asleep]
        for person in awake:
            if person.persona.role == HOMEMAKER:
                return person
        if ctx.day_type != DayType.workday and awake:
            return awake[0]
        return None

    # --- cooking ----------------------------------------------------------------

    def _transitions(self, ctx: TickContext) -> list[tuple[PersonNow, str]]:
        """(person, previous activity) for each person whose activity changed this minute."""
        if not ctx.changed:
            return []
        moves = []
        for person in ctx.people:
            prev = self._prev_act.get(person.person_id, person.activity)
            self._prev_act[person.person_id] = person.activity
            if prev != person.activity:
                moves.append((person, prev))
        return moves

    def _cook_start(self, ctx: TickContext, person: PersonNow, rng: np.random.Generator,
                    out: list[Intent]) -> None:
        if person.room_id is None or person.room_id not in self._chimney:
            return
        chimney = self._chimney[person.room_id]
        speed = float(_draw(rng, *CHIMNEY_SPEED))
        self._power(ctx, chimney, True, person.person_id, COOK, UTILITY, "activity_start", out)
        if ctx.devices.get(chimney.load_id, "fan_speed") != speed and self._claim(chimney.load_id, "fan_speed"):
            out.append(Intent(
                load_id=chimney.load_id, capability="fan_speed", value=speed, person_id=person.person_id,
                activity=COOK, intent=UTILITY, trigger="activity_start", device_action="set_fan_speed",
                policy=self.name, source=None,
            ))

    def _cook_end(self, ctx: TickContext, person: PersonNow, rng: np.random.Generator,
                  out: list[Intent]) -> None:
        room_id = person.prev_room_id
        if room_id is None:
            return
        chimney = self._chimney.get(room_id)
        if chimney is not None:
            self._push(ctx.minute + _draw(rng, *CHIMNEY_OFF_AFTER), "chimney_off", (chimney.load_id, room_id))
        ro = self._ro.get(room_id)
        if ro is not None:
            until = ctx.minute + _draw(rng, *RO_RUN)
            self._ro_until[ro.load_id] = until
            self._power(ctx, ro, True, None, NONE_ACTIVITY, UTILITY, "activity_end", out)
            self._push(until, "ro_off", (ro.load_id, until))

    def _cooking(self, ctx: TickContext, room_id: str) -> bool:
        return any(p.room_id == room_id and p.activity == COOK and not p.asleep for p in ctx.people)

    # --- intents ----------------------------------------------------------------

    def _power(self, ctx: TickContext, load: Load, on: bool, person_id: str | None, activity: str,
               intent: str, trigger: str, out: list[Intent], source: Source | None = None) -> None:
        if ctx.devices.is_on(load.load_id) == on or not self._claim(load.load_id, "power"):
            return
        out.append(Intent(
            load_id=load.load_id, capability="power", value=1.0 if on else 0.0, person_id=person_id,
            activity=activity, intent=intent, trigger=trigger,
            device_action="turn_on" if on else "turn_off", policy=self.name, source=source,
        ))

    def _claim(self, load_id: str, capability: str) -> bool:
        key = (load_id, capability)
        if key in self._emitted:
            return False  # the actuator keeps the first intent per (load, capability)
        self._emitted.add(key)
        return True

    # --- schedule -----------------------------------------------------------------

    def _push(self, minute: int, kind: str, payload: tuple) -> None:
        self._seq += 1
        heapq.heappush(self._heap, (minute, self._seq, kind, payload))

    # --- setup --------------------------------------------------------------------

    def _bind(self, ctx: TickContext, rng: np.random.Generator) -> None:
        manifest = ctx.manifest
        if manifest is self._manifest:
            return
        self._manifest = manifest
        self._city = manifest.city
        self._loads = {ld.load_id: ld for ld in manifest.loads}
        self._pump = self._washer = None
        self._chimney, self._ro, self._geyser = {}, {}, {}
        for ld in manifest.loads:
            if ld.load_class == PUMP_CLASS and self._pump is None:
                self._pump = ld
            elif ld.load_class == WASHER_CLASS and self._washer is None:
                self._washer = ld
            elif ld.load_class == CHIMNEY_CLASS:
                self._chimney.setdefault(ld.room_id, ld)
            elif ld.load_class == RO_CLASS:
                self._ro.setdefault(ld.room_id, ld)
            elif ld.load_class == GEYSER_CLASS:
                self._geyser.setdefault(ld.room_id, ld)
        self._personas = {p.person_id: p.persona for p in ctx.people}
        self._prev_act = {p.person_id: p.activity for p in ctx.people}
        self._schedule_baths(ctx, rng)
