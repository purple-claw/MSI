"""Shading policy: curtains, rollers and skylights (position) and venetian blinds (tilt).

Every shade.* load follows these rules:
- Morning opening: when someone's wake_up starts in a room, each shade there opens (position 1.0)
  with p 0.8. One draw per room per wake-up. Labels: illumination, trigger wake.
- Daylight opening: at sunrise (solar altitude crosses 0 upward), each shade in a room with
  someone awake opens with p 0.8, one draw per room. Trigger daylight.
- Evening closing: at sunset (crossing 0 downward) one offset of 0-30 minutes is drawn for the
  day. From that minute, every shade closes (position 0.0) at the first minute someone is home.
  If the home is empty, the close stays pending until then. Labels: privacy, trigger schedule.
- Wind-down: when someone's wind_down starts in a room, each shade there closes with p 0.9.
  Labels: privacy, trigger activity_start.
- Midday tilt: at 12:00 local, when heat_index > 0.6, every venetian blind tilts to 0.5. Labels:
  thermal_comfort, trigger schedule.

A shade already in the wanted state gets no intent. Iteration uses dicts and tuples.
"""

import numpy as np

from gaia.engine.actuator import Intent
from gaia.engine.context import PersonNow, TickContext
from gaia.home.manifest import HomeManifest, Load
from gaia.world.season import heat_index

OPEN = 1.0
CLOSED = 0.0
OPEN_P = 0.8
WIND_DOWN_CLOSE_P = 0.9
SUNSET_WINDOW_MIN = 30
MIDDAY_MINUTE = 12 * 60
TILT_HEAT = 0.6
TILT = 0.5
_NONE = "none"


class Shading:
    name = "shading"

    def __init__(self) -> None:
        self._manifest: HomeManifest | None = None
        self._shades: dict[str, tuple[Load, ...]] = {}  # room id -> shade loads
        self._venetians: tuple[Load, ...] = ()
        self._activity: dict[str, str] = {}  # person id -> activity at the last changed minute
        self._prev_sun: float | None = None
        self._close_at: int | None = None  # minute of today's evening close
        self._close_pending = False  # the evening close is due and waits for someone to come home
        self._day = None
        self._heat = 0.0
        self._emitted: set[tuple[str, str]] = set()

    def tick(self, ctx: TickContext, rng: np.random.Generator) -> list[Intent]:
        self._bind(ctx)
        self._emitted = set()
        out: list[Intent] = []
        sun = ctx.sun_altitude_deg
        prev_sun, self._prev_sun = self._prev_sun, sun
        if ctx.changed:
            for person in ctx.people:
                prev = self._activity.get(person.person_id)
                self._activity[person.person_id] = person.activity
                if prev is None or prev == person.activity or person.asleep or person.room_id is None:
                    continue
                if person.activity == "wake_up":
                    self._draw_room(ctx, rng, person.room_id, person, OPEN, OPEN_P, "illumination",
                                    "wake", out)
                elif person.activity == "wind_down":
                    self._draw_room(ctx, rng, person.room_id, person, CLOSED, WIND_DOWN_CLOSE_P,
                                    "privacy", "activity_start", out)
        if prev_sun is not None:
            if prev_sun < 0 <= sun:
                self._sunrise(ctx, rng, out)
            elif prev_sun >= 0 > sun:
                self._close_at = ctx.minute + int(rng.integers(0, SUNSET_WINDOW_MIN + 1))
        if self._close_at is not None and ctx.minute >= self._close_at:
            self._close_at = None
            self._close_pending = True
        if self._close_pending and any(p.room_id is not None for p in ctx.people):
            self._close_pending = False
            self._close_all(ctx, out)
        if ctx.local_minute_of_day == MIDDAY_MINUTE and self._heat_now(ctx) > TILT_HEAT:
            for ld in self._venetians:
                if ctx.devices.get(ld.load_id, "tilt") != TILT:
                    self._emit(ld, "tilt", TILT, None, _NONE, "thermal_comfort", "schedule", "set_tilt", out)
        return out

    # --- rules --------------------------------------------------------------------

    def _sunrise(self, ctx, rng, out: list[Intent]) -> None:
        awake: dict[str, PersonNow] = {}
        for person in ctx.people:
            if person.room_id is not None and not person.asleep:
                awake.setdefault(person.room_id, person)
        for room_id, person in awake.items():
            self._draw_room(ctx, rng, room_id, person, OPEN, OPEN_P, "illumination", "daylight", out)

    def _draw_room(self, ctx, rng, room_id: str, person: PersonNow, position: float, p: float,
                   intent: str, trigger: str, out: list[Intent]) -> None:
        """One draw for the room; if it wins, every shade there moves to ``position``."""
        pending = [ld for ld in self._shades.get(room_id, ()) if ctx.devices.get(ld.load_id, "position") != position]
        if not pending:
            return
        if rng.random() < p:
            for ld in pending:
                self._emit(ld, "position", position, person.person_id, person.activity, intent, trigger,
                           _action(position), out)

    def _close_all(self, ctx, out: list[Intent]) -> None:
        for shades in self._shades.values():
            for ld in shades:
                if ctx.devices.get(ld.load_id, "position") != CLOSED:
                    self._emit(ld, "position", CLOSED, None, _NONE, "privacy", "schedule", "close", out)

    # --- intents, state and setup ----------------------------------------------------

    def _emit(self, ld: Load, capability: str, value: float, person_id: str | None, activity: str,
              intent: str, trigger: str, device_action: str, out: list[Intent]) -> None:
        key = (ld.load_id, capability)
        if key in self._emitted:
            return  # the actuator keeps the first intent per (load, capability)
        self._emitted.add(key)
        out.append(Intent(
            load_id=ld.load_id, capability=capability, value=value, person_id=person_id,
            activity=activity, intent=intent, trigger=trigger, device_action=device_action,
            policy=self.name, source=None,
        ))

    def _heat_now(self, ctx: TickContext) -> float:
        day = ctx.local_date
        if day != self._day:
            self._day = day
            self._heat = heat_index(ctx.manifest.city, day)
        return self._heat

    def _bind(self, ctx: TickContext) -> None:
        manifest = ctx.manifest
        if manifest is self._manifest:
            return
        self._manifest = manifest
        shades: dict[str, list[Load]] = {}
        for ld in manifest.loads:
            if ld.load_class.startswith("shade."):
                shades.setdefault(ld.room_id, []).append(ld)
        self._shades = {rid: tuple(lds) for rid, lds in shades.items()}
        self._venetians = tuple(ld for lds in self._shades.values() for ld in lds
                                if ld.load_class == "shade.venetian")
        self._activity = {p.person_id: p.activity for p in ctx.people}


def _action(position: float) -> str:
    return "open" if position == OPEN else "close"
