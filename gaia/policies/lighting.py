"""Lighting v2: room scenes per activity, circadian tunable white, RGB cove, daylight switch-off
and night lights. Replaces the G1 ``lighting_basic`` policy, whose rules it keeps.

Changed minutes (someone's room, sleep or activity moved), per person in household order:
- Leaving a room with lights on: if nobody awake remains and no later member also leaves it
  awake this minute, the last awake leaver decides once. A forgetful person leaves every lit
  circuit on (the waste tracker records it); otherwise all of them go off.
- Falling asleep: the bedroom's lit circuits go off if nobody else is awake in it. If nobody
  else is awake in the home, each lit room with nobody in it goes off with p=0.5 (cleanup).
- Activity change while awake: the room's scene applies if its main light is on or the persona
  needs light for the new activity. Layers the scene does not want go off (activity_end).
- A person moving between two rooms between sunset and sunrise lights the night circuits of
  the first passage or staircase room with them (a stand-in for passing through it).

Every minute:
- An awake person in a dark room whose main light is off gets the room scene (darkness or
  room_entry).
- A lit room whose main light has seen indoor lux above 3x its awake occupant's threshold for
  15 minutes may lose that main light, with the occupant's daylight_off_p. One draw per bright
  spell.
- Night circuits go off 2 minutes after the last passage.
- Every 5 minutes, tunable lights move to the circadian CCT, but only by more than 300 K, and at
  most one cct command per 10 minutes unless the light was power-cycled in between.

Light state is the policy's own mirror of the intents it has emitted this tick, so two decisions
in one minute see each other. Each minute the mirror is first reconciled with the device, so a
light another policy switched off (a notice) is seen as off. A load gets at most one intent per
capability per tick, since the actuator keeps only the first. Iteration uses dicts (insertion
order), never sets.
"""

import numpy as np

from gaia.behaviour.needs import indoor_lux, light_needed
from gaia.engine.actuator import Intent
from gaia.engine.context import PersonNow, TickContext
from gaia.home.manifest import HomeManifest, Load, Room
from gaia.interaction.choose import forgets
from gaia.registry.registry import get_load_class
from gaia.schema.events import Source

CLEANUP_P = 0.5
DAYLIGHT_FACTOR = 3.0  # bright when indoor lux exceeds this multiple of the light threshold
DAYLIGHT_MINUTES = 15
NIGHT_HOLD_MIN = 2
CCT_CHECK_EVERY = 5
CCT_STEP_K = 300.0
CCT_GUARD_MIN = 10
CCT_LO, CCT_HI = 2700.0, 6500.0
DAY_K, EVENING_K, WIND_DOWN_K = 5000.0, 3000.0, 2700.0
WARM_HUE, WARM_SAT, WARM_LEVEL = 0.08, 0.6, 0.3
TV_MAIN_DIMMED, TV_MAIN, TV_ACCENT = 0.3, 0.6, 0.5
WIND_DOWN_MAIN, WIND_DOWN_COVE = 0.4, 0.4
RELAX_MAIN, RELAX_ACCENT = 0.7, 0.5
WAKE_MAIN = 0.5
EPS = 1e-9

_NONE = "none"
_NIGHT = "night"
_DECOR = "decor"
_MEALS = frozenset({"breakfast", "lunch", "dinner"})
_MEAL_ROOMS = frozenset({"dining", "kitchen"})
_AMBIENCE = frozenset({"tv", "dinner", "relax", "wind_down"})
_NIGHT_ROOM_TYPES = frozenset({"passage", "staircase"})


def circadian_cct(minute_of_day: int) -> float:
    """CCT before the persona offset: 5000 K from 07:00 to 17:00, a linear ramp to 3000 K at
    19:00, then 3000 K (also overnight)."""
    if 7 * 60 <= minute_of_day < 17 * 60:
        return DAY_K
    if 17 * 60 <= minute_of_day < 19 * 60:
        return DAY_K + (EVENING_K - DAY_K) * (minute_of_day - 17 * 60) / 120.0
    return EVENING_K


def _ambience_or_illumination(activity: str) -> str:
    return "ambience" if activity in _AMBIENCE else "illumination"


class Lighting:
    name = "lighting"

    def __init__(self) -> None:
        self._manifest: HomeManifest | None = None
        self._rooms: dict[str, Room] = {}
        self._loads: dict[str, Load] = {}
        self._room_loads: dict[str, tuple[Load, ...]] = {}  # rooms with lighting, manifest order
        self._circuits: dict[str, frozenset[str]] = {}
        self._main: dict[str, Load] = {}
        self._light_rooms: tuple[str, ...] = ()
        self._caps: dict[str, frozenset[str]] = {}
        self._cct_loads: tuple[str, ...] = ()
        self._night_room: str | None = None
        self._night_loads: tuple[Load, ...] = ()
        self._on: dict[str, None] = {}  # lights on after this tick's intents
        self._lit_rooms: dict[str, None] = {}  # rooms with any light on
        self._activity: dict[str, str] = {}  # person id -> activity at the last changed minute
        self._cct_last: dict[str, int] = {}  # load id -> minute of its last cct command
        self._bright: dict[str, list] = {}  # room id -> [bright minutes with an occupant, drawn]
        self._last_pass: int | None = None
        self._open_last_minute = False  # some waste interval was open at the previous tick
        self._emitted: set[tuple[str, str]] = set()

    def tick(self, ctx: TickContext, rng: np.random.Generator) -> list[Intent]:
        self._bind(ctx)
        pairs = ctx.open_waste()
        # A notice switch-off happens while its load's waste is open, so the mirror needs a look on
        # the minute the waste was open, the next minute, and any minute someone or something changed.
        if ctx.changed or pairs or self._open_last_minute:
            self._reconcile(ctx)
        self._open_last_minute = bool(pairs)
        self._emitted = set()
        out: list[Intent] = []
        people = ctx.people
        if ctx.changed:
            for i, person in enumerate(people):
                if person.prev_room_id is not None and person.prev_room_id != person.room_id:
                    self._leave(ctx, people, i, rng, out)
                    self._pass(ctx, person, out)
                if person.asleep and not person.prev_asleep:
                    self._fall_asleep(ctx, people, i, rng, out)
                self._activity_change(ctx, person, out)
        for person in people:
            if not person.asleep and person.room_id is not None:
                self._presence(ctx, person, out)
        self._daylight(ctx, rng, out)
        self._night_hold(ctx, out)
        if ctx.minute % CCT_CHECK_EVERY == 0:
            self._cct_sweep(ctx, out)
        return out

    def _reconcile(self, ctx) -> None:
        """Forget lights that another policy switched off (a notice): the mirror follows the device."""
        if not self._on:
            return
        for lid in tuple(self._on):
            if ctx.devices.is_on(lid):
                continue
            del self._on[lid]
            self._cct_last.pop(lid, None)  # a power cycle lifts the cct guard, as in _power
            room_id = self._loads[lid].room_id
            if not any(ld.load_id in self._on for ld in self._room_loads[room_id]):
                self._lit_rooms.pop(room_id, None)

    # --- rules ----------------------------------------------------------------

    def _leave(self, ctx, people, i, rng, out: list[Intent]) -> None:
        person = people[i]
        room = person.prev_room_id
        if room not in self._lit_rooms:
            return
        if any(q.room_id == room and not q.asleep for j, q in enumerate(people) if j != i):
            return
        if any(q.prev_room_id == room and q.room_id != room and not q.asleep for q in people[i + 1:]):
            return  # a later awake leaver in the same minute decides the exit
        if forgets(person.persona, rng):
            return  # every lit circuit stays on; waste tracking notices
        pid, activity = person.acting() or (None, _NONE)
        for load in self._room_loads[room]:
            self._power(ctx, load, False, pid, activity, "energy_saving", "room_exit", "turn_off", None, out)

    def _fall_asleep(self, ctx, people, i, rng, out: list[Intent]) -> None:
        person = people[i]
        bed = person.room_id
        others_awake = [q for j, q in enumerate(people) if j != i and not q.asleep]
        if bed is not None and bed in self._lit_rooms and not any(q.room_id == bed for q in others_awake):
            pid, activity = person.acting() or (None, _NONE)
            for load in self._room_loads[bed]:
                self._power(ctx, load, False, pid, activity, "illumination", "sleep", "turn_off", None, out)
        if others_awake:
            return
        for room_id in self._light_rooms:
            if room_id not in self._lit_rooms or any(q.room_id == room_id for q in people):
                continue
            if rng.random() < CLEANUP_P:
                pid, activity = person.acting() or (None, _NONE)
                for load in self._room_loads[room_id]:
                    self._power(ctx, load, False, pid, activity, "cleanup",
                                "last_to_sleep", "turn_off", Source.app, out)

    def _activity_change(self, ctx, person: PersonNow, out: list[Intent]) -> None:
        prev = self._activity.get(person.person_id, person.activity)
        self._activity[person.person_id] = person.activity
        room = person.room_id
        if person.asleep or room is None or prev == person.activity or room not in self._main:
            return
        if self._main[room].load_id not in self._on and not light_needed(
                indoor_lux(ctx.outdoor_lux, self._rooms[room]), person.persona, person.activity):
            return
        self._scene(ctx, room, person, "activity_start", _ambience_or_illumination(person.activity), out)

    def _presence(self, ctx, person: PersonNow, out: list[Intent]) -> None:
        room = person.room_id
        main = self._main.get(room)
        if main is None or main.load_id in self._on:
            return
        indoor = indoor_lux(ctx.outdoor_lux, self._rooms[room])
        if not light_needed(indoor, person.persona, person.activity):
            return
        trigger = "room_entry" if person.prev_room_id != room else "darkness"
        self._scene(ctx, room, person, trigger, "illumination", out)

    def _pass(self, ctx, person: PersonNow, out: list[Intent]) -> None:
        """A move between two rooms at night is a passage through the night room."""
        if (self._night_room is None or person.asleep or person.prev_room_id is None
                or person.room_id is None or person.prev_room_id == person.room_id):
            return
        if ctx.sun_altitude_deg >= 0:
            return
        self._last_pass = ctx.minute
        for load in self._night_loads:
            self._power(ctx, load, True, person.person_id, person.activity, "illumination",
                        "room_entry", "turn_on", None, out)

    def _daylight(self, ctx, rng, out: list[Intent]) -> None:
        for room_id in tuple(self._bright):  # a spell ends when its room is no longer lit
            if room_id not in self._lit_rooms:
                del self._bright[room_id]
        if not self._lit_rooms:
            return
        outdoor = ctx.outdoor_lux
        for room_id in tuple(self._lit_rooms):
            main = self._main.get(room_id)
            occupants = ctx.awake_in(room_id)
            if main is None or main.load_id not in self._on or not occupants:
                self._bright.pop(room_id, None)
                continue
            head = occupants[0]
            indoor = indoor_lux(outdoor, self._rooms[room_id])
            if indoor <= DAYLIGHT_FACTOR * head.persona.light_threshold_lux:
                self._bright.pop(room_id, None)
                continue
            spell = self._bright.setdefault(room_id, [0, False])  # [bright minutes, drawn]
            spell[0] += 1
            if spell[0] >= DAYLIGHT_MINUTES and not spell[1]:
                spell[1] = True  # one draw per bright spell
                if rng.random() < head.persona.daylight_off_p:
                    self._power(ctx, main, False, head.person_id, head.activity, "energy_saving",
                                "daylight", "turn_off", None, out)

    def _night_hold(self, ctx, out: list[Intent]) -> None:
        if self._last_pass is None or ctx.minute - self._last_pass < NIGHT_HOLD_MIN:
            return
        self._last_pass = None
        for load in self._night_loads:
            self._power(ctx, load, False, None, _NONE, "energy_saving", "timer", "turn_off", None, out)

    def _cct_sweep(self, ctx, out: list[Intent]) -> None:
        for lid in self._cct_loads:
            if lid not in self._on:
                continue
            load = self._loads[lid]
            occupants = ctx.awake_in(load.room_id)
            if occupants:
                head = occupants[0]
                self._cct(ctx, load, head.person_id, head.activity, head.persona.cct_pref_offset_k, out)
            else:
                self._cct(ctx, load, None, _NONE, 0.0, out)

    # --- scenes -----------------------------------------------------------------

    def _plan(self, room_id: str, activity: str, dim: float) -> dict[str, tuple[float, bool]]:
        """Wanted circuits for an activity: circuit -> (level, warm colour). Main is always wanted."""
        circuits = self._circuits[room_id]
        if activity == "tv":
            plan: dict[str, tuple[float, bool]] = {}
            if "cove" in circuits:
                plan["cove"] = (WARM_LEVEL, True)
            if "accent" in circuits:
                plan["accent"] = (TV_ACCENT, False)
            dimmed = "cove" in plan or "accent" in plan
            plan["main"] = (TV_MAIN_DIMMED if dimmed else TV_MAIN, False)
            return plan
        if activity == "cook":
            plan = {"main": (1.0, False)}
            if "task" in circuits:
                plan["task"] = (1.0, False)
            return plan
        if activity in _MEALS and self._rooms[room_id].room_type in _MEAL_ROOMS:
            plan = {"main": (dim, False)}
            if "accent" in circuits:
                plan["accent"] = (dim, False)
            return plan
        if activity == "wind_down":
            plan = {"main": (WIND_DOWN_MAIN, False)}
            if "cove" in circuits:
                plan["cove"] = (WIND_DOWN_COVE, False)
            return plan
        if activity == "relax":
            plan = {"main": (RELAX_MAIN, False)}
            if "accent" in circuits:
                plan["accent"] = (RELAX_ACCENT, False)
            return plan
        if activity == "wake_up":
            return {"main": (WAKE_MAIN, False)}
        return {"main": (dim, False)}

    def _scene(self, ctx, room_id: str, person: PersonNow, trigger: str, intent: str,
               out: list[Intent]) -> None:
        """Apply the room's scene for the person's activity: wanted layers on and shaped, unwanted
        layers off."""
        plan = self._plan(room_id, person.activity, person.persona.dim_pref)
        for load in self._room_loads[room_id]:
            if load.circuit in (_NIGHT, _DECOR):
                continue
            want = plan.get(load.circuit)
            if want is None:
                self._power(ctx, load, False, person.person_id, person.activity, "energy_saving",
                            "activity_end", "turn_off", None, out)
                continue
            self._power(ctx, load, True, person.person_id, person.activity, intent, trigger,
                        "turn_on", None, out)
            if load.load_id not in self._on:
                continue  # the power intent was dropped as a duplicate this tick
            level, warm = want
            self._shape(ctx, load, level, warm, person, intent, trigger, out)

    def _shape(self, ctx, load: Load, level: float, warm: bool, person: PersonNow, intent: str,
               trigger: str, out: list[Intent]) -> None:
        caps = self._caps[load.load_id]
        if "level" in caps:
            self._set(ctx, load, "level", level, person, intent, trigger, out)
        if warm and "hue" in caps:
            self._set(ctx, load, "hue", WARM_HUE, person, intent, trigger, out)
            self._set(ctx, load, "saturation", WARM_SAT, person, intent, trigger, out)
        if "cct" in caps:
            self._cct(ctx, load, person.person_id, person.activity, person.persona.cct_pref_offset_k, out)

    def _cct(self, ctx, load: Load, person_id: str | None, activity: str, offset: float,
             out: list[Intent]) -> None:
        if load.load_id not in self._on:
            return
        base = WIND_DOWN_K if activity == "wind_down" else circadian_cct(ctx.local_minute_of_day)
        target = float(round(min(max(base + offset, CCT_LO), CCT_HI)))
        current = float(ctx.devices.get(load.load_id, "cct"))
        if abs(target - current) <= CCT_STEP_K:
            return
        last = self._cct_last.get(load.load_id)
        if last is not None and ctx.minute - last < CCT_GUARD_MIN:
            return
        if self._emit(load, "cct", target, person_id, activity, "ambience", "timer", "set_cct", None, out):
            self._cct_last[load.load_id] = ctx.minute

    # --- intents ----------------------------------------------------------------

    def _power(self, ctx, load: Load, on: bool, person_id: str | None, activity: str, intent: str,
               trigger: str, device_action: str, source: Source | None, out: list[Intent]) -> bool:
        lid = load.load_id
        if (lid in self._on) == on:
            return False
        if not self._emit(load, "power", 1.0 if on else 0.0, person_id, activity, intent, trigger,
                          device_action, source, out):
            return False
        if on:
            self._on[lid] = None
        else:
            self._on.pop(lid, None)
        self._cct_last.pop(lid, None)  # a power cycle lifts the cct guard
        if any(ld.load_id in self._on for ld in self._room_loads[load.room_id]):
            self._lit_rooms[load.room_id] = None
        else:
            self._lit_rooms.pop(load.room_id, None)
        return True

    def _set(self, ctx, load: Load, capability: str, value: float, person: PersonNow, intent: str,
             trigger: str, out: list[Intent]) -> None:
        if abs(float(ctx.devices.get(load.load_id, capability)) - value) <= EPS:
            return
        action = "set_level" if capability == "level" else "set_color"
        self._emit(load, capability, value, person.person_id, person.activity, intent, trigger,
                   action, None, out)

    def _emit(self, load: Load, capability: str, value: float, person_id: str | None, activity: str,
              intent: str, trigger: str, device_action: str, source: Source | None,
              out: list[Intent]) -> bool:
        key = (load.load_id, capability)
        if key in self._emitted:
            return False  # the actuator keeps the first intent per (load, capability)
        self._emitted.add(key)
        out.append(Intent(
            load_id=load.load_id, capability=capability, value=value, person_id=person_id,
            activity=activity, intent=intent, trigger=trigger, device_action=device_action,
            policy=self.name, source=source,
        ))
        return True

    # --- setup ------------------------------------------------------------------

    def _bind(self, ctx: TickContext) -> None:
        manifest = ctx.manifest
        if manifest is self._manifest:
            return
        self._manifest = manifest
        self._rooms = {room.room_id: room for room in manifest.rooms}
        room_loads: dict[str, tuple[Load, ...]] = {}
        for room in manifest.rooms:
            lights = tuple(ld for ld in manifest.loads_in(room.room_id) if ld.circuit != "none")
            if lights:
                room_loads[room.room_id] = lights
        self._room_loads = room_loads
        self._light_rooms = tuple(room_loads)
        self._loads = {ld.load_id: ld for lights in room_loads.values() for ld in lights}
        self._circuits = {rid: frozenset(ld.circuit for ld in lights) for rid, lights in room_loads.items()}
        self._main = {}
        for rid in self._light_rooms:
            main = manifest.main_light(rid)
            if main is not None:
                self._main[rid] = main
        self._caps = {
            lid: frozenset(c.name for c in get_load_class(ld.load_class).capabilities)
            for lid, ld in self._loads.items()
        }
        self._cct_loads = tuple(lid for lid in self._loads if "cct" in self._caps[lid])
        self._night_room = next(
            (r.room_id for r in manifest.rooms if r.room_type in _NIGHT_ROOM_TYPES
             and any(ld.circuit == _NIGHT for ld in room_loads.get(r.room_id, ()))),
            None,
        )
        self._night_loads = tuple(ld for ld in room_loads.get(self._night_room or "", ())
                                  if ld.circuit == _NIGHT)
        self._on = {lid: None for lid in self._loads if ctx.devices.is_on(lid)}
        self._lit_rooms = {rid: None for rid in self._light_rooms
                           if any(ld.load_id in self._on for ld in room_loads[rid])}
        self._activity = {p.person_id: p.activity for p in ctx.people}
