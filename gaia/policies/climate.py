"""Climate policy: air conditioners, room (ceiling) fans and the kitchen and bathroom exhaust.

Heat comes from gaia.world.season: heat_index, 0-1, per city and day.

A stay is one person in one room, from arrival or sleep start until they leave or their sleep
state changes. A stay decides the AC and the fan once each, from the persona's preferences:
- Sleep stay: the AC in the room goes on at sleep start with p = ac_affinity * heat_index.
- Awake stay: the same p, drawn at the first minute the stay has lasted 30 minutes with
  heat_index > 0.5.
- An AC switched on that way also switches each ceiling fan in the room on at speed 2, p 0.5.
- Fan: a person awake in a room, or asleep in a bedroom, with heat_index > 0.3 gets the room's
  ceiling fans on with p = fan_affinity, at speed 1 + round(4 * heat_index).
- The AC is set to cool at the persona's setpoint, fan speed 3.

Changed minutes (someone's room or sleep state moved), per person in household order:
- Leaving a room with an AC or ceiling fan on: if nobody else is in the room (asleep or awake)
  and no later member also leaves it this minute, the leaver decides once. A forgetful leaver
  leaves everything on (the waste tracker records it); otherwise all of them go off.
  Sleepers keep the room's AC running.
- Waking up: the AC in the room goes off with p 0.8, unless someone else is in the room.

Exhaust: a fan.exhaust goes on while an occupant does the activity mapped to its room type in
EXHAUST_ACTIVITY_ROOM (cooking in the kitchen; bathing and toilet visits in a bathroom), and goes
off 5-10 minutes after the activity ends. Each load draws its own
timer from the policy rng when the activity ends. At the timer, a forgetful occupant (p =
forgetfulness * EXHAUST_FORGET_FACTOR) leaves it on; then it stays on until someone next does the
activity. That forgotten exhaust is the waste the tracker records (unattended:15).

Sleep-start AC needs heat_index >= SLEEP_AC_MIN_HEAT; below it no AC is drawn for sleep.

One intent per (load, capability) per tick, the first one wins. Iteration uses dicts and tuples.
"""

import numpy as np

from gaia.engine.actuator import Intent
from gaia.engine.context import PersonNow, TickContext
from gaia.home.manifest import HomeManifest, Load, Room
from gaia.household.personas import Persona
from gaia.interaction.choose import forgets
from gaia.world.season import heat_index

AC_PREFIX = "climate."
CEILING_FANS = frozenset({"fan.bldc_ceiling", "fan.ceiling"})
EXHAUST = "fan.exhaust"
# Activity -> room type where the exhaust runs.
EXHAUST_ACTIVITY_ROOM: dict[str, str] = {"cook": "kitchen", "bath": "bathroom", "toilet": "bathroom"}
AC_MODE = "cool"
AC_FAN_SPEED = 3.0
AC_COUPLED_FAN_SPEED = 2.0
AC_AWAKE_MINUTES = 30
AC_AWAKE_HEAT = 0.5
AC_WAKE_OFF_P = 0.8
AC_FAN_P = 0.5
FAN_HEAT = 0.3
EXHAUST_OFF_MIN = 5
EXHAUST_OFF_MAX = 10
EXHAUST_FORGET_FACTOR = 0.5  # at the off timer, a forgetful occupant leaves the exhaust on with p = forgetfulness * this
SLEEP_AC_MIN_HEAT = 0.3  # no AC at sleep start when the heat index is below this
EPS = 1e-9
NEVER = 1 << 62


_ACTIONS = {"mode": "set_mode", "setpoint": "set_setpoint", "fan_speed": "set_fan_speed"}


def fan_speed(heat: float) -> float:
    """Presence fan speed: 1 + round(4 * heat), within the 1-5 range of the ceiling fan classes."""
    return float(1 + round(4 * heat))


def _who(person: PersonNow | None) -> tuple[str | None, str]:
    """The actor of a command: the person (PersonNow.acting), else the system."""
    acting = None if person is None else person.acting()
    return (None, "none") if acting is None else acting


def _differs(current, value) -> bool:
    if isinstance(value, str):
        return current != value
    return abs(float(current) - float(value)) > EPS


class _Stay:
    __slots__ = ("room_id", "asleep", "start", "ac_decided", "fan_decided")

    def __init__(self, room_id: str, asleep: bool, start: int) -> None:
        self.room_id = room_id
        self.asleep = asleep
        self.start = start  # minute the stay began
        self.ac_decided = False
        self.fan_decided = False


class Climate:
    name = "climate"

    def __init__(self) -> None:
        self._manifest: HomeManifest | None = None
        self._rooms: dict[str, Room] = {}
        self._ac: dict[str, tuple[Load, ...]] = {}  # room id -> AC loads
        self._fans: dict[str, tuple[Load, ...]] = {}  # room id -> ceiling fans
        self._exhaust: tuple[Load, ...] = ()
        self._climate_rooms: dict[str, None] = {}  # rooms with an AC or a ceiling fan
        self._stays: dict[str, _Stay] = {}  # person id -> current stay
        self._next_check = 0  # first minute that needs a full pass without a change
        self._exhaust_off_at: dict[str, tuple[int, float]] = {}  # exhaust load -> (off minute, forget p)
        self._exhaust_forgotten: dict[str, None] = {}  # exhaust loads left on by a forgetful occupant
        self._exhaust_persona: dict[str, Persona] = {}  # room id -> persona of its last exhaust user
        self._day = None
        self._heat = 0.0
        self._emitted: set[tuple[str, str]] = set()

    def tick(self, ctx: TickContext, rng: np.random.Generator) -> list[Intent]:
        # A quiet minute needs nothing: no change, no AC check due, no exhaust timer, and not midnight
        # (heat index changes at local midnight only). The first minute is never quiet, so it binds.
        if (ctx.minute < self._next_check and not self._exhaust_off_at and not ctx.changed
                and ctx.local_minute_of_day):
            return []
        self._bind(ctx)
        self._emitted = set()
        out: list[Intent] = []
        heat = self._heat_now(ctx)
        people = ctx.people
        if ctx.changed:
            for i, person in enumerate(people):
                if person.prev_room_id is not None and person.prev_room_id != person.room_id:
                    self._leave(ctx, people, i, rng, out)
                if person.prev_asleep and not person.asleep and person.room_id is not None:
                    self._wake(ctx, people, i, rng, out)
        for person in people:
            if person.room_id in self._climate_rooms:
                self._stay_minute(ctx, person, heat, rng, out)
            else:
                self._stays.pop(person.person_id, None)
        self._next_check = self._next_ac_check(ctx.minute, heat)
        # Exhaust decisions change only when someone's activity or room changes, or when a timer is due.
        if self._exhaust and (ctx.changed or self._exhaust_off_at):
            self._exhaust_minute(ctx, rng, out)
        return out

    def _next_ac_check(self, minute: int, heat: float) -> int:
        """The next minute an awake stay reaches its AC check. Nothing is due while heat is at or below
        AC_AWAKE_HEAT, since heat changes only at midnight."""
        if heat <= AC_AWAKE_HEAT:
            return NEVER
        due = [stay.start + AC_AWAKE_MINUTES for stay in self._stays.values()
               if not stay.asleep and not stay.ac_decided and self._ac.get(stay.room_id)
               and stay.start + AC_AWAKE_MINUTES > minute]
        return min(due, default=NEVER)

    # --- stays and presence -----------------------------------------------------

    def _stay_minute(self, ctx, person: PersonNow, heat: float, rng, out: list[Intent]) -> None:
        if person.room_id is None:
            self._stays.pop(person.person_id, None)
            return
        stay = self._stays.get(person.person_id)
        if stay is None or stay.room_id != person.room_id or stay.asleep != person.asleep:
            stay = _Stay(person.room_id, person.asleep, ctx.minute)
            self._stays[person.person_id] = stay
            if person.asleep and self._ac.get(person.room_id):
                stay.ac_decided = True
                if heat >= SLEEP_AC_MIN_HEAT and rng.random() < person.persona.ac_affinity * heat:
                    self._ac_on(ctx, person, "sleep", rng, out)
        if not person.asleep:
            if (not stay.ac_decided and ctx.minute - stay.start >= AC_AWAKE_MINUTES
                    and heat > AC_AWAKE_HEAT and self._ac.get(person.room_id)):
                stay.ac_decided = True
                if rng.random() < person.persona.ac_affinity * heat:
                    self._ac_on(ctx, person, "timer", rng, out)
        present = not person.asleep or self._rooms[person.room_id].room_type == "bedroom"
        if present and heat > FAN_HEAT and not stay.fan_decided and self._fans.get(person.room_id):
            stay.fan_decided = True
            if rng.random() < person.persona.fan_affinity:
                trigger = "sleep" if person.asleep else "room_entry"
                self._fans_on(ctx, person, person.room_id, fan_speed(heat), trigger, out)

    def _ac_on(self, ctx, person: PersonNow, trigger: str, rng, out: list[Intent]) -> None:
        """The AC's setpoint follows the persona. The actor is the person (see PersonNow.acting)."""
        room = person.room_id
        units = [ld for ld in self._ac[room] if not ctx.devices.is_on(ld.load_id)]
        if not units:
            return
        for ld in units:
            self._power(ld, True, person, "thermal_comfort", trigger, out)
            self._set(ctx, ld, "mode", AC_MODE, person, "thermal_comfort", trigger, out)
            self._set(ctx, ld, "setpoint", person.persona.ac_setpoint, person, "thermal_comfort", trigger, out)
            self._set(ctx, ld, "fan_speed", AC_FAN_SPEED, person, "thermal_comfort", trigger, out)
        if rng.random() < AC_FAN_P:
            self._fans_on(ctx, person, room, AC_COUPLED_FAN_SPEED, trigger, out)

    def _fans_on(self, ctx, person: PersonNow, room: str, speed: float, trigger: str,
                 out: list[Intent]) -> None:
        for ld in self._fans.get(room, ()):
            if ctx.devices.is_on(ld.load_id):
                continue
            self._power(ld, True, person, "thermal_comfort", trigger, out)
            self._set(ctx, ld, "fan_speed", speed, person, "thermal_comfort", trigger, out)

    # --- exits and wake-ups -------------------------------------------------------

    def _leave(self, ctx, people, i: int, rng, out: list[Intent]) -> None:
        person = people[i]
        room = person.prev_room_id
        loads = self._climate_loads(room)
        if not any(ctx.devices.is_on(ld.load_id) for ld in loads):
            return
        if any(q.room_id == room for j, q in enumerate(people) if j != i):
            return  # someone else is still there, asleep or awake
        if any(q.prev_room_id == room and q.room_id != room and not q.asleep for q in people[i + 1:]):
            return  # a later awake leaver in the same minute decides the exit
        if forgets(person.persona, rng):
            return  # everything stays on; the waste tracker notices
        for ld in loads:
            if ctx.devices.is_on(ld.load_id):
                self._power(ld, False, person, "energy_saving", "room_exit", out)

    def _wake(self, ctx, people, i: int, rng, out: list[Intent]) -> None:
        person = people[i]
        room = person.room_id
        units = [ld for ld in self._ac.get(room, ()) if ctx.devices.is_on(ld.load_id)]
        if not units or any(q.room_id == room for j, q in enumerate(people) if j != i):
            return
        if rng.random() < AC_WAKE_OFF_P:
            for ld in units:
                self._power(ld, False, person, "energy_saving", "wake", out)

    # --- exhaust ----------------------------------------------------------------------

    def _exhaust_minute(self, ctx, rng, out: list[Intent]) -> None:
        if not self._exhaust:
            return
        doing: dict[str, PersonNow] = {}  # room id -> the first occupant doing its exhaust activity
        for person in ctx.people:
            if person.room_id is None or person.asleep:
                continue
            if EXHAUST_ACTIVITY_ROOM.get(person.activity) == self._rooms[person.room_id].room_type:
                doing.setdefault(person.room_id, person)
        for ld in self._exhaust:
            on = ctx.devices.is_on(ld.load_id)
            doer = doing.get(ld.room_id)
            if doer is not None:
                self._exhaust_off_at.pop(ld.load_id, None)
                self._exhaust_forgotten.pop(ld.load_id, None)
                self._exhaust_persona[ld.room_id] = doer.persona
                if not on:
                    self._power(ld, True, doer, "air_quality", "activity_start", out)
            elif not on:
                # Switched off outside this policy (a notice): no timer or forgotten mark stays behind.
                self._exhaust_off_at.pop(ld.load_id, None)
                self._exhaust_forgotten.pop(ld.load_id, None)
            elif ld.load_id not in self._exhaust_forgotten:
                timer = self._exhaust_off_at.get(ld.load_id)
                if timer is None:
                    persona = self._exhaust_persona.get(ld.room_id)
                    forget_p = persona.forgetfulness * EXHAUST_FORGET_FACTOR if persona else 0.0
                    due = ctx.minute + int(rng.integers(EXHAUST_OFF_MIN, EXHAUST_OFF_MAX + 1))
                    self._exhaust_off_at[ld.load_id] = (due, forget_p)
                elif ctx.minute >= timer[0]:
                    del self._exhaust_off_at[ld.load_id]
                    if timer[1] > 0 and rng.random() < timer[1]:
                        self._exhaust_forgotten[ld.load_id] = None  # left on, so it is waste
                    else:
                        self._emit(ld, "power", 0.0, None, "none", "air_quality", "timer", "turn_off", out)

    # --- intents ----------------------------------------------------------------------

    def _power(self, ld: Load, on: bool, person: PersonNow | None, intent: str, trigger: str,
               out: list[Intent]) -> None:
        pid, activity = _who(person)
        self._emit(ld, "power", 1.0 if on else 0.0, pid, activity, intent,
                   trigger, "turn_on" if on else "turn_off", out)

    def _set(self, ctx, ld: Load, capability: str, value, person: PersonNow | None, intent: str, trigger: str,
             out: list[Intent]) -> None:
        if _differs(ctx.devices.get(ld.load_id, capability), value):
            pid, activity = _who(person)
            self._emit(ld, capability, value, pid, activity, intent, trigger,
                       _ACTIONS[capability], out)

    def _emit(self, ld: Load, capability: str, value, person_id: str | None, activity: str,
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

    # --- setup ------------------------------------------------------------------------

    def _climate_loads(self, room_id: str | None) -> tuple[Load, ...]:
        if room_id is None:
            return ()
        return self._ac.get(room_id, ()) + self._fans.get(room_id, ())

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
        self._rooms = {room.room_id: room for room in manifest.rooms}
        ac: dict[str, list[Load]] = {}
        fans: dict[str, list[Load]] = {}
        exhaust: list[Load] = []
        for ld in manifest.loads:
            if ld.load_class.startswith(AC_PREFIX):
                ac.setdefault(ld.room_id, []).append(ld)
            elif ld.load_class in CEILING_FANS:
                fans.setdefault(ld.room_id, []).append(ld)
            elif ld.load_class == EXHAUST:
                exhaust.append(ld)
        self._ac = {rid: tuple(lds) for rid, lds in ac.items()}
        self._fans = {rid: tuple(lds) for rid, lds in fans.items()}
        self._climate_rooms = {rid: None for rid in (*self._ac, *self._fans)}
        self._exhaust = tuple(exhaust)
