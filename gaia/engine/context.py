"""What a policy sees each minute: time, sun, people, home devices and occupancy helpers.

The engine builds one ``HomeEnv`` per home (grids, the sun for every minute, the calendar and
a per-minute ``changed`` flag) and one cheap ``TickContext`` per minute. ``people`` is built on
first access, and reused from the previous minute only when the two minutes before it are quiet.
"""

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from gaia.devices.state import HomeDevices
from gaia.engine.waste import WasteTracker
from gaia.home.manifest import HomeManifest
from gaia.household.personas import Member, Persona
from gaia.world.calendar import DayType
from gaia.world.sun import outdoor_lux as _lux_of_altitude

SLEEP_CODE = 0  # fill_grid is seeded with SLEEP -> 0
MINUTES_PER_DAY = 1440


@dataclass(frozen=True, slots=True)
class ActivitySpan:
    """One whole run of an activity for one person, planned for the entire run."""
    person_id: str
    start: int  # minute since the run start
    end: int
    room_id: str | None


@dataclass(frozen=True, slots=True)
class PersonNow:
    person_id: str
    persona: Persona
    room_id: str | None  # None = away from the home
    asleep: bool
    activity: str
    prev_room_id: str | None  # the room one minute earlier (equal to room_id at minute 0)
    prev_asleep: bool
    prev_activity: str  # the activity one minute earlier (equal to activity at minute 0)

    def acting(self) -> tuple[str, str] | None:
        """(person id, activity) when this person may be named as the actor of a command at this
        minute, else None. A person is an actor when awake and at home now (with the activity they
        have now), or when they were awake and at home one minute earlier and changed state this
        minute (fell asleep, left a room, or left home). Then the activity is the one they had one
        minute earlier. Someone asleep or away at both minutes is never an actor."""
        if self.room_id is not None and not self.asleep:
            return self.person_id, self.activity
        if self.prev_room_id is not None and not self.prev_asleep:
            return self.person_id, self.prev_activity
        return None


class DeviceView:
    """Read-only window onto the home's device states."""

    def __init__(self, devices: HomeDevices) -> None:
        self._devices = devices

    def is_on(self, load_id: str) -> bool:
        return self._devices.is_on(load_id)

    def get(self, load_id: str, capability: str):
        return self._devices.get(load_id).get(capability)


class OccupancyGrid:
    """Per-member minute grids (room index, activity code), interned PersonNow objects and the
    ``changed`` flags: minute m differs from m-1 for someone's room, sleep or activity.

    ``reuse_people`` (private, for the equivalence test) turns off the tuple reuse in snapshot.
    """

    def __init__(self, members: tuple[Member, ...], room_ids: list[str], rooms: list[list[int]],
                 acts: list[list[int]], act_names: list[str], reuse_people: bool = True) -> None:
        self._members = members
        self._room_ids = room_ids
        self._rooms = rooms
        self._acts = acts
        self._act_names = act_names
        self._reuse = reuse_people
        self._interned: list[dict[tuple, PersonNow]] = [{} for _ in members]
        self._last: tuple[PersonNow, ...] = ()
        self._last_minute = -2
        self._spans: dict[str, tuple[ActivitySpan, ...]] = {}
        self.changed: list[bool] = self._changed_flags(rooms, acts)

    def spans(self, activity: str) -> tuple[ActivitySpan, ...]:
        """Every run of ``activity`` in the grid, for all members, ordered by start minute then
        member order. Computed once per activity name."""
        cached = self._spans.get(activity)
        if cached is not None:
            return cached
        found: list[tuple[int, int, ActivitySpan]] = []
        if activity in self._act_names:
            code = self._act_names.index(activity)
            for i, member in enumerate(self._members):
                mask = np.asarray(self._acts[i]) == code
                edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
                rooms = np.asarray(self._rooms[i])
                for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                    room = int(rooms[start])
                    found.append((int(start), i, ActivitySpan(
                        member.person_id, int(start), int(end),
                        self._room_ids[room] if room >= 0 else None)))
        spans = tuple(span for _, _, span in sorted(found, key=lambda t: (t[0], t[1])))
        self._spans[activity] = spans
        return spans

    @staticmethod
    def _changed_flags(rooms: list[list[int]], acts: list[list[int]]) -> list[bool]:
        if not rooms:
            return []
        flags = np.zeros(len(rooms[0]), dtype=bool)
        for room_row, act_row in zip(rooms, acts):
            room = np.asarray(room_row)
            act = np.asarray(act_row)  # an activity change is a change, even within one room
            flags[1:] |= (room[1:] != room[:-1]) | (act[1:] != act[:-1])
        return flags.tolist()

    def snapshot(self, m: int) -> tuple[PersonNow, ...]:
        # A PersonNow holds the state at m and at m-1. Both are unchanged only when minutes m-1
        # and m are both quiet, so the previous tuple is valid only then.
        if (self._reuse and m and self._last_minute == m - 1
                and not self.changed[m] and not self.changed[m - 1]):
            self._last_minute = m
            return self._last
        out = []
        for i, member in enumerate(self._members):
            r, a = self._rooms[i][m], self._acts[i][m]
            pr, pa = (r, a) if m == 0 else (self._rooms[i][m - 1], self._acts[i][m - 1])
            key = (r, a, pr, pa)
            person = self._interned[i].get(key)
            if person is None:
                person = PersonNow(
                    person_id=member.person_id,
                    persona=member.persona,
                    room_id=self._room_ids[r] if r >= 0 else None,
                    asleep=a == SLEEP_CODE,
                    activity=self._act_names[a],
                    prev_room_id=self._room_ids[pr] if pr >= 0 else None,
                    prev_asleep=pa == SLEEP_CODE,
                    prev_activity=self._act_names[pa],
                )
                self._interned[i][key] = person
            out.append(person)
        self._last = tuple(out)
        self._last_minute = m
        return self._last


@dataclass(frozen=True, slots=True)
class HomeEnv:
    """Per-home data shared by every tick."""

    manifest: HomeManifest
    devices: DeviceView
    grid: OccupancyGrid
    sun: list[float]  # solar elevation (deg) for every minute of the run
    calendar: tuple[tuple[date, DayType], ...]  # (local date, day type) per local day
    waste: WasteTracker | None = None  # the run's waste tracker; None when nothing tracks waste


@dataclass(frozen=True, slots=True)
class TickContext:
    minute: int  # minutes since local midnight of the start date
    tick_ms: int
    _env: HomeEnv = field(repr=False, compare=False)
    _people: tuple[PersonNow, ...] | None = field(default=None, repr=False, compare=False)

    # --- home -------------------------------------------------------------

    @property
    def manifest(self) -> HomeManifest:
        return self._env.manifest

    @property
    def devices(self) -> DeviceView:
        return self._env.devices

    @property
    def changed(self) -> bool:
        """True when some member's room or sleep state differs from the previous minute."""
        return self._env.grid.changed[self.minute]

    # --- time -------------------------------------------------------------

    @property
    def local_minute_of_day(self) -> int:
        return self.minute % MINUTES_PER_DAY

    @property
    def local_date(self) -> date:
        return self._env.calendar[self.minute // MINUTES_PER_DAY][0]

    @property
    def day_type(self) -> DayType:
        return self._env.calendar[self.minute // MINUTES_PER_DAY][1]

    @property
    def month(self) -> int:
        return self.local_date.month

    def local_date_at(self, minute: int) -> date:
        """Local date of any minute of the run, not only the current one."""
        return self._env.calendar[minute // MINUTES_PER_DAY][0]

    def activity_spans(self, activity: str) -> tuple[ActivitySpan, ...]:
        """Every planned run of ``activity`` in the whole run (a lookahead the routine-driven
        policies use, such as a geyser that heats before a bath)."""
        return self._env.grid.spans(activity)

    # --- waste --------------------------------------------------------------

    def open_waste(self) -> tuple[tuple[str, str], ...]:
        """(load id, room id) of the waste intervals open at the start of this minute."""
        if self._env.waste is None:
            return ()
        return self._env.waste.open_pairs()

    # --- sun --------------------------------------------------------------

    @property
    def sun_altitude_deg(self) -> float:
        return self._env.sun[self.minute]

    @property
    def outdoor_lux(self) -> float:
        return _lux_of_altitude(self._env.sun[self.minute])

    # --- people -----------------------------------------------------------

    @property
    def people(self) -> tuple[PersonNow, ...]:
        if self._people is None:
            object.__setattr__(self, "_people", self._env.grid.snapshot(self.minute))
        return self._people

    def awake_in(self, room_id: str) -> tuple[PersonNow, ...]:
        return tuple(p for p in self.people if p.room_id == room_id and not p.asleep)

    def any_awake_in(self, room_id: str) -> bool:
        for p in self.people:  # a plain loop: this runs every minute for every open waste interval
            if p.room_id == room_id and not p.asleep:
                return True
        return False

    def any_in(self, room_id: str) -> bool:
        """True when someone, awake or asleep, is in the room."""
        for p in self.people:
            if p.room_id == room_id:
                return True
        return False

    def entered(self) -> list[tuple[PersonNow, str]]:
        if not self.changed:
            return []
        return [(p, p.room_id) for p in self.people
                if p.room_id is not None and p.room_id != p.prev_room_id]

    def left(self) -> list[tuple[PersonNow, str]]:
        if not self.changed:
            return []
        return [(p, p.prev_room_id) for p in self.people
                if p.prev_room_id is not None and p.prev_room_id != p.room_id]

    def fell_asleep(self) -> list[PersonNow]:
        if not self.changed:
            return []
        return [p for p in self.people if p.asleep and not p.prev_asleep]

    def woke(self) -> list[PersonNow]:
        if not self.changed:
            return []
        return [p for p in self.people if not p.asleep and p.prev_asleep]
