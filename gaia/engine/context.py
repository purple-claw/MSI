"""What a policy sees each minute: time, sun, people, home devices and occupancy helpers.

``TickContext`` is cheap to build. Per-minute room and activity grids are precomputed by the
engine, the sun is evaluated only when a policy asks for it, and ``people`` is built once per
tick on first access and interned, so a quiet minute allocates almost nothing.
"""

from dataclasses import dataclass, field
from datetime import date

from gaia.devices.state import HomeDevices
from gaia.home.manifest import HomeManifest
from gaia.household.personas import Member, Persona
from gaia.world.cities import City
from gaia.world.calendar import DayType
from gaia.world.sun import outdoor_lux as _lux_of_altitude
from gaia.world.sun import sun_altitude_deg as _altitude_of

SLEEP_CODE = 0  # fill_grid is seeded with SLEEP -> 0
MINUTES_PER_DAY = 1440


@dataclass(frozen=True, slots=True)
class PersonNow:
    person_id: str
    persona: Persona
    room_id: str | None  # None = away from the home
    asleep: bool
    activity: str
    prev_room_id: str | None  # the room one minute earlier (equal to room_id at minute 0)
    prev_asleep: bool


class DeviceView:
    """Read-only window onto the home's device states."""

    def __init__(self, devices: HomeDevices) -> None:
        self._devices = devices

    def is_on(self, load_id: str) -> bool:
        return self._devices.is_on(load_id)

    def get(self, load_id: str, capability: str):
        return self._devices.get(load_id).get(capability)


class SunClock:
    """Solar altitude at one city, memoised for the most recent tick."""

    def __init__(self, city: City) -> None:
        self._city = city
        self._tick: int | None = None
        self._altitude = 0.0

    def altitude(self, tick_ms: int) -> float:
        if tick_ms != self._tick:
            self._tick = tick_ms
            self._altitude = _altitude_of(self._city, tick_ms)
        return self._altitude


class OccupancyGrid:
    """Per-member minute grids (room index, activity code) and the interned PersonNow objects."""

    def __init__(self, members: tuple[Member, ...], room_ids: list[str], rooms: list[list[int]],
                 acts: list[list[int]], act_names: list[str]) -> None:
        self._members = members
        self._room_ids = room_ids
        self._rooms = rooms
        self._acts = acts
        self._act_names = act_names
        self._interned: list[dict[tuple, PersonNow]] = [{} for _ in members]

    def snapshot(self, m: int) -> tuple[PersonNow, ...]:
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
                )
                self._interned[i][key] = person
            out.append(person)
        return tuple(out)


@dataclass(frozen=True, slots=True)
class TickContext:
    minute: int  # minutes since local midnight of the start date
    tick_ms: int
    manifest: HomeManifest
    devices: DeviceView
    _grid: OccupancyGrid = field(repr=False, compare=False)
    _sun: SunClock = field(repr=False, compare=False)
    _calendar: tuple[tuple[date, DayType], ...] = field(repr=False, compare=False)
    _people: tuple[PersonNow, ...] | None = field(default=None, repr=False, compare=False)

    # --- time -------------------------------------------------------------

    @property
    def local_minute_of_day(self) -> int:
        return self.minute % MINUTES_PER_DAY

    @property
    def local_date(self) -> date:
        return self._calendar[self.minute // MINUTES_PER_DAY][0]

    @property
    def day_type(self) -> DayType:
        return self._calendar[self.minute // MINUTES_PER_DAY][1]

    @property
    def month(self) -> int:
        return self.local_date.month

    # --- sun --------------------------------------------------------------

    @property
    def sun_altitude_deg(self) -> float:
        return self._sun.altitude(self.tick_ms)

    @property
    def outdoor_lux(self) -> float:
        return _lux_of_altitude(self.sun_altitude_deg)

    # --- people -----------------------------------------------------------

    @property
    def people(self) -> tuple[PersonNow, ...]:
        if self._people is None:
            object.__setattr__(self, "_people", self._grid.snapshot(self.minute))
        return self._people

    def awake_in(self, room_id: str) -> tuple[PersonNow, ...]:
        return tuple(p for p in self.people if p.room_id == room_id and not p.asleep)

    def entered(self) -> list[tuple[PersonNow, str]]:
        return [(p, p.room_id) for p in self.people
                if p.room_id is not None and p.room_id != p.prev_room_id]

    def left(self) -> list[tuple[PersonNow, str]]:
        return [(p, p.prev_room_id) for p in self.people
                if p.prev_room_id is not None and p.prev_room_id != p.room_id]

    def fell_asleep(self) -> list[PersonNow]:
        return [p for p in self.people if p.asleep and not p.prev_asleep]

    def woke(self) -> list[PersonNow]:
        return [p for p in self.people if not p.asleep and p.prev_asleep]
