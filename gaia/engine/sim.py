"""Per-home discrete simulation: a one-minute clock over per-person presence grids.

Light model: only each room's main light. Each tick, a present person turns on a dark
room's light when light is needed. Leaving a room, sleeping in it, or an empty-house
sweep turns lights off. A forgetful exit leaves the light on and records a GtWaste.
Randomness: "home", "household", "plan/<person>/<day>" and one "engine" stream.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from functools import partial

from gaia.behaviour.day_plan import plan_day
from gaia.behaviour.needs import indoor_lux, light_needed
from gaia.engine.protocols import latency_ms
from gaia.engine.timeline import MINUTES_PER_DAY, SLEEP, fill_grid, stitch_days
from gaia.home.archetypes import APARTMENTS, ARCHETYPES
from gaia.home.manifest import HomeManifest, Load
from gaia.household.personas import Household, Member, make_household
from gaia.interaction.choose import choose_source, forgets
from gaia.rng import rng_for
from gaia.schema.events import Domain, Event, EventKind, Source
from gaia.world.calendar import day_type, local_midnight_utc_ms
from gaia.world.cities import CITIES
from gaia.world.sun import outdoor_lux, sun_altitude_deg

MS_PER_MIN = 60_000
MIN_WASTE_MS = MS_PER_MIN  # shorter waste intervals are not emitted as ground truth
CONTEXT_EVERY_MIN = 15
SLEEP_CODE = 0  # fill_grid is seeded with SLEEP -> 0
_SLOT_NAMES = {"own_bedroom": "bedroom", "own_bathroom": "bathroom"}


@dataclass(frozen=True)
class HomeSpec:
    home_id: str
    home_seed: int
    city: str
    archetype: str
    start_date: date
    days: int
    festivals: frozenset[date]


@dataclass(frozen=True)
class GtCause:
    event_id: str
    cause: str  # need_light | leave_room | sleep | forgot_cleanup
    person_id: str | None
    activity: str | None


@dataclass(frozen=True)
class GtWaste:
    load_id: str
    person_id: str
    start_ts_utc: int
    end_ts_utc: int | None


@dataclass(frozen=True)
class ContextRow:
    ts_utc: int
    home_id: str
    outdoor_lux: float
    sun_altitude_deg: float


@dataclass
class SimResult:
    manifest: HomeManifest
    household: Household
    events: list[Event]
    context: list[ContextRow]
    gt_cause: list[GtCause]
    gt_waste: list[GtWaste]


def _room_of(slot: str | None, member: Member, home_id: str, room_index: dict[str, int]) -> int:
    """Map a symbolic day-plan slot to a room index for this member (-1 = away)."""
    if slot is None:
        return -1
    name = getattr(member, _SLOT_NAMES[slot]) if slot in _SLOT_NAMES else slot
    return room_index[f"{home_id}:{name}"]


class _Run:
    def __init__(self, spec: HomeSpec, manifest: HomeManifest, household: Household,
                 start_ms: int, rooms: list[list[int]], acts: list[list[int]],
                 act_names: list[str]) -> None:
        self.spec = spec
        self.manifest = manifest
        self.members = household.members
        self.city = CITIES[spec.city]
        self.start_ms = start_ms
        self.rooms = rooms
        self.acts = acts
        self.act_names = act_names
        self.rng = rng_for(spec.home_seed, "engine")
        self.mains: list[Load | None] = [manifest.main_light(r.room_id) for r in manifest.rooms]
        self.light_on = [False] * len(manifest.rooms)
        self.last_state_ts = [-1] * len(manifest.rooms)  # per main-light load, for ordering
        self.open_waste: dict[int, tuple[str, int]] = {}
        self.events: list[Event] = []
        self.gt_cause: list[GtCause] = []
        self.gt_waste: list[GtWaste] = []
        self.context: list[ContextRow] = []
        self._counter = 0
        self._alt_ts: int | None = None
        self._alt = 0.0

    def run(self, total: int) -> None:
        n = len(self.members)
        for m in range(total):
            tick = self.start_ms + m * MS_PER_MIN
            if m % CONTEXT_EVERY_MIN == 0:
                alt = self._altitude(tick)
                self.context.append(ContextRow(tick, self.spec.home_id, outdoor_lux(alt), alt))
            for i in range(n):
                if m > 0:
                    self._leave(i, m, tick)
                    self._fall_asleep(i, m, tick)
                self._close_waste_on_entry(i, m, tick)
                self._presence(i, m, tick)

    def finish(self) -> None:
        """Close the run: intervals still open at the end have no end timestamp.

        Waste shorter than MIN_WASTE_MS is dropped here; its events and causes are kept.
        """
        self.events.sort(key=lambda e: (e.ts_utc, e.event_id))
        for r, (pid, start) in self.open_waste.items():
            self.gt_waste.append(GtWaste(self.mains[r].load_id, pid, start, None))
        self.open_waste.clear()
        self.gt_waste = [w for w in self.gt_waste
                         if w.end_ts_utc is None or w.end_ts_utc - w.start_ts_utc >= MIN_WASTE_MS]

    # --- per-tick rules -------------------------------------------------------

    def _leave(self, i: int, m: int, tick: int) -> None:
        prev, cur = self.rooms[i][m - 1], self.rooms[i][m]
        if prev == cur or prev < 0 or not self.light_on[prev] or self._awake_occupied(prev, m, skip=i):
            return
        if any(self._leaves_awake(j, m, prev) for j in range(i + 1, len(self.members))):
            return  # a later awake leaver in the same tick decides the exit
        member = self.members[i]
        if forgets(member.persona, self.rng):
            self.open_waste.setdefault(prev, (member.person_id, tick))
            return
        source = choose_source(member.persona, self.rng)
        self._switch(prev, False, tick, source, member.person_id, "leave_room", member.person_id,
                     self._act(i, m))

    def _fall_asleep(self, i: int, m: int, tick: int) -> None:
        if self.acts[i][m] != SLEEP_CODE or self.acts[i][m - 1] == SLEEP_CODE:
            return
        member = self.members[i]
        bed = self.rooms[i][m]
        awake_others = [j for j in range(len(self.members))
                        if j != i and self.acts[j][m] != SLEEP_CODE]
        if bed >= 0 and self.light_on[bed] and not any(self.rooms[j][m] == bed for j in awake_others):
            source = choose_source(member.persona, self.rng)
            self._switch(bed, False, tick, source, member.person_id, "sleep", member.person_id, SLEEP)
        if awake_others:
            return
        for r in range(len(self.mains)):
            if self.light_on[r] and not self._occupied(r, m):
                if self.rng.random() < 0.5:
                    self._switch(r, False, tick, Source.app, member.person_id, "forgot_cleanup",
                                 member.person_id, SLEEP)

    def _close_waste_on_entry(self, i: int, m: int, tick: int) -> None:
        """A waste interval means light on with no awake member in the room: an awake entry ends it."""
        r = self.rooms[i][m]
        if r >= 0 and self.acts[i][m] != SLEEP_CODE and r in self.open_waste:
            pid, start = self.open_waste.pop(r)
            self.gt_waste.append(GtWaste(self.mains[r].load_id, pid, start, tick))

    def _presence(self, i: int, m: int, tick: int) -> None:
        act, r = self.acts[i][m], self.rooms[i][m]
        if act == SLEEP_CODE or r < 0 or self.light_on[r] or self.mains[r] is None:
            return
        name = self.act_names[act]
        indoor = indoor_lux(outdoor_lux(self._altitude(tick)), self.manifest.rooms[r])
        member = self.members[i]
        if light_needed(indoor, member.persona, name):
            source = choose_source(member.persona, self.rng)
            self._switch(r, True, tick, source, member.person_id, "need_light", member.person_id, name)

    # --- helpers --------------------------------------------------------------

    def _leaves_awake(self, j: int, m: int, room: int) -> bool:
        """Member j is awake and leaves `room` at minute m (it was there at m-1)."""
        return (self.rooms[j][m - 1] == room and self.rooms[j][m] != room
                and self.acts[j][m] != SLEEP_CODE)

    def _occupied(self, r: int, m: int, skip: int | None = None) -> bool:
        return any(j != skip and self.rooms[j][m] == r for j in range(len(self.members)))

    def _awake_occupied(self, r: int, m: int, skip: int) -> bool:
        return any(j != skip and self.acts[j][m] != SLEEP_CODE and self.rooms[j][m] == r
                   for j in range(len(self.members)))

    def _act(self, i: int, m: int) -> str:
        return self.act_names[self.acts[i][m]]

    def _altitude(self, tick: int) -> float:
        if tick != self._alt_ts:
            self._alt_ts = tick
            self._alt = sun_altitude_deg(self.city, tick)
        return self._alt

    def _next_id(self) -> str:
        eid = f"{self.spec.home_id}:{self._counter:09d}"
        self._counter += 1
        return eid

    def _switch(self, r: int, on: bool, tick: int, source: Source, actor: str, cause: str,
                person_id: str | None, activity: str | None) -> None:
        """Emit a main-light command and its state; update light, waste and cause records."""
        load = self.mains[r]
        room = self.manifest.rooms[r]
        new, old = (1.0, 0.0) if on else (0.0, 1.0)
        cmd_ts = max(tick + int(self.rng.integers(0, MS_PER_MIN)), self.last_state_ts[r] + 1)
        cmd_id = self._next_id()
        self._emit(cmd_ts, EventKind.command, r, new, old, source, actor, None, cmd_id)
        self.gt_cause.append(GtCause(cmd_id, cause, person_id, activity))
        state_ts = max(cmd_ts + latency_ms(load.protocol, self.rng), self.last_state_ts[r] + 1)
        self.last_state_ts[r] = state_ts
        self._emit(state_ts, EventKind.state, r, new, old, source, actor, cmd_id, self._next_id())
        self.light_on[r] = on
        if not on and r in self.open_waste:
            pid, start = self.open_waste.pop(r)
            self.gt_waste.append(GtWaste(load.load_id, pid, start, state_ts))

    def _emit(self, ts: int, kind: EventKind, r: int, value: float, prev: float, source: Source,
              actor: str, ref: str | None, event_id: str) -> None:
        load = self.mains[r]
        room = self.manifest.rooms[r]
        self.events.append(Event(
            ts_utc=ts, tz_offset_min=self.city.tz_offset_min, home_id=self.spec.home_id,
            event_id=event_id, kind=kind, room_id=room.room_id, room_type=room.room_type,
            load_id=load.load_id, load_class=load.load_class, capability="power",
            value=value, prev_value=prev, source=source, actor=actor, protocol=load.protocol,
            ref_id=ref, domain=Domain.synthetic,
        ))


def _build_grids(spec: HomeSpec) -> tuple[HomeManifest, Household, list[list[int]], list[list[int]], list[str]]:
    """Build the manifest, household and per-person minute grids (rooms, activity codes, names)."""
    manifest = ARCHETYPES[spec.archetype](spec.home_id, spec.city, rng_for(spec.home_seed, "home"))
    apartment = APARTMENTS[spec.archetype]
    household = make_household(manifest, apartment, rng_for(spec.home_seed, "household"))
    total = spec.days * MINUTES_PER_DAY
    room_index = {room.room_id: i for i, room in enumerate(manifest.rooms)}

    act_codes = {SLEEP: SLEEP_CODE}
    rooms: list[list[int]] = []
    acts: list[list[int]] = []
    for member in household.members:
        plans = []
        for d in range(spec.days):
            kind = day_type(spec.start_date + timedelta(days=d), spec.festivals)
            rng = rng_for(spec.home_seed, f"plan/{member.person_id}/{d}")
            plans.append(plan_day(member.persona, kind, rng))
        resolve = partial(_room_of, member=member, home_id=spec.home_id, room_index=room_index)
        room_arr, act_arr = fill_grid(stitch_days(plans, resolve), total, act_codes)
        rooms.append(room_arr.tolist())
        acts.append(act_arr.tolist())
    act_names = [name for name, _ in sorted(act_codes.items(), key=lambda kv: kv[1])]
    return manifest, household, rooms, acts, act_names


def simulate_home(spec: HomeSpec) -> SimResult:
    manifest, household, rooms, acts, act_names = _build_grids(spec)
    start_ms = local_midnight_utc_ms(spec.start_date, CITIES[spec.city].tz_offset_min)
    total = spec.days * MINUTES_PER_DAY
    run = _Run(spec, manifest, household, start_ms, rooms, acts, act_names)
    run.run(total)
    run.finish()
    return SimResult(manifest, household, run.events, run.context, run.gt_cause, run.gt_waste)
