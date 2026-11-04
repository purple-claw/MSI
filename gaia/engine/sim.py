"""Per-home discrete simulation: a one-minute clock over per-person presence grids.

Each minute the engine builds one TickContext and runs the policies in POLICY_ORDER. Each
policy's intents go to the Actuator, which emits command and state events and one Label per
command. The WasteTracker follows the resulting states and records ground-truth waste.
Randomness: "home", "household", "plan/<person>/<day>"; per home the actuator streams
"interaction" (sources), "protocol" (latency) and "engine" (command jitter); and per policy
"policy/<name>".
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import partial
from itertools import groupby

from gaia.behaviour.day_plan import plan_day
from gaia.devices.state import HomeDevices
from gaia.engine.actuator import Actuator
from gaia.engine.context import (
    SLEEP_CODE,
    DeviceView,
    HomeEnv,
    OccupancyGrid,
    TickContext,
)
from gaia.engine.timeline import MINUTES_PER_DAY, SLEEP, fill_grid, stitch_days
from gaia.engine.waste import GtWaste, WasteTracker
from gaia.home.archetypes import ARCHETYPE_SPECS, ARCHETYPES
from gaia.home.manifest import HomeManifest
from gaia.household.personas import Household, Member, make_household
from gaia.labels.model import Label
from gaia.policies import POLICY_ORDER
from gaia.policies.base import Policy
from gaia.rng import rng_for
from gaia.schema.events import Event, EventKind
from gaia.world.calendar import day_type, local_midnight_utc_ms
from gaia.world.cities import CITIES
from gaia.world.sun import sun_altitude_grid

MS_PER_MIN = 60_000
CONTEXT_EVERY_MIN = 15
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
class GtPresence:
    """A maximal run of constant room (None = away) and sleep state for one person."""

    person_id: str
    room_id: str | None
    start_ts_utc: int
    end_ts_utc: int
    asleep: bool


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
    gt_waste: list[GtWaste]
    gt_presence: list[GtPresence]
    spec: HomeSpec
    labels: list[Label] = field(default_factory=list)


def _room_of(slot: str | None, member: Member, home_id: str, room_index: dict[str, int]) -> int:
    """Map a symbolic day-plan slot to a room index for this member (-1 = away)."""
    if slot is None:
        return -1
    name = getattr(member, _SLOT_NAMES[slot]) if slot in _SLOT_NAMES else slot
    return room_index[f"{home_id}:{name}"]


class _Run:
    def __init__(self, spec: HomeSpec, manifest: HomeManifest, household: Household,
                 start_ms: int, rooms: list[list[int]], acts: list[list[int]],
                 act_names: list[str], policy_order: Sequence[type[Policy]] = POLICY_ORDER,
                 reuse_people: bool = True) -> None:
        names = [cls.name for cls in policy_order]
        if len(set(names)) != len(names):
            raise ValueError(f"policy names must be unique within a run: {names}")
        city = CITIES[spec.city]
        self.spec = spec
        self.manifest = manifest
        self.household = household
        self.members = household.members
        self.start_ms = start_ms
        self.devices = HomeDevices(manifest)
        total = len(acts[0]) if acts else 0
        days = -(-total // MINUTES_PER_DAY)
        calendar = tuple(
            (d, day_type(d, spec.festivals))
            for d in (spec.start_date + timedelta(days=k) for k in range(days))
        )
        self.home_env = HomeEnv(
            manifest=manifest,
            devices=DeviceView(self.devices),
            grid=OccupancyGrid(self.members, [room.room_id for room in manifest.rooms],
                               rooms, acts, act_names, reuse_people),
            sun=sun_altitude_grid(city, start_ms, total),
            calendar=calendar,
        )
        self.actuator = Actuator(
            home_id=spec.home_id, manifest=manifest, devices=self.devices,
            tz_offset_min=city.tz_offset_min, home_seed=spec.home_seed,
            personas_by_id={m.person_id: m.persona for m in self.members},
            on_state=self._on_state,
        )
        self.policies = [cls() for cls in policy_order]
        self._policy_rngs = [rng_for(spec.home_seed, f"policy/{name}") for name in names]
        self.waste = WasteTracker(manifest)
        self.gt_waste: list[GtWaste] = []
        self.context: list[ContextRow] = []
        self._last_ctx: TickContext | None = None

    @property
    def events(self) -> list[Event]:
        return self.actuator.events

    @property
    def labels(self) -> list[Label]:
        return self.actuator.labels

    def run(self, total: int) -> None:
        for m in range(total):
            self.step(m)

    def step(self, m: int) -> None:
        """Simulate minute ``m``: the context row, then each policy's intents, then waste."""
        ctx = TickContext(minute=m, tick_ms=self.start_ms + m * MS_PER_MIN, _env=self.home_env)
        if m % CONTEXT_EVERY_MIN == 0:
            self.context.append(ContextRow(ctx.tick_ms, self.spec.home_id, ctx.outdoor_lux,
                                           ctx.sun_altitude_deg))
        for policy, rng in zip(self.policies, self._policy_rngs):
            intents = policy.tick(ctx, rng)
            if intents:
                self.actuator.apply(intents, ctx.tick_ms)
        self.waste.on_tick(ctx)
        self._last_ctx = ctx

    def finish(self) -> None:
        """Close the run: intervals still open at the end have no end timestamp.

        Waste shorter than MIN_WASTE_MS is dropped here; its events and labels are kept.
        """
        if self._last_ctx is not None:
            for policy in self.policies:
                finish = getattr(policy, "finish", None)
                if finish is not None:
                    finish(self._last_ctx)
        self.actuator.events.sort(key=lambda e: (e.ts_utc, e.event_id))
        self.gt_waste = self.waste.finish()

    # --- helpers --------------------------------------------------------------

    def _on_state(self, load_id: str, capability: str, ts_ms: int) -> None:
        """Only the capability that says on or off moves the waste tracker."""
        state = self.devices.get(load_id)
        if capability == state.on_off_capability():
            self.waste.on_state(load_id, ts_ms, state.is_on())


def _build_grids(spec: HomeSpec) -> tuple[HomeManifest, Household, list[list[int]], list[list[int]], list[str]]:
    """Build the manifest, household and per-person minute grids (rooms, activity codes, names)."""
    manifest = ARCHETYPES[spec.archetype](spec.home_id, spec.city, rng_for(spec.home_seed, "home"))
    archetype = ARCHETYPE_SPECS[spec.archetype]
    household = make_household(manifest, archetype, rng_for(spec.home_seed, "household"))
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


def _presence_key(room_and_act: tuple[int, int]) -> tuple[int, bool]:
    room_idx, act_code = room_and_act
    return room_idx, act_code == SLEEP_CODE


def _presence_runs(start_ms: int, manifest: HomeManifest, household: Household,
                   rooms: list[list[int]], acts: list[list[int]]) -> list[GtPresence]:
    """Per person, the maximal runs of constant (room, asleep) over the whole grid."""
    runs: list[GtPresence] = []
    for i, member in enumerate(household.members):
        minute = 0
        for (room_idx, asleep), group in groupby(zip(rooms[i], acts[i]), key=_presence_key):
            length = sum(1 for _ in group)
            start = start_ms + minute * MS_PER_MIN
            room_id = None if room_idx < 0 else manifest.rooms[room_idx].room_id
            runs.append(GtPresence(member.person_id, room_id, start,
                                   start + length * MS_PER_MIN, asleep))
            minute += length
    return runs


def simulate_home(spec: HomeSpec, policy_order: Sequence[type[Policy]] = POLICY_ORDER) -> SimResult:
    manifest, household, rooms, acts, act_names = _build_grids(spec)
    start_ms = local_midnight_utc_ms(spec.start_date, CITIES[spec.city].tz_offset_min)
    total = spec.days * MINUTES_PER_DAY
    run = _Run(spec, manifest, household, start_ms, rooms, acts, act_names, policy_order)
    run.run(total)
    run.finish()
    presence = _presence_runs(start_ms, manifest, household, rooms, acts)
    return SimResult(manifest, household, run.events, run.context, run.gt_waste, presence, spec,
                     labels=run.labels)
