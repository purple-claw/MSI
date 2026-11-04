from collections import defaultdict
from datetime import date

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gaia.engine.sim import SLEEP_CODE, HomeSpec, _build_grids, _Run, simulate_home
from gaia.engine.timeline import MINUTES_PER_DAY
from gaia.registry.registry import get_load_class
from gaia.schema.events import EventKind
from gaia.world.calendar import local_midnight_utc_ms

ARCHETYPES = ["apartment_1rk", "apartment_1bhk", "apartment_2bhk", "apartment_3bhk", "apartment_4bhk",
              "independent_house", "villa"]
DAYS = 3
START = date(2026, 11, 2)
SEEDS = st.integers(0, 10**6)


def _spec(seed: int, archetype: str) -> HomeSpec:
    return HomeSpec(home_id="h1", home_seed=seed, city="hyderabad", archetype=archetype,
                    start_date=START, days=DAYS, festivals=frozenset())


def _run(seed: int, archetype: str):
    r = simulate_home(_spec(seed, archetype))
    assert r.events, "event list must be non-empty"
    return r


@pytest.mark.parametrize("archetype", ARCHETYPES)
@settings(max_examples=25, deadline=None)
@given(seed=SEEDS)
def test_ts_monotonic(archetype, seed):
    ts = [e.ts_utc for e in _run(seed, archetype).events]
    assert all(a <= b for a, b in zip(ts, ts[1:]))


@pytest.mark.parametrize("archetype", ARCHETYPES)
@settings(max_examples=25, deadline=None)
@given(seed=SEEDS)
def test_every_state_has_command(archetype, seed):
    events = _run(seed, archetype).events
    ts_of = {e.event_id: e.ts_utc for e in events if e.kind == EventKind.command}
    for e in events:
        if e.kind == EventKind.state:
            assert e.ref_id in ts_of and ts_of[e.ref_id] < e.ts_utc


@pytest.mark.parametrize("archetype", ARCHETYPES)
@settings(max_examples=25, deadline=None)
@given(seed=SEEDS)
def test_no_redundant_state(archetype, seed):
    # Keyed by (load, capability): a dimmable's power state and its level can share a value.
    last: dict[tuple[str, str], float] = {}
    for e in _run(seed, archetype).events:
        if e.kind != EventKind.state:
            continue
        key = (e.load_id, e.capability)
        if key in last:
            assert last[key] != e.value
        last[key] = e.value


@pytest.mark.parametrize("archetype", ARCHETYPES)
@settings(max_examples=25, deadline=None)
@given(seed=SEEDS)
def test_labels_cover_commands(archetype, seed):
    r = _run(seed, archetype)
    commands = [e.event_id for e in r.events if e.kind == EventKind.command]
    labelled = [lb.event_id for lb in r.labels]
    assert sorted(commands) == sorted(labelled) and len(set(labelled)) == len(labelled)


@pytest.mark.parametrize("archetype", ARCHETYPES)
@settings(max_examples=25, deadline=None)
@given(seed=SEEDS)
def test_waste_intervals_valid(archetype, seed):
    for w in _run(seed, archetype).gt_waste:
        assert w.end_ts_utc is None or w.end_ts_utc > w.start_ts_utc


@pytest.mark.parametrize("archetype", ARCHETYPES)
@settings(max_examples=25, deadline=None)
@given(seed=SEEDS)
def test_waste_never_overlaps_awake_occupancy(archetype, seed):
    spec = _spec(seed, archetype)
    r = _run(seed, archetype)
    _, _, rooms, acts, _ = _build_grids(spec)
    total = DAYS * MINUTES_PER_DAY
    start = local_midnight_utc_ms(START, 330)
    room_index = {room.room_id: i for i, room in enumerate(r.manifest.rooms)}
    room_of_load = {ld.load_id: room_index[ld.room_id] for ld in r.manifest.loads}
    for w in r.gt_waste:
        if get_load_class(w.load_class).waste_rule != "unattended":
            continue  # overrun waste (e.g. fan.exhaust) ignores occupancy by design, see gaia/engine/waste.py
        idx = room_of_load[w.load_id]
        lo = (w.start_ts_utc - start) // 60_000
        # ceil: an interval ending mid-minute still counts that minute as covered
        hi = total if w.end_ts_utc is None else -(-(w.end_ts_utc - start) // 60_000)
        for m in range(lo, min(hi, total)):
            for j in range(len(rooms)):
                assert not (acts[j][m] != 0 and rooms[j][m] == idx), (w, m, j)


@pytest.mark.parametrize("archetype", ARCHETYPES)
@settings(max_examples=25, deadline=None)
@given(seed=SEEDS)
def test_per_load_states_alternate_and_follow_commands(archetype, seed):
    events = _run(seed, archetype).events
    cmd_ts = {e.event_id: e.ts_utc for e in events if e.kind == EventKind.command}
    by_load: dict[str, list] = defaultdict(list)
    for e in events:
        if e.kind == EventKind.state:
            by_load[e.load_id].append(e)
    for states in by_load.values():
        states.sort(key=lambda e: e.ts_utc)
        ts = [s.ts_utc for s in states]
        assert all(a < b for a, b in zip(ts, ts[1:]))
        assert states[0].value == 1.0  # lights start off, so the first state is on
        assert all(s.ts_utc > cmd_ts[s.ref_id] for s in states)
        by_cap: dict[str, list] = defaultdict(list)
        for s in states:  # alternation is per capability: power and level may share a value
            by_cap[s.capability].append(s)
        for cap_states in by_cap.values():
            assert all(a.value != b.value for a, b in zip(cap_states, cap_states[1:]))


@pytest.mark.parametrize("archetype", ARCHETYPES)
@settings(max_examples=25, deadline=None)
@given(seed=SEEDS)
def test_emitted_waste_is_at_least_a_minute(archetype, seed):
    for w in _run(seed, archetype).gt_waste:
        assert w.end_ts_utc is None or w.end_ts_utc - w.start_ts_utc >= 60_000, w



@pytest.mark.parametrize("archetype", ARCHETYPES)
@settings(max_examples=25, deadline=None)
@given(seed=SEEDS)
def test_lit_unoccupied_minutes_lie_inside_waste(archetype, seed):
    # Each minute that ends with a main light on and no awake member in its room must be covered
    # by an emitted GtWaste interval. Light state is the simulator's decision-time state, so a
    # leave-room switch-off with command jitter is not counted as lit.
    spec = _spec(seed, archetype)
    manifest, household, rooms, acts, names = _build_grids(spec)
    total = DAYS * MINUTES_PER_DAY
    start = local_midnight_utc_ms(START, 330)
    run = _Run(spec, manifest, household, start, rooms, acts, names)
    n = len(household.members)
    lit_unoccupied: list[tuple[str, int]] = []  # (load id, minute)
    for m in range(total):
        run.step(m)
        for r, room in enumerate(manifest.rooms):
            main = manifest.main_light(room.room_id)
            if main is None or not run.devices.is_on(main.load_id):
                continue
            if not any(rooms[j][m] == r and acts[j][m] != SLEEP_CODE for j in range(n)):
                lit_unoccupied.append((main.load_id, m))
    run.finish()
    for load_id, m in lit_unoccupied:
        tick = start + m * 60_000
        assert any(w.load_id == load_id and w.start_ts_utc <= tick
                   and (w.end_ts_utc is None or w.end_ts_utc > tick)
                   for w in run.gt_waste), (load_id, m)
