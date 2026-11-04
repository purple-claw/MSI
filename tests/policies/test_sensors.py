from datetime import date

from gaia.engine.sim import SLEEP_CODE, HomeSpec, _build_grids, _Run, simulate_home
from gaia.engine.timeline import MINUTES_PER_DAY
from gaia.home.archetypes import ARCHETYPES
from gaia.policies import POLICY_ORDER
from gaia.registry.registry import get_load_class
from gaia.rng import rng_for
from gaia.schema.events import EventKind
from gaia.world.calendar import local_midnight_utc_ms
from gaia.world.cities import CITIES

MS = 60_000
START = date(2026, 11, 2)
TZ = CITIES["hyderabad"].tz_offset_min
START_MS = local_midnight_utc_ms(START, TZ)
NO_NOTICE = [cls for cls in POLICY_ORDER if cls.name != "notice"]
RICH_CASES = [("apartment_3bhk", "rich"), ("villa", "rich"), ("independent_house", "some")]


def _spec(seed: int, archetype: str, days: int = 2) -> HomeSpec:
    return HomeSpec(home_id="h1", home_seed=seed, city="hyderabad", archetype=archetype,
                    start_date=START, days=days, festivals=frozenset())


def _seed_with(archetype: str, richness: str) -> int:
    for seed in range(1, 500):
        if ARCHETYPES[archetype]("h1", "hyderabad", rng_for(seed, "home")).sensor_richness == richness:
            return seed
    raise AssertionError(f"no seed gives {archetype} richness {richness}")


def _sensor_events(result, load_class: str, capability: str):
    loads = {ld.load_id for ld in result.manifest.loads if ld.load_class == load_class}
    return [e for e in result.events if e.kind == EventKind.sensor and e.load_id in loads
            and e.capability == capability]


def _awake_in(rooms, acts, room_idx: int, m: int) -> bool:
    return any(rooms[j][m] == room_idx and acts[j][m] != SLEEP_CODE for j in range(len(rooms)))


def _occupied(rooms, room_idx: int, m: int) -> bool:
    return any(rooms[j][m] == room_idx for j in range(len(rooms)))


def test_motion_consistent_with_presence():
    for arch, richness in RICH_CASES:
        spec = _spec(_seed_with(arch, richness), arch)
        result = simulate_home(spec)
        manifest, _, rooms, acts, _ = _build_grids(spec)
        room_idx = {room.room_id: i for i, room in enumerate(manifest.rooms)}
        ones = [e for e in _sensor_events(result, "sensor.motion", "occupancy") if e.value == 1.0]
        assert ones, arch
        for e in ones:
            m = (e.ts_utc - START_MS) // MS
            assert _awake_in(rooms, acts, room_idx[e.room_id], m), (arch, e)


def test_no_motion_from_sleepers():
    for arch, richness in RICH_CASES:
        spec = _spec(_seed_with(arch, richness), arch)
        result = simulate_home(spec)
        manifest, _, rooms, acts, _ = _build_grids(spec)
        room_idx = {room.room_id: i for i, room in enumerate(manifest.rooms)}
        for e in _sensor_events(result, "sensor.motion", "occupancy"):
            if e.value != 1.0:
                continue
            m = (e.ts_utc - START_MS) // MS
            idx = room_idx[e.room_id]
            sleepers_only = _occupied(rooms, idx, m) and not _awake_in(rooms, acts, idx, m)
            assert not sleepers_only, (arch, e)


def test_presence_covers_sleep():
    for arch, richness in RICH_CASES:
        spec = _spec(_seed_with(arch, richness), arch)
        result = simulate_home(spec)
        manifest, _, rooms, acts, _ = _build_grids(spec)
        room_idx = {room.room_id: i for i, room in enumerate(manifest.rooms)}
        total = spec.days * MINUTES_PER_DAY
        for ld in manifest.loads:
            if ld.load_class != "sensor.presence":
                continue
            changes = sorted(((e.ts_utc - START_MS) // MS, e.value)
                             for e in _sensor_events(result, "sensor.presence", "occupancy")
                             if e.load_id == ld.load_id)
            idx = room_idx[ld.room_id]
            state, k = 0.0, 0
            for m in range(total):
                while k < len(changes) and changes[k][0] <= m:
                    state = changes[k][1]
                    k += 1
                assert state == (1.0 if _occupied(rooms, idx, m) else 0.0), (arch, ld.load_id, m)


def test_energy_meter_matches_sum():
    arch = "apartment_3bhk"
    spec = _spec(_seed_with(arch, "rich"), arch, days=1)
    manifest, household, rooms, acts, names = _build_grids(spec)
    run = _Run(spec, manifest, household, START_MS, rooms, acts, names, NO_NOTICE)
    meters = [ld for ld in manifest.loads if ld.load_class == "meter.energy"]
    assert len(meters) == 1
    readings = 0
    for m in range(spec.days * MINUTES_PER_DAY):
        before = len(run.events)
        run.step(m)
        new = run.events[before:]
        if m % 5:
            assert not [e for e in new if e.load_id == meters[0].load_id]
            continue
        total = 0.0
        for ld in manifest.loads:
            power = get_load_class(ld.load_class).power_w
            if run.devices.is_on(ld.load_id):
                total += power.get("on", 0.0)
            else:
                total += power.get("standby", 0.0)
        (reading,) = [e for e in new if e.load_id == meters[0].load_id]
        assert reading.value == float(round(total)), m
        readings += 1
    assert readings == spec.days * MINUTES_PER_DAY // 5


def test_richness_none_has_no_sensor_events():
    seed = _seed_with("apartment_1rk", "none")
    result = simulate_home(_spec(seed, "apartment_1rk"))
    assert not [ld for ld in result.manifest.loads if ld.load_class.startswith(("sensor.", "meter."))]
    assert not [e for e in result.events if e.kind == EventKind.sensor]
    assert not [lb for lb in result.labels if lb.source_ref == "emit_sensor"]


def test_rich_home_adds_expected_sensor_loads():
    manifest = ARCHETYPES["apartment_3bhk"]("h1", "hyderabad", rng_for(_seed_with("apartment_3bhk", "rich"), "home"))
    by_class = {}
    for ld in manifest.loads:
        if ld.load_class.startswith(("sensor.", "meter.")):
            by_class.setdefault(ld.load_class, []).append(ld)
    living = next(r for r in manifest.rooms if r.room_type == "living")
    bedrooms = [r for r in manifest.rooms if r.room_type == "bedroom"]
    assert {ld.room_id for ld in by_class["sensor.presence"]} >= {living.room_id, *(r.room_id for r in bedrooms)}
    assert [ld.room_id for ld in by_class["sensor.lux"]] == [living.room_id]
    assert len(by_class["meter.energy"]) == 1
    assert {ld.room_id for ld in by_class["sensor.humidity"]} >= {r.room_id for r in bedrooms}
    assert {ld.room_id for ld in by_class["sensor.contact"]} == {living.room_id}
