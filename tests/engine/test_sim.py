from datetime import date

from gaia.engine.sim import HomeSpec, _Run, simulate_home
from gaia.home.archetypes import APARTMENTS, ARCHETYPES
from gaia.household.personas import Household, make_household
from gaia.rng import rng_for
from gaia.schema.events import EventKind
from gaia.world.calendar import local_midnight_utc_ms


def _spec(seed=1, days=7):
    return HomeSpec(home_id="h1", home_seed=seed, city="hyderabad", archetype="apartment_2bhk",
                    start_date=date(2026, 11, 2), days=days, festivals=frozenset())


def test_produces_events_and_context():
    r = simulate_home(_spec())
    assert len(r.events) > 50 and len(r.context) == 7 * 24 * 4


def test_deterministic():
    a, b = simulate_home(_spec(5)), simulate_home(_spec(5))
    assert a.events == b.events and a.gt_cause == b.gt_cause


def test_lights_on_after_sunset_not_noon():
    r = simulate_home(_spec(days=14))
    on_hours_ist = [((e.ts_utc // 60000 + 330) // 60) % 24 for e in r.events
                    if e.kind == EventKind.command and e.value == 1.0 and e.room_type == "living"]
    assert sum(17 <= h <= 23 for h in on_hours_ist) > sum(11 <= h <= 14 for h in on_hours_ist)


def test_last_awake_leaver_never_leaves_light_silently_on():
    # A (asleep from minute 10) and B (awake, light on, leaves for living at minute 50) share bedroom_1.
    spec = HomeSpec(home_id="h1", home_seed=3, city="hyderabad", archetype="apartment_2bhk",
                    start_date=date(2026, 11, 2), days=1, festivals=frozenset())
    manifest = ARCHETYPES[spec.archetype](spec.home_id, spec.city, rng_for(spec.home_seed, "home"))
    household = make_household(manifest, APARTMENTS[spec.archetype], rng_for(spec.home_seed, "household"))
    a, b = household.members[0], household.members[1]
    assert a.bedroom == b.bedroom == "bedroom_1"
    index = {r.room_id: i for i, r in enumerate(manifest.rooms)}
    bed, living = index["h1:bedroom_1"], index["h1:living"]
    total = 120
    act_names = ["sleep", "tv", "living"]  # code 0 must be sleep
    rooms = [[bed] * total, [bed] * 50 + [living] * (total - 50)]
    acts = [[1] * 10 + [0] * (total - 10), [1] * 50 + [2] * (total - 50)]
    run = _Run(spec, manifest, Household(members=(a, b)),
               local_midnight_utc_ms(spec.start_date, 330), rooms, acts, act_names)
    run.run(total)
    run.finish()

    start = local_midnight_utc_ms(spec.start_date, 330)
    exit_ts = start + 50 * 60_000
    load = manifest.main_light("h1:bedroom_1").load_id
    assert any(e.kind == EventKind.state and e.load_id == load and e.value == 1.0 for e in run.events)
    off_on_exit = any(e.kind == EventKind.command and e.load_id == load and e.value == 0.0
                      and e.ts_utc >= exit_ts for e in run.events)
    waste_on_exit = any(w.load_id == load and w.start_ts_utc == exit_ts for w in run.gt_waste)
    assert off_on_exit or waste_on_exit


def test_bedroom_light_and_sleep_causes_over_a_month():
    # Wind-down and wake-up are awake activities in the bedroom, so the main light comes on
    # in the dark, and sleep onset with the light on is a labelled sleep switch-off.
    bedroom_on = 0
    sleep_causes = 0
    for archetype in ("apartment_2bhk", "apartment_3bhk", "apartment_4bhk"):
        for seed in (11, 12, 13):
            spec = HomeSpec(home_id="h1", home_seed=seed, city="hyderabad", archetype=archetype,
                            start_date=date(2026, 11, 2), days=30, festivals=frozenset())
            r = simulate_home(spec)
            main_ids = {ld.load_id for ld in r.manifest.loads if ld.is_main_light}
            bedroom_on += sum(1 for e in r.events if e.kind == EventKind.command and e.value == 1.0
                              and e.room_type == "bedroom" and e.load_id in main_ids)
            sleep_causes += sum(1 for c in r.gt_cause if c.cause == "sleep")
    assert bedroom_on > 0
    assert sleep_causes > 0
