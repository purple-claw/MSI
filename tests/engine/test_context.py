from datetime import date

from gaia.engine.context import TickContext
from gaia.engine.sim import HomeSpec, _build_grids, _Run, simulate_home
from gaia.home.archetypes import APARTMENTS, ARCHETYPES
from gaia.household.personas import Household, make_household
from gaia.rng import rng_for
from gaia.world.calendar import local_midnight_utc_ms

SLEEP, AWAKE, OUT = 0, 1, 2  # activity codes; names: sleep, tv, living
NAMES = ["sleep", "tv", "living"]
BED, LIVING, AWAY = "h1:bedroom_1", "h1:living", None


def _scenario(rows):
    """One member, one minute per row: (room id or None for away, activity code)."""
    spec = HomeSpec(home_id="h1", home_seed=4, city="hyderabad", archetype="apartment_2bhk",
                    start_date=date(2026, 11, 2), days=1, festivals=frozenset())
    manifest = ARCHETYPES[spec.archetype](spec.home_id, spec.city, rng_for(spec.home_seed, "home"))
    household = make_household(manifest, APARTMENTS[spec.archetype], rng_for(spec.home_seed, "household"))
    index = {r.room_id: i for i, r in enumerate(manifest.rooms)}
    places = [index[room] if room else -1 for room, _ in rows]
    codes = [code for _, code in rows]
    start = local_midnight_utc_ms(spec.start_date, 330)
    run = _Run(spec, manifest, Household(members=(household.members[0],)), start,
               [places], [codes], NAMES)
    return run, household.members[0]


def _ctx(run, m):
    return TickContext(minute=m, tick_ms=run.start_ms + m * 60_000, _env=run.home_env)


def _people_in_order(run, minutes):
    """Build the people tuple for each minute in order, as the engine does."""
    return [_ctx(run, m).people for m in range(minutes)]


def test_quiet_minutes_reuse_the_people_tuple_by_identity():
    run, _ = _scenario([(BED, AWAKE)] * 10)
    tuples = _people_in_order(run, 6)
    assert tuples[4] is tuples[3] and tuples[5] is tuples[4]


def test_minute_after_entry_keeps_the_previous_room():
    # Bedroom until minute 9, living from minute 10: the entry minute and the two quiet minutes
    # after it must all say that the previous room was the bedroom, then living.
    run, a = _scenario([(BED, AWAKE)] * 10 + [(LIVING, AWAKE)] * 10)
    tuples = _people_in_order(run, 13)
    assert (tuples[10][0].room_id, tuples[10][0].prev_room_id) == (LIVING, BED)
    for m in (11, 12):
        assert (tuples[m][0].room_id, tuples[m][0].prev_room_id) == (LIVING, LIVING), m
    assert _ctx(run, 11).entered() == [] and _ctx(run, 12).left() == []


def test_minute_after_falling_asleep_keeps_the_previous_sleep_state():
    run, a = _scenario([(BED, AWAKE)] * 10 + [(BED, SLEEP)] * 10)
    tuples = _people_in_order(run, 13)
    assert tuples[10][0].asleep and not tuples[10][0].prev_asleep
    for m in (11, 12):
        assert tuples[m][0].asleep and tuples[m][0].prev_asleep


def test_activity_change_in_the_same_room_is_seen():
    run, a = _scenario([(LIVING, AWAKE)] * 5 + [(LIVING, OUT)] * 5)
    tuples = _people_in_order(run, 8)
    assert tuples[4][0].activity == "tv"
    assert tuples[5][0].activity == "living" and tuples[5][0].prev_room_id == LIVING
    assert tuples[6][0].activity == "living"
    assert tuples[7] is tuples[6]  # the next minute is quiet again, so the tuple is reused


def test_helpers_report_transitions_at_the_minute_they_happen():
    run, a = _scenario([(BED, AWAKE)] * 10 + [(LIVING, AWAKE)] * 10
                       + [(BED, SLEEP)] * 10 + [(BED, AWAKE)] * 10 + [(AWAY, AWAKE)] * 10)
    _people_in_order(run, 50)  # sequential, as the engine does

    assert _ctx(run, 0).changed is False
    assert _ctx(run, 0).entered() == [] and _ctx(run, 0).left() == []

    at_living = _ctx(run, 10)
    assert at_living.changed
    assert at_living.left() == [(at_living.people[0], BED)]
    assert at_living.entered() == [(at_living.people[0], LIVING)]

    asleep = _ctx(run, 20)
    assert [p.person_id for p in asleep.fell_asleep()] == [a.person_id]
    assert asleep.woke() == []

    woke = _ctx(run, 30)
    assert [p.person_id for p in woke.woke()] == [a.person_id]
    assert woke.fell_asleep() == []

    away = _ctx(run, 40)
    assert away.left() == [(away.people[0], BED)]
    assert away.entered() == []  # away is not a room
    assert away.people[0].room_id is None and away.people[0].prev_room_id == BED


def test_awake_in_and_any_awake_in_respect_sleep_and_room():
    run, a = _scenario([(BED, AWAKE)] * 10 + [(LIVING, AWAKE)] * 10 + [(BED, SLEEP)] * 10)
    _people_in_order(run, 30)
    assert [p.person_id for p in _ctx(run, 5).awake_in(BED)] == [a.person_id]
    assert _ctx(run, 5).any_awake_in(BED) is True
    assert _ctx(run, 5).any_awake_in(LIVING) is False
    assert _ctx(run, 25).awake_in(BED) == ()  # asleep in bed: in the room, not awake
    assert _ctx(run, 25).any_awake_in(BED) is False
    assert _ctx(run, 15).awake_in(LIVING)[0].person_id == a.person_id


def test_reuse_does_not_change_events_or_labels():
    for seed in (1, 2, 3):
        for archetype in ("apartment_2bhk", "apartment_3bhk"):
            spec = HomeSpec(home_id="h1", home_seed=seed, city="hyderabad", archetype=archetype,
                            start_date=date(2026, 11, 2), days=7, festivals=frozenset())
            manifest, household, rooms, acts, names = _build_grids(spec)
            start = local_midnight_utc_ms(spec.start_date, 330)
            total = spec.days * 1440
            results = []
            for reuse in (True, False):
                run = _Run(spec, manifest, household, start, rooms, acts, names, reuse_people=reuse)
                run.run(total)
                run.finish()
                results.append((run.events, run.labels, run.gt_waste))
            assert results[0] == results[1], (seed, archetype)
