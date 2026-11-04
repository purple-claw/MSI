from datetime import date

from gaia.engine.context import TickContext
from gaia.engine.sim import HomeSpec, _Run
from gaia.home.archetypes import APARTMENTS, ARCHETYPES
from gaia.household.personas import Household, make_household
from gaia.rng import rng_for
from gaia.world.calendar import local_midnight_utc_ms

SLEEP, AWAKE, OUT = 0, 1, 2


def _run():
    spec = HomeSpec(home_id="h1", home_seed=4, city="hyderabad", archetype="apartment_2bhk",
                    start_date=date(2026, 11, 2), days=1, festivals=frozenset())
    manifest = ARCHETYPES[spec.archetype](spec.home_id, spec.city, rng_for(spec.home_seed, "home"))
    household = make_household(manifest, APARTMENTS[spec.archetype], rng_for(spec.home_seed, "household"))
    a = household.members[0]
    index = {r.room_id: i for i, r in enumerate(manifest.rooms)}
    bed, living = index["h1:bedroom_1"], index["h1:living"]
    # awake in bed 0-9, living 10-19, asleep in bed 20-29, awake in bed 30-39, away 40-49
    places = [bed] * 10 + [living] * 10 + [bed] * 20 + [-1] * 10
    codes = [AWAKE] * 10 + [AWAKE] * 10 + [SLEEP] * 10 + [AWAKE] * 10 + [OUT] * 10
    start = local_midnight_utc_ms(spec.start_date, 330)
    run = _Run(spec, manifest, Household(members=(a,)), start, [places], [codes],
               ["sleep", "tv", "living"])
    return run, a, manifest


def _ctx(run, m):
    return TickContext(minute=m, tick_ms=run.start_ms + m * 60_000, _env=run.home_env)


def test_helpers_report_transitions_at_the_minute_they_happen():
    run, a, manifest = _run()
    bed, living = "h1:bedroom_1", "h1:living"

    first = _ctx(run, 0)  # minute 0 has no previous minute: nothing changed
    assert first.changed is False
    assert first.entered() == [] and first.left() == []
    assert first.fell_asleep() == [] and first.woke() == []

    at_living = _ctx(run, 10)
    assert at_living.changed
    assert at_living.left() == [(at_living.people[0], bed)]
    assert at_living.entered() == [(at_living.people[0], living)]

    asleep = _ctx(run, 20)
    assert [p.person_id for p in asleep.fell_asleep()] == [a.person_id]
    assert asleep.woke() == []

    woke = _ctx(run, 30)
    assert [p.person_id for p in woke.woke()] == [a.person_id]
    assert woke.fell_asleep() == []

    away = _ctx(run, 40)
    assert away.left() == [(away.people[0], bed)]
    assert away.entered() == []  # away is not a room
    assert away.people[0].room_id is None and away.people[0].prev_room_id == bed


def test_quiet_minutes_report_nothing_and_reuse_the_people_tuple():
    run, _, _ = _run()
    first = _ctx(run, 3)
    second = _ctx(run, 4)
    assert second.changed is False
    assert second.entered() == [] and second.left() == []
    assert second.people is first.people or second.people == first.people


def test_awake_in_and_any_awake_in_respect_sleep_and_room():
    run, a, _ = _run()
    bed, living = "h1:bedroom_1", "h1:living"
    awake_bed = _ctx(run, 5)
    assert [p.person_id for p in awake_bed.awake_in(bed)] == [a.person_id]
    assert awake_bed.any_awake_in(bed) is True
    assert awake_bed.any_awake_in(living) is False

    sleeping = _ctx(run, 25)  # asleep in bed: in the room, but not awake
    assert sleeping.awake_in(bed) == ()
    assert sleeping.any_awake_in(bed) is False

    assert _ctx(run, 15).awake_in(living)[0].person_id == a.person_id
