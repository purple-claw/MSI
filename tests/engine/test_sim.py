from dataclasses import replace
from datetime import date

import pytest

from gaia.engine.sim import HomeSpec, _Run, simulate_home
from gaia.engine.actuator import Intent
from gaia.policies import POLICY_ORDER, Lighting
from gaia.home.archetypes import ARCHETYPE_SPECS, ARCHETYPES
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
    assert a.events == b.events and a.labels == b.labels and a.gt_waste == b.gt_waste


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
    household = make_household(manifest, ARCHETYPE_SPECS[spec.archetype], rng_for(spec.home_seed, "household"))
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
            sleep_causes += sum(1 for lb in r.labels if lb.trigger == "sleep")
    assert bedroom_on > 0
    assert sleep_causes > 0


def test_only_the_last_awake_leaver_decides_the_exit():
    # A and B are both awake in bedroom_1 with the light on and both leave for living at minute 50.
    # A never forgets, B always forgets. B is last in household order, so B alone decides: a
    # forgotten exit that A never sees, and no leave_room switch-off attributed to A.
    spec = HomeSpec(home_id="h1", home_seed=3, city="hyderabad", archetype="apartment_2bhk",
                    start_date=date(2026, 11, 2), days=1, festivals=frozenset())
    manifest = ARCHETYPES[spec.archetype](spec.home_id, spec.city, rng_for(spec.home_seed, "home"))
    household = make_household(manifest, ARCHETYPE_SPECS[spec.archetype], rng_for(spec.home_seed, "household"))
    a, b = household.members[0], household.members[1]
    a = replace(a, persona=replace(a.persona, forgetfulness=0.0))
    b = replace(b, persona=replace(b.persona, forgetfulness=1.0))
    assert a.bedroom == b.bedroom == "bedroom_1"
    index = {r.room_id: i for i, r in enumerate(manifest.rooms)}
    bed, living = index["h1:bedroom_1"], index["h1:living"]
    total = 120
    act_names = ["sleep", "tv", "living"]
    rooms = [[bed] * 50 + [living] * (total - 50)] * 2
    acts = [[1] * 50 + [2] * (total - 50)] * 2
    run = _Run(spec, manifest, Household(members=(a, b)),
               local_midnight_utc_ms(spec.start_date, 330), rooms, acts, act_names)
    run.run(total)
    run.finish()

    exit_ts = local_midnight_utc_ms(spec.start_date, 330) + 50 * 60_000
    load = manifest.main_light("h1:bedroom_1").load_id
    assert not any(lb.trigger == "room_exit" and lb.person_id == a.person_id for lb in run.labels)
    assert [(w.person_id, w.start_ts_utc) for w in run.gt_waste if w.load_id == load] == [
        (b.person_id, exit_ts)]


def test_protocol_latency_does_not_shift_interaction_draws(monkeypatch):
    import gaia.engine.actuator as actuator_mod

    spec = HomeSpec(home_id="h1", home_seed=21, city="hyderabad", archetype="apartment_2bhk",
                    start_date=date(2026, 11, 2), days=3, festivals=frozenset())

    def command_tuples(result):
        # In label order (one label per command), so command timestamps, which depend on latency,
        # do not matter.
        by_id = {e.event_id: e for e in result.events if e.kind == EventKind.command}
        return [(by_id[lb.event_id].load_id, by_id[lb.event_id].source, by_id[lb.event_id].value)
                for lb in result.labels]

    real = command_tuples(simulate_home(spec))
    monkeypatch.setattr(actuator_mod, "latency_ms", lambda protocol, rng: 100)
    stubbed = command_tuples(simulate_home(spec))
    assert real and stubbed == real


def test_presence_runs_tile_the_range_per_person():
    days = 3
    spec = HomeSpec(home_id="h1", home_seed=8, city="hyderabad", archetype="apartment_3bhk",
                    start_date=date(2026, 11, 2), days=days, festivals=frozenset())
    r = simulate_home(spec)
    start = local_midnight_utc_ms(spec.start_date, 330)
    end = start + days * 1440 * 60_000
    persons = [m.person_id for m in r.household.members]
    assert sorted({p.person_id for p in r.gt_presence}) == sorted(persons)
    for pid in persons:
        runs = sorted((p for p in r.gt_presence if p.person_id == pid), key=lambda p: p.start_ts_utc)
        assert runs[0].start_ts_utc == start and runs[-1].end_ts_utc == end
        for a, b in zip(runs, runs[1:]):
            assert a.end_ts_utc == b.start_ts_utc, (a, b)  # no gaps, no overlaps
            assert (a.room_id, a.asleep) != (b.room_id, b.asleep), (a, b)  # maximal runs
        assert all(p.end_ts_utc > p.start_ts_utc for p in runs)


class _FanProbe:
    """Emits on/off intents on a fan that no other policy touches."""

    name = "fan_probe"

    def tick(self, ctx, rng):
        if ctx.minute % 37:
            return []
        fan = next(ld.load_id for ld in ctx.manifest.loads if ld.load_class == "fan.ceiling")
        on = not ctx.devices.is_on(fan)
        return [Intent(load_id=fan, capability="power", value=1.0 if on else 0.0,
                       person_id=ctx.people[0].person_id, activity="living",
                       intent="thermal_comfort", trigger="activity_start",
                       device_action="turn_on" if on else "turn_off", policy=self.name)]


def test_policy_stream_isolation():
    spec = HomeSpec(home_id="h1", home_seed=17, city="hyderabad", archetype="apartment_3bhk",
                    start_date=date(2026, 11, 2), days=3, festivals=frozenset())
    base = simulate_home(spec)
    probed = simulate_home(spec, policy_order=[*POLICY_ORDER, _FanProbe])
    lights = {ld.load_id for ld in base.manifest.loads if ld.is_main_light}
    assert any(e.load_id not in lights for e in probed.events)  # the probe really emitted
    assert [e for e in probed.events if e.load_id in lights] == [e for e in base.events if e.load_id in lights]
    assert [lb for lb in probed.labels if lb.source_ref == "lighting"] == base.labels
    assert [w for w in probed.gt_waste if w.load_id in lights] == base.gt_waste


def test_policy_names_must_be_unique():
    spec = HomeSpec(home_id="h1", home_seed=3, city="hyderabad", archetype="apartment_2bhk",
                    start_date=date(2026, 11, 2), days=1, festivals=frozenset())
    with pytest.raises(ValueError, match="unique"):
        simulate_home(spec, policy_order=[Lighting, Lighting])
