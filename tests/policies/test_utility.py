from dataclasses import replace
from datetime import date
from statistics import median

from gaia.engine.sim import HomeSpec, _Run, simulate_home
from gaia.home.archetypes import ARCHETYPE_SPECS, ARCHETYPES
from gaia.household.personas import Household, make_household
from gaia.policies import POLICY_ORDER
from gaia.rng import rng_for
from gaia.schema.events import EventKind
from gaia.world.calendar import local_midnight_utc_ms
from gaia.world.cities import CITIES

MIN = 60_000
TZ = CITIES["hyderabad"].tz_offset_min
MONDAY = date(2026, 11, 2)
START = local_midnight_utc_ms(MONDAY, TZ)
ARCH = "apartment_3bhk"
SEED = 3  # p0 working professional; p1-p3 homemakers; geyser in bathroom_1, chimney, RO and washer in kitchen
ARCHETYPE_NAMES = ["apartment_1rk", "apartment_1bhk", "apartment_2bhk", "apartment_3bhk", "apartment_4bhk",
                   "independent_house", "villa"]
NO_NOTICE = [cls for cls in POLICY_ORDER if cls.name != "notice"]


def _household(archetype, seed):
    manifest = ARCHETYPES[archetype]("h1", "hyderabad", rng_for(seed, "home"))
    return manifest, make_household(manifest, ARCHETYPE_SPECS[archetype], rng_for(seed, "household"))


def _run(rows_by_member, *, changes=None, policy_order=None):
    """One day, explicit rows: rows_by_member[i] is member i's (place, activity) for each minute."""
    manifest, household = _household(ARCH, SEED)
    changes = changes or {}
    members = []
    for i in range(len(rows_by_member)):
        member = household.members[i]
        if i in changes:
            member = replace(member, persona=replace(member.persona, **changes[i]))
        members.append(member)
    spec = HomeSpec(home_id="h1", home_seed=SEED, city="hyderabad", archetype=ARCH,
                    start_date=MONDAY, days=1, festivals=frozenset())
    index = {r.room_id: k for k, r in enumerate(manifest.rooms)}
    names = ["sleep"]
    for rows in rows_by_member:
        for _, act in rows:
            if act not in names:
                names.append(act)
    codes = {name: k for k, name in enumerate(names)}
    rooms, acts = [], []
    for rows in rows_by_member:
        rows = list(rows) + [rows[-1]] * (1440 - len(rows))
        rooms.append([-1 if place is None else index[f"h1:{place}"] for place, _ in rows])
        acts.append([codes[act] for _, act in rows])
    run = _Run(spec, manifest, Household(members=tuple(members)), START, rooms, acts, names,
               policy_order=policy_order or POLICY_ORDER)
    run.run(1440)
    run.finish()
    return run, manifest


def _load(manifest, load_class, room_id):
    return next(ld.load_id for ld in manifest.loads if ld.load_class == load_class and ld.room_id == room_id)


def _power(run, load_id):
    """(minute since local midnight of the run start, value) of each power command on the load."""
    return [((e.ts_utc - START) / MIN, e.value) for e in run.events
            if e.kind == EventKind.command and e.load_id == load_id and e.capability == "power"]


def _label(run, event):
    return next(lb for lb in run.labels if lb.event_id == event.event_id)


def _bath_day(bath_minutes=20):
    return ([("bedroom_1", "sleep")] * 360 + [("living", "breakfast")] * 240
            + [("bathroom_1", "bath")] * bath_minutes + [("living", "relax")] * (1440 - 600 - bath_minutes))


def test_geyser_before_bath():
    run, manifest = _run([_bath_day()], changes={0: {"forgetfulness": 0.0, "notice_rate": 0.0}})
    geyser = _load(manifest, "water.geyser", "h1:bathroom_1")
    ons = [m for m, v in _power(run, geyser) if v == 1.0]
    offs = [m for m, v in _power(run, geyser) if v == 0.0]
    assert len(ons) == 1 and 569 <= ons[0] <= 586, ons  # 15-30 minutes before the bath at 600
    assert len(offs) == 1 and 620 <= offs[0] <= 622, offs  # off at the bath's end
    on_label = _label(run, next(e for e in run.events if e.kind == EventKind.command and e.load_id == geyser
                                and e.value == 1.0))
    assert on_label.intent == "hygiene" and on_label.person_id == run.household.members[0].person_id


def test_geyser_overrun_waste():
    run, manifest = _run([_bath_day()], changes={0: {"forgetfulness": 1.0, "notice_rate": 0.0}})
    geyser = _load(manifest, "water.geyser", "h1:bathroom_1")
    on_state = next(e for e in run.events if e.kind == EventKind.state and e.load_id == geyser and e.value == 1.0)
    (waste,) = [w for w in run.gt_waste if w.load_id == geyser]
    assert waste.person_id is None
    assert waste.start_ts_utc == on_state.ts_utc + 60 * MIN


def test_a_geyser_that_is_switched_off_in_time_wastes_nothing():
    run, manifest = _run([_bath_day()], changes={0: {"forgetfulness": 0.0, "notice_rate": 0.0}})
    geyser = _load(manifest, "water.geyser", "h1:bathroom_1")
    assert not [w for w in run.gt_waste if w.load_id == geyser]


def test_chimney_runs_at_speed_two_or_three_while_cooking_and_stops_after():
    cook = ([("bedroom_1", "sleep")] * 360 + [("living", "relax")] * 240
            + [("kitchen", "cook")] * 60 + [("living", "relax")] * 780)
    run, manifest = _run([cook], changes={0: {"notice_rate": 0.0}}, policy_order=NO_NOTICE)
    chimney = _load(manifest, "kitchen.chimney", "h1:kitchen")
    ons = [m for m, v in _power(run, chimney) if v == 1.0]
    offs = [m for m, v in _power(run, chimney) if v == 0.0]
    assert len(ons) == 1 and 599 <= ons[0] <= 601
    assert len(offs) == 1 and 660 <= offs[0] <= 666
    speeds = [e.value for e in run.events if e.kind == EventKind.command and e.load_id == chimney
              and e.capability == "fan_speed"]
    assert speeds and set(speeds) <= {2.0, 3.0}


def test_ro_purifier_runs_ten_to_twenty_minutes_after_a_cook():
    cook = ([("bedroom_1", "sleep")] * 360 + [("living", "relax")] * 240
            + [("kitchen", "cook")] * 60 + [("living", "relax")] * 780)
    run, manifest = _run([cook], changes={0: {"notice_rate": 0.0}}, policy_order=NO_NOTICE)
    ro = _load(manifest, "water.ro_purifier", "h1:kitchen")
    ons = [m for m, v in _power(run, ro) if v == 1.0]
    offs = [m for m, v in _power(run, ro) if v == 0.0]
    assert len(ons) == 1 and 659 <= ons[0] <= 661
    assert len(offs) == 1 and 669 <= offs[0] <= 682
    assert all(_label(run, e).person_id is None for e in run.events
               if e.kind == EventKind.command and e.load_id == ro)


def test_pump_daily():
    manifest, _ = _household("villa", 3)
    pump = _load(manifest, "water.pump", "h1:utility")
    spec = HomeSpec(home_id="h1", home_seed=3, city="hyderabad", archetype="villa",
                    start_date=MONDAY, days=3, festivals=frozenset())
    result = simulate_home(spec, policy_order=NO_NOTICE)
    cmds = [(e.ts_utc - START) / MIN for e in result.events
            if e.kind == EventKind.command and e.load_id == pump and e.value == 1.0]
    for day in range(3):
        starts = [m - day * 1440 for m in cmds if day * 1440 <= m < (day + 1) * 1440]
        assert 1 <= len(starts) <= 2
        assert 360 <= starts[0] < 480
        if len(starts) == 2:
            assert 1080 <= starts[1] < 1200
    on_events = [e for e in result.events if e.kind == EventKind.command and e.load_id == pump
                 and e.value == 1.0]
    labels = {lb.event_id: lb for lb in result.labels}
    for e in on_events:
        lb = labels[e.event_id]
        assert lb.person_id is None and lb.intent == "utility" and lb.trigger == "timer"
        assert lb.source_ref == "utility"


def test_washing_machine_runs_two_to_four_times_a_week_in_the_morning_by_the_homemaker():
    manifest, household = _household(ARCH, SEED)
    washer = _load(manifest, "laundry.washing_machine", "h1:kitchen")
    spec = HomeSpec(home_id="h1", home_seed=SEED, city="hyderabad", archetype=ARCH,
                    start_date=MONDAY, days=7, festivals=frozenset())
    result = simulate_home(spec)
    homemakers = {m.person_id for m in household.members if m.persona.role == "homemaker"}
    labels = {lb.event_id: lb for lb in result.labels}
    starts = [e for e in result.events if e.kind == EventKind.command and e.load_id == washer
              and e.value == 1.0]
    assert 2 <= len(starts) <= 4
    for e in starts:
        minute = (e.ts_utc - START) / MIN % 1440
        assert 540 <= minute < 660
        assert labels[e.event_id].person_id in homemakers
        assert labels[e.event_id].intent == "utility"


def test_bathroom_waste_median_under_2h():
    durations = []
    for seed in range(20):
        archetype = ARCHETYPE_NAMES[seed % len(ARCHETYPE_NAMES)]
        spec = HomeSpec(home_id="h1", home_seed=seed, city="hyderabad", archetype=archetype,
                        start_date=MONDAY, days=60, festivals=frozenset())
        result = simulate_home(spec)
        rooms = {ld.load_id: ld.room_id for ld in result.manifest.loads}
        durations += [(w.end_ts_utc - w.start_ts_utc) / MIN for w in result.gt_waste
                      if w.end_ts_utc is not None and rooms[w.load_id].endswith(("bathroom_1", "bathroom_2",
                                                                                 "bathroom_3", "bathroom_4",
                                                                                 "bathroom_5"))]
    assert durations
    assert median(durations) < 120
