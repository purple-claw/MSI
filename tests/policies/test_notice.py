from dataclasses import replace
from datetime import date

from gaia.engine.sim import HomeSpec, _Run
from gaia.home.archetypes import ARCHETYPE_SPECS, ARCHETYPES
from gaia.household.personas import Household, make_household
from gaia.policies import POLICY_ORDER
from gaia.rng import rng_for
from gaia.schema.events import EventKind, Source
from gaia.world.calendar import local_midnight_utc_ms
from gaia.world.cities import CITIES

MIN = 60_000
TZ = CITIES["hyderabad"].tz_offset_min
MONDAY = date(2026, 11, 2)
START = local_midnight_utc_ms(MONDAY, TZ)
ARCH = "apartment_3bhk"
SEED = 3  # household: p0 working professional, p1-p3 homemakers; geysers in bathroom_1 and bathroom_3


def _run(rows_by_member, *, changes=None, policy_order=None):
    """One day: rows_by_member[i] is member i's (place, activity) per minute, in household order."""
    spec = HomeSpec(home_id="h1", home_seed=SEED, city="hyderabad", archetype=ARCH,
                    start_date=MONDAY, days=1, festivals=frozenset())
    manifest = ARCHETYPES[ARCH]("h1", "hyderabad", rng_for(SEED, "home"))
    household = make_household(manifest, ARCHETYPE_SPECS[ARCH], rng_for(SEED, "household"))
    changes = changes or {}
    members = []
    for i in range(len(rows_by_member)):
        member = household.members[i]
        if i in changes:
            member = replace(member, persona=replace(member.persona, **changes[i]))
        members.append(member)
    total = 1440
    index = {r.room_id: k for k, r in enumerate(manifest.rooms)}
    names = ["sleep"]
    for rows in rows_by_member:
        for _, act in rows:
            if act not in names:
                names.append(act)
    codes = {name: k for k, name in enumerate(names)}
    rooms, acts = [], []
    for rows in rows_by_member:
        rows = list(rows) + [rows[-1]] * (total - len(rows))
        rooms.append([-1 if place is None else index[f"h1:{place}"] for place, _ in rows])
        acts.append([codes[act] for _, act in rows])
    run = _Run(spec, manifest, Household(members=tuple(members)), START, rooms, acts, names,
               policy_order=policy_order or POLICY_ORDER)
    run.run(total)
    run.finish()
    return run, manifest


def _minute(ts_ms):
    return (ts_ms - START) / MIN


def _power_commands(run, load_id):
    return [e for e in run.events if e.kind == EventKind.command and e.load_id == load_id
            and e.capability == "power"]


def _label(run, event):
    return next(lb for lb in run.labels if lb.event_id == event.event_id)


def _notice_events(run):
    labels = {lb.event_id: lb for lb in run.labels if lb.intent == "noticed_waste"}
    return [e for e in run.events if e.kind == EventKind.command and e.event_id in labels], labels


def _living_lights(manifest):
    return {ld.load_id for ld in manifest.loads if ld.room_id == "h1:living" and ld.circuit != "none"}


def _evening_rows(a_leaves_at=1200):
    """A (member 0): in living until 20:00, in the kitchen 20:00-21:00, back in living at 21:00."""
    return ([("bedroom_1", "sleep")] * 420
            + [("living", "relax")] * (a_leaves_at - 420)
            + [("kitchen", "dinner")] * 60
            + [("living", "relax")] * 60
            + [("bedroom_1", "sleep")] * (1440 - a_leaves_at - 120))


def test_notice_switches_off_a_wasted_light_outside_the_room_with_wall_source():
    a = _evening_rows()
    b = [("bedroom_2", "relax")] * 1440  # awake, another room, notices every minute
    run, manifest = _run([a, b], changes={0: {"forgetfulness": 1.0}, 1: {"notice_rate": 1.0}})
    notices, labels = _notice_events(run)
    lights = _living_lights(manifest)
    living_offs = [e for e in notices if e.load_id in lights]
    assert living_offs, "a light left on in an empty living room must be noticed"
    assert 1200 <= _minute(living_offs[0].ts_utc) <= 1203  # A leaves the living room at 20:00
    for e in living_offs:
        lb = labels[e.event_id]
        assert e.source == Source.wall
        assert e.value == 0.0 and e.capability == "power"
        assert lb.trigger == "notice" and lb.person_id == run.household.members[1].person_id


def test_a_light_switched_off_by_notice_is_switched_on_again_by_its_policy():
    a = _evening_rows()
    b = [("bedroom_2", "relax")] * 1440
    run, manifest = _run([a, b], changes={0: {"forgetfulness": 1.0}, 1: {"notice_rate": 1.0}})
    main = manifest.main_light("h1:living").load_id
    back_on = [e for e in _power_commands(run, main) if e.value == 1.0 and _minute(e.ts_utc) > 1200]
    assert back_on, "lighting must turn the living light on when A comes back at 21:00"
    assert 1260 <= _minute(back_on[0].ts_utc) <= 1262
    assert _label(run, back_on[0]).source_ref == "lighting"


def test_a_member_in_the_load_room_does_not_notice_and_outside_members_do():
    # A forgetful geyser overruns; its room holds one awake member, a second member is in the living room.
    bath = ([("bedroom_1", "sleep")] * 360 + [("living", "breakfast")] * 240
            + [("bathroom_1", "bath")] * 20 + [("living", "relax")] * 820)
    in_room = [("bathroom_1", "relax")] * 1440
    outside = [("living", "relax")] * 1440
    run, manifest = _run(
        [bath, in_room, outside],
        changes={0: {"forgetfulness": 1.0, "notice_rate": 0.0}, 1: {"notice_rate": 1.0}, 2: {"notice_rate": 1.0}})
    geyser = next(ld.load_id for ld in manifest.loads if ld.load_class == "water.geyser" and ld.room_id == "h1:bathroom_1")
    notices, labels = _notice_events(run)
    geyser_notices = [e for e in notices if e.load_id == geyser]
    assert geyser_notices
    assert {labels[e.event_id].person_id for e in geyser_notices} == {run.household.members[2].person_id}


def test_the_first_member_in_household_order_who_notices_switches_the_load_off():
    a = _evening_rows()
    b = [("bedroom_2", "relax")] * 1440
    c = [("bedroom_3", "relax")] * 1440
    run, manifest = _run([a, b, c], changes={0: {"forgetfulness": 1.0}, 1: {"notice_rate": 1.0},
                                              2: {"notice_rate": 1.0}})
    notices, labels = _notice_events(run)
    lights = _living_lights(manifest)
    assert notices
    for e in notices:  # member 1 is ahead of member 2 in household order, and notices every minute
        assert labels[e.event_id].person_id == run.household.members[1].person_id


def test_media_does_not_power_a_noticed_tv_back_on_before_its_owner_returns():
    # A watches TV, forgets it when leaving at 20:00 (forgetfulness 1). B notices it at once.
    # The forgotten-TV timer (30-180 min) must not switch it on again, and A's return at 21:00 does.
    a = ([("bedroom_1", "sleep")] * 420 + [("living", "relax")] * 720 + [("living", "tv")] * 60
         + [("kitchen", "dinner")] * 60 + [("living", "tv")] * 60 + [("bedroom_1", "sleep")] * 120)
    b = [("bedroom_2", "relax")] * 1440
    run, manifest = _run([a, b], changes={0: {"forgetfulness": 1.0}, 1: {"notice_rate": 1.0}})
    tv = next(ld.load_id for ld in manifest.loads if ld.load_class == "media.tv" and ld.room_id == "h1:living")
    notices, _ = _notice_events(run)
    tv_off = [e for e in notices if e.load_id == tv]
    assert tv_off and 1200 <= _minute(tv_off[0].ts_utc) <= 1203
    between = [e for e in _power_commands(run, tv) if e.value == 1.0 and 1202 < _minute(e.ts_utc) < 1260]
    assert not between
    back = [e for e in _power_commands(run, tv) if e.value == 1.0 and _minute(e.ts_utc) >= 1260]
    assert back and 1260 <= _minute(back[0].ts_utc) <= 1262
