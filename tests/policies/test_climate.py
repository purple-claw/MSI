from dataclasses import replace
from datetime import date

from gaia.engine.context import TickContext
from gaia.engine.sim import HomeSpec, _Run, simulate_home
from gaia.home.archetypes import ARCHETYPE_SPECS, ARCHETYPES
from gaia.household.personas import Household, make_household
from gaia.policies.climate import Climate, fan_speed
from gaia.rng import rng_for
from gaia.schema.events import EventKind
from gaia.world.calendar import local_midnight_utc_ms
from gaia.world.cities import CITIES

MIN = 60_000
DELHI_MAY = date(2026, 5, 16)  # heat index 1.0, so an AC decision with ac_affinity 1.0 is certain
DELHI_JAN = date(2027, 1, 15)  # heat index 0.1
TZ = CITIES["delhi"].tz_offset_min


class FixedRng:
    """Stands in for the policy stream; every draw returns the same value."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        return self.value

    def integers(self, lo, hi):
        self.calls += 1
        return lo


def _setup(archetype, rows, city="delhi", start=DELHI_MAY, seed=4, forgetfulness=0.0,
           ac_affinity=None, fan_affinity=None, setpoint=None):
    spec = HomeSpec(home_id="h1", home_seed=seed, city=city, archetype=archetype, start_date=start,
                    days=1, festivals=frozenset())
    manifest = ARCHETYPES[archetype](spec.home_id, spec.city, rng_for(spec.home_seed, "home"))
    household = make_household(manifest, ARCHETYPE_SPECS[archetype], rng_for(spec.home_seed, "household"))
    member = household.members[0]
    persona = replace(member.persona, forgetfulness=forgetfulness)
    if ac_affinity is not None:
        persona = replace(persona, ac_affinity=ac_affinity)
    if fan_affinity is not None:
        persona = replace(persona, fan_affinity=fan_affinity)
    if setpoint is not None:
        persona = replace(persona, ac_setpoint=setpoint)
    member = replace(member, persona=persona)
    index = {r.room_id: i for i, r in enumerate(manifest.rooms)}
    names = ["sleep"]
    for _, act in rows:
        if act not in names:
            names.append(act)
    codes = {name: i for i, name in enumerate(names)}
    rooms = [[-1 if place is None else index[f"h1:{place}"] for place, _ in rows]]
    acts = [[codes[act] for _, act in rows]]
    run = _Run(spec, manifest, Household(members=(member,)), local_midnight_utc_ms(start, TZ),
               rooms, acts, names, policy_order=[Climate])
    return run, member, manifest


def _rows(*segments):
    out = []
    for place, act, minutes in segments:
        out += [(place, act)] * minutes
    return out


def _start_ms(start=DELHI_MAY):
    return local_midnight_utc_ms(start, TZ)


def _ac_ids(manifest, room_id):
    return [ld.load_id for ld in manifest.loads_in(room_id) if ld.load_class.startswith("climate.")]


def _find_seed_with_bedroom_ac(seed_limit=80):
    for seed in range(seed_limit):
        _, _, manifest = _setup("apartment_2bhk", _rows(("living", "living", 1)), seed=seed)
        if _ac_ids(manifest, "h1:bedroom_1"):
            return seed
    raise AssertionError("no seed with a bedroom AC")


def _on_minutes(events, load_ids, end_ms) -> int:
    """Minutes the loads' power is on; an interval still open at ``end_ms`` counts to the end."""
    on_since: dict[str, int] = {}
    total = 0
    for e in events:
        if e.kind != EventKind.state or e.load_id not in load_ids or e.capability != "power":
            continue
        if e.value == 1.0:
            on_since.setdefault(e.load_id, e.ts_utc)
        elif e.load_id in on_since:
            total += e.ts_utc - on_since.pop(e.load_id)
    total += sum(end_ms - start for start in on_since.values())
    return total // MIN


def _commands(run, load_id, capability):
    return [e for e in run.events if e.kind == EventKind.command and e.load_id == load_id
            and e.capability == capability]


def test_ac_more_in_may_than_january():
    def ac_on_minutes(start):
        total = 0
        for seed in range(1, 9):
            spec = HomeSpec(home_id="h1", home_seed=seed, city="delhi", archetype="apartment_3bhk",
                            start_date=start, days=7, festivals=frozenset())
            result = simulate_home(spec, policy_order=[Climate])
            ac = {ld.load_id for ld in result.manifest.loads if ld.load_class.startswith("climate.")}
            total += _on_minutes(result.events, ac, _start_ms(start) + 7 * 1440 * MIN)
        return total

    may = ac_on_minutes(DELHI_MAY)
    jan = ac_on_minutes(DELHI_JAN)
    assert may > 0
    assert may > 3 * jan


def test_setpoint_follows_persona_and_is_in_range():
    seed = _find_seed_with_bedroom_ac()
    run, _, manifest = _setup("apartment_2bhk", _rows(("bedroom_1", "sleep", 30)), seed=seed,
                              ac_affinity=1.0, setpoint=26.0)
    run.run(30)
    run.finish()
    values = [e.value for lid in _ac_ids(manifest, "h1:bedroom_1") for e in _commands(run, lid, "setpoint")]
    assert values == [26.0]
    assert all(20.0 <= v <= 27.0 for v in values)


def test_setpoint_default_is_24():
    # The device starts at 24 C, so the default needs no setpoint command, only the AC on.
    seed = _find_seed_with_bedroom_ac()
    run, _, manifest = _setup("apartment_2bhk", _rows(("bedroom_1", "sleep", 30)), seed=seed,
                              ac_affinity=1.0)
    run.run(30)
    run.finish()
    (ac,) = _ac_ids(manifest, "h1:bedroom_1")
    assert run.devices.get(ac).get("setpoint") == 24.0
    assert _commands(run, ac, "power")


def test_fan_speed_formula_tracks_heat():
    assert fan_speed(0.1) == 1.0
    assert fan_speed(0.31) == 2.0
    assert fan_speed(1.0) == 5.0


def test_fan_on_with_presence_at_speed_from_heat():
    run, _, manifest = _setup("apartment_2bhk", _rows(("living", "living", 20)),
                              ac_affinity=0.0, fan_affinity=1.0)
    run.run(20)
    run.finish()
    fans = [ld.load_id for ld in manifest.loads_in("h1:living") if ld.load_class == "fan.ceiling"]
    assert fans
    assert [e.value for lid in fans for e in _commands(run, lid, "fan_speed")] == [5.0]


def test_no_fan_on_cool_day():
    run, _, manifest = _setup("apartment_2bhk", _rows(("living", "living", 20)),
                              start=DELHI_JAN, ac_affinity=0.0, fan_affinity=1.0)
    run.run(20)
    run.finish()
    fans = [ld.load_id for ld in manifest.loads_in("h1:living") if ld.load_class == "fan.ceiling"]
    assert not [e for lid in fans for e in _commands(run, lid, "power")]


def test_sleeping_occupant_is_not_waste_for_ac():
    seed = _find_seed_with_bedroom_ac()
    run, _, manifest = _setup("apartment_2bhk", _rows(("bedroom_1", "sleep", 90)), seed=seed,
                              ac_affinity=1.0)
    run.run(90)
    run.finish()
    ac = set(_ac_ids(manifest, "h1:bedroom_1"))
    assert _on_minutes(run.events, ac, _start_ms() + 90 * MIN) > 0
    assert not [w for w in run.gt_waste if w.load_id in ac]


def test_ac_unattended_waste_recorded_after_forgetful_exit():
    seed = _find_seed_with_bedroom_ac()
    # Awake in the bedroom for 40 minutes: the AC decision lands at minute 30. Then they leave.
    rows = _rows(("bedroom_1", "relax", 40), ("living", "living", 20))
    run, a, manifest = _setup("apartment_2bhk", rows, seed=seed, forgetfulness=1.0,
                              ac_affinity=1.0, fan_affinity=0.0)
    run.run(60)
    run.finish()
    ac = set(_ac_ids(manifest, "h1:bedroom_1"))
    (w,) = [w for w in run.gt_waste if w.load_id in ac]
    assert (w.person_id, w.start_ts_utc, w.end_ts_utc) == (a.person_id, _start_ms() + 40 * MIN, None)


def test_exit_without_forgetting_switches_ac_off_with_room_exit_label():
    seed = _find_seed_with_bedroom_ac()
    rows = _rows(("bedroom_1", "relax", 40), ("living", "living", 20))
    run, a, _ = _setup("apartment_2bhk", rows, seed=seed, forgetfulness=0.0,
                       ac_affinity=1.0, fan_affinity=0.0)
    run.run(60)
    run.finish()
    assert not run.gt_waste
    exits = [lb for lb in run.labels if lb.trigger == "room_exit" and lb.device_action == "turn_off"]
    assert exits
    assert all(lb.person_id == a.person_id and lb.room_id == "h1:bedroom_1" for lb in exits)


def test_ac_decision_is_drawn_once_per_stay():
    # One awake stay of 120 minutes in a bedroom with a fan and an AC: one fan draw at arrival
    # and one AC draw at minute 30. Nothing more, and a losing draw means no AC.
    seed = _find_seed_with_bedroom_ac()
    run, _, manifest = _setup("apartment_2bhk", _rows(("bedroom_1", "relax", 120)), seed=seed,
                              ac_affinity=0.7, fan_affinity=0.8)
    policy = run.policies[0]
    rng = FixedRng(0.99)
    for m in range(120):
        ctx = TickContext(minute=m, tick_ms=run.start_ms + m * MIN, _env=run.home_env)
        policy.tick(ctx, rng)
    assert rng.calls == 2


def test_exhaust_runs_during_cooking_and_stops_5_to_10_minutes_after():
    rows = _rows(("living", "living", 5), ("kitchen", "cook", 30), ("living", "living", 30))
    run, _, manifest = _setup("apartment_2bhk", rows, ac_affinity=0.0, fan_affinity=0.0)
    run.run(65)
    run.finish()
    (exhaust,) = [ld.load_id for ld in manifest.loads_in("h1:kitchen") if ld.load_class == "fan.exhaust"]
    power = _commands(run, exhaust, "power")
    on = [e for e in power if e.value == 1.0]
    off = [e for e in power if e.value == 0.0]
    assert len(on) == 1 and len(off) == 1
    assert (on[0].ts_utc - _start_ms()) // MIN == 5
    assert 40 <= (off[0].ts_utc - _start_ms()) // MIN <= 45
    labels = {lb.event_id: lb for lb in run.labels}
    assert (labels[on[0].event_id].intent, labels[on[0].event_id].trigger) == ("air_quality", "activity_start")
    assert (labels[off[0].event_id].intent, labels[off[0].event_id].trigger) == ("air_quality", "timer")


def test_cook_and_bath_leave_no_exhaust_waste():
    # The exhaust goes off 5-10 minutes after a 30-minute cook or bath, while the occupant is gone.
    # Those minutes are within the 15-minute grace, so the tracker records nothing.
    rows = _rows(("living", "living", 5), ("kitchen", "cook", 30), ("living", "living", 30))
    run, _, _ = _setup("apartment_2bhk", rows, ac_affinity=0.0, fan_affinity=0.0)
    run.run(65)
    run.finish()
    assert not [w for w in run.gt_waste if w.load_class == "fan.exhaust"]
    rows = _rows(("living", "living", 5), ("bathroom_1", "bath", 30), ("living", "living", 30))
    run, _, _ = _setup("apartment_2bhk", rows, ac_affinity=0.0, fan_affinity=0.0)
    run.run(65)
    run.finish()
    assert [e for e in run.events if e.kind == EventKind.command and e.load_class == "fan.exhaust"]
    assert not [w for w in run.gt_waste if w.load_class == "fan.exhaust"]


def test_forgotten_exhaust_is_waste_from_last_attended_plus_15_minutes():
    # forgetfulness 2.0 makes the forget draw certain (p = forgetfulness * 0.5 = 1.0): a forced forget.
    rows = _rows(("living", "living", 5), ("kitchen", "cook", 30), ("living", "living", 30))
    run, a, manifest = _setup("apartment_2bhk", rows, forgetfulness=2.0, ac_affinity=0.0, fan_affinity=0.0)
    run.run(65)
    run.finish()
    exhaust = {ld.load_id for ld in manifest.loads_in("h1:kitchen") if ld.load_class == "fan.exhaust"}
    (w,) = [w for w in run.gt_waste if w.load_id in exhaust]
    # Last attended: the kitchen tick at minute 34. The waste starts 15 minutes later.
    assert (w.person_id, w.start_ts_utc, w.end_ts_utc) == (a.person_id, _start_ms() + 49 * MIN, None)


def test_sleep_ac_needs_heat_at_least_0_3():
    # January in Delhi: heat index 0.1, so no AC at sleep start even with ac_affinity 1.0.
    seed = _find_seed_with_bedroom_ac()
    run, _, manifest = _setup("apartment_2bhk", _rows(("bedroom_1", "sleep", 30)), start=DELHI_JAN,
                              seed=seed, ac_affinity=1.0)
    run.run(30)
    run.finish()
    ac = _ac_ids(manifest, "h1:bedroom_1")
    assert not [e for lid in ac for e in _commands(run, lid, "power")]
