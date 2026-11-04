from dataclasses import replace
from datetime import date

from gaia.engine.context import TickContext
from gaia.engine.sim import HomeSpec, _Run, simulate_home
from gaia.home.archetypes import ARCHETYPE_SPECS, ARCHETYPES
from gaia.household.personas import Household, make_household
from gaia.policies.lighting import Lighting
from gaia.rng import rng_for
from gaia.schema.events import EventKind
from gaia.world.calendar import local_midnight_utc_ms
from gaia.world.cities import CITIES

START = date(2026, 11, 2)
TZ = CITIES["hyderabad"].tz_offset_min
MIN = 60_000


class FixedRng:
    """Stands in for the policy stream; every draw returns the same value."""

    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


def _setup(archetype, rows, forgetfulness=0.0, daylight_off_p=None, seed=4):
    spec = HomeSpec(home_id="h1", home_seed=seed, city="hyderabad", archetype=archetype,
                    start_date=START, days=1, festivals=frozenset())
    manifest = ARCHETYPES[archetype](spec.home_id, spec.city, rng_for(spec.home_seed, "home"))
    household = make_household(manifest, ARCHETYPE_SPECS[archetype], rng_for(spec.home_seed, "household"))
    member = household.members[0]
    persona = replace(member.persona, forgetfulness=forgetfulness)
    if daylight_off_p is not None:
        persona = replace(persona, daylight_off_p=daylight_off_p)
    member = replace(member, persona=persona)
    index = {r.room_id: i for i, r in enumerate(manifest.rooms)}
    names = ["sleep"]
    for _, act in rows:
        if act not in names:
            names.append(act)
    codes = {name: i for i, name in enumerate(names)}
    rooms = [[-1 if place is None else index[f"h1:{place}"] for place, _ in rows]]
    acts = [[codes[act] for _, act in rows]]
    run = _Run(spec, manifest, Household(members=(member,)), local_midnight_utc_ms(START, TZ),
               rooms, acts, names, policy_order=[Lighting])
    return run, member, manifest


def _rows(*segments):
    out = []
    for place, act, minutes in segments:
        out += [(place, act)] * minutes
    return out


def _load(manifest, room, circuit):
    return next(ld for ld in manifest.loads_in(f"h1:{room}") if ld.circuit == circuit)


def _labels_on(run, load_id):
    cmd_ids = {e.event_id for e in run.events if e.kind == EventKind.command and e.load_id == load_id}
    return [lb for lb in run.labels if lb.event_id in cmd_ids]


# --- G1 behaviour, kept ---------------------------------------------------------

def test_entry_in_the_dark_switches_on_with_darkness_label():
    run, a, manifest = _setup("apartment_2bhk", _rows(("bedroom_1", "tv", 5)))
    run.run(5)
    run.finish()
    (on,) = [lb for lb in run.labels if lb.device_action == "turn_on"]
    assert (on.intent, on.trigger, on.person_id, on.activity) == (
        "illumination", "darkness", a.person_id, "tv")
    assert on.room_id == "h1:bedroom_1" and on.source_ref == "lighting"


def test_sleep_switches_off_with_sleep_label():
    run, a, _ = _setup("apartment_2bhk", _rows(("bedroom_1", "tv", 10), ("bedroom_1", "sleep", 5)))
    run.run(15)
    run.finish()
    (off,) = [lb for lb in run.labels if lb.trigger == "sleep"]
    assert (off.intent, off.device_action, off.person_id, off.activity) == (
        "illumination", "turn_off", a.person_id, "sleep")


def test_leaving_the_room_switches_off_with_room_exit_label():
    run, a, _ = _setup("apartment_2bhk", _rows(("bedroom_1", "tv", 40), ("living", "living", 20)))
    run.run(60)
    run.finish()
    (off,) = [lb for lb in run.labels if lb.trigger == "room_exit"]
    assert (off.intent, off.device_action, off.person_id, off.activity) == (
        "energy_saving", "turn_off", a.person_id, "living")
    assert not run.gt_waste


def test_forgetful_exit_leaves_light_on_and_records_waste():
    run, a, manifest = _setup("apartment_2bhk", _rows(("bedroom_1", "tv", 40), ("living", "living", 20)),
                       forgetfulness=1.0)
    run.run(60)
    run.finish()
    assert not [lb for lb in run.labels if lb.trigger == "room_exit"]
    (w,) = run.gt_waste
    assert w.load_id == manifest.main_light("h1:bedroom_1").load_id
    assert (w.person_id, w.start_ts_utc, w.end_ts_utc) == (
        a.person_id, local_midnight_utc_ms(START, TZ) + 40 * MIN, None)


def test_last_to_sleep_cleanup_switches_off_an_unoccupied_lit_room():
    # A is awake in the living room with the light on, forgets to switch it off, and goes to
    # bed asleep at minute 30. Only the cleanup draw decides the living room light.
    rows = _rows(("living", "tv", 30), ("bedroom_1", "sleep", 10))
    run, a, manifest = _setup("apartment_2bhk", rows, forgetfulness=1.0)
    run.run(30)  # minutes 0..29 played for real; the light in living is on at the end
    ctx = TickContext(minute=30, tick_ms=run.start_ms + 30 * MIN, _env=run.home_env)
    living = manifest.main_light("h1:living").load_id
    assert run.devices.is_on(living)

    (off,) = Lighting().tick(ctx, FixedRng(0.0))  # draw < 0.5: cleanup fires
    assert (off.load_id, off.value, off.intent, off.trigger, off.device_action) == (
        living, 0.0, "cleanup", "last_to_sleep", "turn_off")
    assert off.person_id == a.person_id and off.activity == "sleep"
    assert Lighting().tick(ctx, FixedRng(0.99)) == []  # draw >= 0.5: light stays on


def test_entry_into_a_dark_room_is_labelled_room_entry():
    run, a, manifest = _setup("apartment_2bhk", _rows(("living", "tv", 10), ("bedroom_1", "tv", 10)))
    run.run(20)
    run.finish()
    bed_on = [lb for lb in run.labels
              if lb.room_id == "h1:bedroom_1" and lb.device_action == "turn_on"]
    assert len(bed_on) == 1
    assert (bed_on[0].trigger, bed_on[0].intent, bed_on[0].person_id) == (
        "room_entry", "illumination", a.person_id)


# --- layers, colour and night lights ---------------------------------------------

def _premium_living_with_cove():
    for seed in range(200):
        run, a, manifest = _setup("independent_house", [], seed=seed)
        if any(ld.circuit == "cove" for ld in manifest.loads_in("h1:living")):
            return seed
    raise AssertionError("no seed with a living cove in 200 tries")


def test_tv_start_dims_main_warms_cove_and_labels_ambience():
    seed = _premium_living_with_cove()
    rows = _rows(("living", "living", 3), ("living", "tv", 5))
    run, a, manifest = _setup("independent_house", rows, seed=seed)
    run.run(8)
    run.finish()
    cove = _load(manifest, "living", "cove")
    main = manifest.main_light("h1:living")
    tv_labels = [lb for lb in run.labels if lb.activity == "tv" and lb.trigger == "activity_start"]
    assert tv_labels and all(lb.intent == "ambience" for lb in tv_labels)

    values = {(e.load_id, e.capability): e.value for e in run.events
              if e.kind == EventKind.command}
    assert values[(cove.load_id, "level")] == 0.3
    assert values[(cove.load_id, "hue")] == 0.08
    assert values[(cove.load_id, "saturation")] == 0.6
    assert values[(main.load_id, "level")] == 0.3


def test_night_light_timer():
    # Bedroom to living at minute 5 is a pass through the passage, which lights its night
    # circuit (sun below the horizon). Nobody passes again, so it goes off at minute 7.
    rows = _rows(("bedroom_1", "living", 5), ("living", "living", 5))
    run, a, manifest = _setup("independent_house", rows)
    run.run(10)
    run.finish()
    passage = next(r.room_id for r in manifest.rooms if r.room_type == "passage")
    night = [ld for ld in manifest.loads_in(passage) if ld.circuit == "night"]
    assert night
    start = run.start_ms
    for ld in night:
        cmds = sorted((e for e in run.events if e.kind == EventKind.command and e.load_id == ld.load_id),
                      key=lambda e: e.ts_utc)
        on, off = cmds
        assert (on.value, off.value) == (1.0, 0.0)
        assert start + 5 * MIN <= on.ts_utc < start + 6 * MIN
        assert start + 7 * MIN <= off.ts_utc < start + 8 * MIN
    lbs = [lb for lb in run.labels if lb.room_id == passage]
    assert {(lb.device_action, lb.trigger, lb.intent) for lb in lbs} == {
        ("turn_on", "room_entry", "illumination"), ("turn_off", "timer", "energy_saving")}
    assert all(lb.person_id is None for lb in lbs if lb.trigger == "timer")


def test_daylight_switch_off_after_fifteen_bright_minutes():
    # Balcony (window factor 0.3) lit at dawn, occupied from 05:50. The main light goes off
    # once indoor lux has stayed above 3x the threshold for 15 minutes; p is forced to 1.
    rows = _rows(("balcony", "living", 100))
    pad = [(None, "living")] * 350
    run, a, manifest = _setup("independent_house", pad + rows, daylight_off_p=1.0)
    run.run(len(pad) + len(rows))
    run.finish()
    main = manifest.main_light("h1:balcony")
    (off,) = [lb for lb in run.labels if lb.trigger == "daylight"]
    assert (off.intent, off.device_action, off.room_id, off.person_id) == (
        "energy_saving", "turn_off", "h1:balcony", a.person_id)
    off_ev = [e for e in run.events if e.kind == EventKind.command and e.load_id == main.load_id
              and e.value == 0.0 and e.capability == "power"]
    assert len(off_ev) == 1 and run.start_ms + 350 * MIN < off_ev[0].ts_utc


# --- whole-home checks over several days -------------------------------------------

def _sim(archetype, days, seed):
    spec = HomeSpec(home_id="h1", home_seed=seed, city="hyderabad", archetype=archetype,
                    start_date=START, days=days, festivals=frozenset())
    return simulate_home(spec)


def test_tv_uses_cove_not_full_main():
    # Literal share of tv-start intents on the cove is capped near 50%: each tv start that
    # turns the cove on also dims the main to 0.3, one label each, and the cove colour and level
    # persist between starts. So the check is that no tv-start intent in a cove room drives the
    # main above 0.3, and that the cove is switched on.
    cove_on = main_levels = 0
    for seed in range(4):
        r = _sim("independent_house", 30, seed)
        cove_rooms = {ld.room_id for ld in r.manifest.loads if ld.circuit == "cove"}
        circuit_of = {ld.load_id: ld.circuit for ld in r.manifest.loads}
        room_of = {ld.load_id: ld.room_id for ld in r.manifest.loads}
        cmd = {e.event_id: e for e in r.events if e.kind == EventKind.command}
        for lb in r.labels:
            if lb.activity != "tv" or lb.trigger != "activity_start":
                continue
            e = cmd[lb.event_id]
            if room_of[e.load_id] not in cove_rooms:
                continue
            if circuit_of[e.load_id] == "cove" and e.capability == "power" and e.value == 1.0:
                cove_on += 1
            if circuit_of[e.load_id] == "main" and e.capability == "level":
                main_levels += 1
                assert e.value <= 0.3, (seed, lb)
    assert cove_on > 0 and main_levels > 0


def test_cct_follows_time():
    cct_times = []
    for seed in range(3):
        r = _sim("villa", 30, seed)
        for e in r.events:
            if e.kind == EventKind.command and e.capability == "cct":
                local_min = (e.ts_utc // MIN + e.tz_offset_min) % 1440
                cct_times.append((local_min, e.value))
    assert cct_times
    evening = [v for m, v in cct_times if m >= 19 * 60]
    assert evening and all(v <= 3500 for v in evening)


def test_no_cct_spam():
    for seed in range(3):
        r = _sim("villa", 30, seed)
        by_load = {}
        for e in sorted(r.events, key=lambda e: (e.ts_utc, e.event_id)):
            if e.kind != EventKind.command or e.capability not in ("cct", "power"):
                continue
            last_cct, power_since = by_load.get(e.load_id, (None, False))
            if e.capability == "power":
                by_load[e.load_id] = (last_cct, True)
                continue
            if last_cct is not None and not power_since:
                assert e.ts_utc - last_cct >= 10 * MIN, (e.load_id, last_cct, e.ts_utc)
            by_load[e.load_id] = (e.ts_utc, False)


def test_daylight_off_happens():
    triggers = 0
    for seed in range(4):
        r = _sim("independent_house", 60, seed)
        triggers += sum(1 for lb in r.labels if lb.trigger == "daylight")
    assert triggers > 0
