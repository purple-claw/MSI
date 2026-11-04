from dataclasses import replace
from datetime import date

from gaia.engine.context import TickContext
from gaia.engine.sim import HomeSpec, _Run
from gaia.home.archetypes import APARTMENTS, ARCHETYPES
from gaia.household.personas import Household, make_household
from gaia.policies.lighting_basic import LightingBasic
from gaia.rng import rng_for
from gaia.world.calendar import local_midnight_utc_ms

SLEEP = 0
AWAKE = 1  # act code 1 is "tv" (awake, indoors); code 2 is "living"
OUT = 2


class FixedRng:
    """Stands in for the policy stream; every draw returns the same value."""

    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


def _setup(forgetfulness: float, total: int, rows):
    spec = HomeSpec(home_id="h1", home_seed=4, city="hyderabad", archetype="apartment_2bhk",
                    start_date=date(2026, 11, 2), days=1, festivals=frozenset())
    manifest = ARCHETYPES[spec.archetype](spec.home_id, spec.city, rng_for(spec.home_seed, "home"))
    household = make_household(manifest, APARTMENTS[spec.archetype], rng_for(spec.home_seed, "household"))
    a = household.members[0]
    a = replace(a, persona=replace(a.persona, forgetfulness=forgetfulness))
    index = {r.room_id: i for i, r in enumerate(manifest.rooms)}
    bed, living = index["h1:bedroom_1"], index["h1:living"]
    rooms = [[bed if place == "bed" else living for place, _ in rows] ]
    acts = [[code for _, code in rows]]
    run = _Run(spec, manifest, Household(members=(a,)),
               local_midnight_utc_ms(spec.start_date, 330), rooms, acts, ["sleep", "tv", "living"])
    return run, a, manifest


def _rows(*segments):
    out = []
    for place, code, minutes in segments:
        out += [(place, code)] * minutes
    return out


def test_entry_in_the_dark_switches_on_with_darkness_label():
    run, a, manifest = _setup(0.0, 5, _rows(("bed", AWAKE, 5)))
    run.run(5)
    run.finish()
    (on,) = [lb for lb in run.labels if lb.device_action == "turn_on"]
    assert (on.intent, on.trigger, on.person_id, on.activity) == (
        "illumination", "darkness", a.person_id, "tv")
    assert on.room_id == "h1:bedroom_1" and on.source_ref == "lighting_basic"


def test_sleep_switches_off_with_sleep_label():
    run, a, _ = _setup(0.0, 15, _rows(("bed", AWAKE, 10), ("bed", SLEEP, 5)))
    run.run(15)
    run.finish()
    (off,) = [lb for lb in run.labels if lb.trigger == "sleep"]
    assert (off.intent, off.device_action, off.person_id, off.activity) == (
        "illumination", "turn_off", a.person_id, "sleep")


def test_leaving_the_room_switches_off_with_room_exit_label():
    run, a, _ = _setup(0.0, 60, _rows(("bed", AWAKE, 40), ("living", OUT, 20)))
    run.run(60)
    run.finish()
    (off,) = [lb for lb in run.labels if lb.trigger == "room_exit"]
    assert (off.intent, off.device_action, off.person_id, off.activity) == (
        "energy_saving", "turn_off", a.person_id, "living")
    assert not [w for w in run.gt_waste]


def test_forgetful_exit_leaves_light_on_and_records_waste():
    run, a, _ = _setup(1.0, 60, _rows(("bed", AWAKE, 40), ("living", OUT, 20)))
    run.run(60)
    run.finish()
    assert not [lb for lb in run.labels if lb.trigger == "room_exit"]
    (w,) = run.gt_waste
    assert (w.person_id, w.start_ts_utc, w.end_ts_utc) == (
        a.person_id, local_midnight_utc_ms(date(2026, 11, 2), 330) + 40 * 60_000, None)


def test_last_to_sleep_cleanup_switches_off_an_unoccupied_lit_room():
    # A is awake in the living room with the light on, forgets to switch it off, and goes to
    # bed asleep at minute 30. Only the cleanup draw decides the living room light.
    rows = _rows(("living", AWAKE, 30), ("bed", SLEEP, 10))
    run, a, manifest = _setup(1.0, 40, rows)
    run.run(30)  # minutes 0..29 played for real; the light in living is on at the end
    ctx = TickContext(minute=30, tick_ms=run.start_ms + 30 * 60_000, _env=run.home_env)
    living = manifest.main_light("h1:living").load_id
    assert run.devices.is_on(living)

    (off,) = LightingBasic().tick(ctx, FixedRng(0.0))  # draw < 0.5: cleanup fires
    assert (off.load_id, off.value, off.intent, off.trigger, off.device_action) == (
        living, 0.0, "cleanup", "last_to_sleep", "turn_off")
    assert off.person_id == a.person_id and off.activity == "sleep"
    assert LightingBasic().tick(ctx, FixedRng(0.99)) == []  # draw >= 0.5: light stays on


def test_entry_into_a_dark_room_is_labelled_room_entry():
    # A is awake in the living room (light on, darkness), then walks into the dark bedroom.
    run, a, manifest = _setup(0.0, 20, _rows(("living", AWAKE, 10), ("bed", AWAKE, 10)))
    run.run(20)
    run.finish()
    bed_on = [lb for lb in run.labels
              if lb.room_id == "h1:bedroom_1" and lb.device_action == "turn_on"]
    assert len(bed_on) == 1
    assert (bed_on[0].trigger, bed_on[0].intent, bed_on[0].person_id) == (
        "room_entry", "illumination", a.person_id)
