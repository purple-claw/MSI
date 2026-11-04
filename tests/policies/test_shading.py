from dataclasses import replace
from datetime import date
from statistics import median

from gaia.engine.context import TickContext
from gaia.engine.sim import HomeSpec, _Run, simulate_home
from gaia.home.archetypes import ARCHETYPE_SPECS, ARCHETYPES
from gaia.home.manifest import Load
from gaia.household.personas import Household, make_household
from gaia.policies.shading import OPEN, Shading
from gaia.rng import rng_for
from gaia.schema.events import EventKind, Protocol
from gaia.world.calendar import local_midnight_utc_ms
from gaia.world.cities import CITIES
from gaia.world.sun import sun_altitude_grid

MIN = 60_000
START = date(2026, 11, 2)
HYD_TZ = CITIES["hyderabad"].tz_offset_min


class FixedRng:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value

    def integers(self, lo, hi):
        return lo


def _local_minute(ts_utc: int, tz_offset_min: int) -> int:
    return (ts_utc // MIN + tz_offset_min) % 1440


def _shade_ids(manifest, room_id=None):
    loads = manifest.loads if room_id is None else manifest.loads_in(room_id)
    return {ld.load_id for ld in loads if ld.load_class.startswith("shade.")}


def _position_commands(events, shade_ids):
    return [e for e in events if e.kind == EventKind.command and e.load_id in shade_ids
            and e.capability == "position"]


def test_curtains_open_morning_close_evening():
    opens, closes = [], []
    for seed in range(1, 5):
        spec = HomeSpec(home_id="h1", home_seed=seed, city="hyderabad", archetype="apartment_4bhk",
                        start_date=START, days=7, festivals=frozenset())
        result = simulate_home(spec, policy_order=[Shading])
        for e in _position_commands(result.events, _shade_ids(result.manifest)):
            (opens if e.value == 1.0 else closes).append(_local_minute(e.ts_utc, e.tz_offset_min))
    assert opens and closes
    assert 5 * 60 <= median(opens) <= 10 * 60
    assert 17 * 60 <= median(closes) <= 23 * 60


def test_evening_close_follows_sunset_within_30_minutes():
    start_ms = local_midnight_utc_ms(START, HYD_TZ)
    spec = HomeSpec(home_id="h1", home_seed=2, city="hyderabad", archetype="apartment_4bhk",
                    start_date=START, days=3, festivals=frozenset())
    result = simulate_home(spec, policy_order=[Shading])
    sun = sun_altitude_grid(CITIES["hyderabad"], start_ms, 3 * 1440)
    sunsets = [m for m in range(1, len(sun)) if sun[m - 1] >= 0 > sun[m]]
    labels = {lb.event_id: lb for lb in result.labels}
    evening = [e for e in _position_commands(result.events, _shade_ids(result.manifest))
               if e.value == 0.0 and labels[e.event_id].trigger == "schedule"]
    assert evening
    arrivals = {p.start_ts_utc for p in result.gt_presence if p.room_id is not None}
    for e in evening:
        minute = (e.ts_utc - start_ms) // MIN
        last_sunset = max(s for s in sunsets if s <= minute)
        # Within the 30-minute window, or pending until someone came home at this minute.
        assert minute - last_sunset >= 0
        assert minute - last_sunset <= 30 or start_ms + minute * MIN in arrivals
        assert labels[e.event_id].intent == "privacy"


def _home_with_venetian(city, start):
    spec = HomeSpec(home_id="h1", home_seed=3, city=city, archetype="apartment_4bhk",
                    start_date=start, days=1, festivals=frozenset())
    manifest = ARCHETYPES[spec.archetype](spec.home_id, spec.city, rng_for(spec.home_seed, "home"))
    venetian = Load("h1:venetian", "h1:living", "shade.venetian", Protocol.knx, "knx_dali", "none")
    return spec, replace(manifest, loads=(*manifest.loads, venetian))


def _noon_intents(city, start):
    spec, manifest = _home_with_venetian(city, start)
    household = make_household(manifest, ARCHETYPE_SPECS[spec.archetype], rng_for(spec.home_seed, "household"))
    rooms = [[-1] * 721 for _ in household.members]
    acts = [[0] * 721 for _ in household.members]
    run = _Run(spec, manifest, household, local_midnight_utc_ms(start, CITIES[city].tz_offset_min),
               rooms, acts, ["sleep"], policy_order=[Shading])
    ctx = TickContext(minute=720, tick_ms=run.start_ms + 720 * MIN, _env=run.home_env)
    return run.policies[0].tick(ctx, FixedRng(0.0)), "h1:venetian"


def test_midday_tilt_only_when_hot():
    hot, venetian = _noon_intents("delhi", date(2026, 5, 16))
    tilts = [i for i in hot if i.load_id == venetian and i.capability == "tilt"]
    assert [(t.value, t.intent, t.trigger) for t in tilts] == [(0.5, "thermal_comfort", "schedule")]
    cool, venetian = _noon_intents("delhi", date(2027, 1, 15))
    assert not [i for i in cool if i.capability == "tilt"]


def test_wind_down_closes_bedroom_shades_on_one_draw_at_p_09():
    seed = next(s for s in range(60) if any(
        ld.load_class.startswith("shade.") for ld in ARCHETYPES["apartment_4bhk"](
            "h1", "delhi", rng_for(s, "home")).loads_in("h1:bedroom_1")))
    spec = HomeSpec(home_id="h1", home_seed=seed, city="delhi", archetype="apartment_4bhk",
                    start_date=START, days=1, festivals=frozenset())
    manifest = ARCHETYPES[spec.archetype](spec.home_id, spec.city, rng_for(spec.home_seed, "home"))
    household = make_household(manifest, ARCHETYPE_SPECS[spec.archetype], rng_for(spec.home_seed, "household"))
    member = household.members[0]
    room_id = f"h1:{member.bedroom}"
    shades = _shade_ids(manifest, room_id)
    assert shades
    room_index = next(i for i, r in enumerate(manifest.rooms) if r.room_id == room_id)
    # Asleep at minute 0, awake in the bedroom in wind_down from minute 1.
    names = ["sleep", "wind_down"]
    solo = Household(members=(member,))
    run = _Run(spec, manifest, solo, local_midnight_utc_ms(START, CITIES["delhi"].tz_offset_min),
               [[room_index, room_index]], [[0, 1]], names, policy_order=[Shading])

    for lid in shades:  # shades start closed; this test needs them open
        run.devices.get(lid).set("position", OPEN)

    def wind_down_intents(draw):
        policy = Shading()
        policy.tick(TickContext(minute=0, tick_ms=run.start_ms, _env=run.home_env), FixedRng(draw))
        return policy.tick(TickContext(minute=1, tick_ms=run.start_ms + MIN, _env=run.home_env),
                           FixedRng(draw))

    closed = [i for i in wind_down_intents(0.85) if i.capability == "position"]
    assert {i.load_id for i in closed} == shades
    assert all((i.value, i.intent, i.trigger) == (0.0, "privacy", "activity_start") for i in closed)
    assert not [i for i in wind_down_intents(0.95) if i.capability == "position"]


def test_evening_close_waits_for_the_first_return_home():
    start_ms = local_midnight_utc_ms(START, HYD_TZ)
    sun = sun_altitude_grid(CITIES["hyderabad"], start_ms, 1440)
    sunset = next(m for m in range(721, 1440) if sun[m - 1] >= 0 > sun[m])
    back = sunset + 40  # home only after the sunset window has closed
    seed = next(s for s in range(60) if _shade_ids(ARCHETYPES["apartment_4bhk"](
        "h1", "hyderabad", rng_for(s, "home")), "h1:living"))
    spec = HomeSpec(home_id="h1", home_seed=seed, city="hyderabad", archetype="apartment_4bhk",
                    start_date=START, days=1, festivals=frozenset())
    manifest = ARCHETYPES[spec.archetype](spec.home_id, spec.city, rng_for(spec.home_seed, "home"))
    household = make_household(manifest, ARCHETYPE_SPECS[spec.archetype], rng_for(spec.home_seed, "household"))
    living = next(i for i, r in enumerate(manifest.rooms) if r.room_id == "h1:living")
    span = 60
    run = _Run(spec, manifest, Household(members=(household.members[0],)), start_ms,
               [[-1] * back + [living] * span], [[0] * back + [1] * span], ["sleep", "living"],
               policy_order=[Shading])
    shades = _shade_ids(manifest, "h1:living")
    for lid in shades:  # shades start closed; this test needs them open
        run.devices.get(lid).set("position", OPEN)
    run.run(back + span)
    run.finish()
    closes = [e for e in _position_commands(run.events, shades) if e.value == 0.0]
    assert closes
    first = min((e.ts_utc - start_ms) // MIN for e in closes)
    assert back <= first <= back + 5


def test_evening_close_does_not_carry_over_an_empty_night():
    # Away from before sunset until after the next sunrise, home from noon: no evening close in daylight.
    start_ms = local_midnight_utc_ms(START, HYD_TZ)
    total = 2 * 1440
    sun = sun_altitude_grid(CITIES["hyderabad"], start_ms, total)
    rise = next(m for m in range(1440, total) if sun[m - 1] < 0 <= sun[m])
    back = 1440 + 720  # home at noon on the day after the empty night
    seed = next(s for s in range(60) if _shade_ids(ARCHETYPES["apartment_4bhk"](
        "h1", "hyderabad", rng_for(s, "home")), "h1:living"))
    spec = HomeSpec(home_id="h1", home_seed=seed, city="hyderabad", archetype="apartment_4bhk",
                    start_date=START, days=2, festivals=frozenset())
    manifest = ARCHETYPES[spec.archetype](spec.home_id, spec.city, rng_for(spec.home_seed, "home"))
    household = make_household(manifest, ARCHETYPE_SPECS[spec.archetype], rng_for(spec.home_seed, "household"))
    living = next(i for i, r in enumerate(manifest.rooms) if r.room_id == "h1:living")
    span = 60  # stop before the second sunset, which would close the shades legitimately
    run = _Run(spec, manifest, Household(members=(household.members[0],)), start_ms,
               [[-1] * back + [living] * span], [[0] * back + [1] * span], ["sleep", "living"],
               policy_order=[Shading])
    shades = _shade_ids(manifest, "h1:living")
    for lid in shades:
        run.devices.get(lid).set("position", OPEN)
    run.run(back + span)
    run.finish()
    closes = [e for e in _position_commands(run.events, shades) if e.value == 0.0]
    assert not [e for e in closes if (e.ts_utc - start_ms) // MIN >= rise]
