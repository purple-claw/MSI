from dataclasses import replace
from datetime import date

import pytest

from gaia.engine.sim import HomeSpec, _Run, simulate_home
from gaia.home.archetypes import ARCHETYPE_SPECS, ARCHETYPES
from gaia.household.personas import Household, Persona, make_household
from gaia.policies import POLICY_ORDER
from gaia.policies import media as media_mod
from gaia.policies.media import Media
from gaia.registry.registry import get_load_class
from gaia.rng import rng_for
from gaia.schema.events import EventKind
from gaia.world.calendar import local_midnight_utc_ms
from gaia.world.cities import CITIES

MIN = 60_000
TZ = CITIES["hyderabad"].tz_offset_min
SATURDAY = date(2026, 11, 7)
MONDAY = date(2026, 11, 2)


def _members(archetype, seed):
    manifest = ARCHETYPES[archetype]("h1", "hyderabad", rng_for(seed, "home"))
    household = make_household(manifest, ARCHETYPE_SPECS[archetype], rng_for(seed, "household"))
    return manifest, household


def _has_in_room(manifest, room_id, load_class):
    return any(ld.room_id == room_id and ld.load_class == load_class for ld in manifest.loads)


def _seed_with(archetype, predicate, limit=200):
    for seed in range(limit):
        manifest, _ = _members(archetype, seed)
        if predicate(manifest):
            return seed
    raise AssertionError(f"no seed in {limit} for {archetype}")


def _rows(*segments):
    out = []
    for place, act, minutes in segments:
        out += [(place, act)] * minutes
    return out


def _run(archetype, seed, rows, *, start=MONDAY, days=1, festivals=frozenset(), policy_order=None,
         **persona_changes):
    """Run the policies over one person's rows (place, activity), padded to whole days."""
    spec = HomeSpec(home_id="h1", home_seed=seed, city="hyderabad", archetype=archetype,
                    start_date=start, days=days, festivals=festivals)
    manifest, household = _members(archetype, seed)
    member = household.members[0]
    persona = replace(member.persona, **persona_changes) if persona_changes else member.persona
    member = replace(member, persona=persona)
    household = Household(members=(member,))
    total = days * 1440
    rows = list(rows)
    rows += [rows[-1]] * (total - len(rows))
    index = {r.room_id: i for i, r in enumerate(manifest.rooms)}
    names = ["sleep"]
    for _, act in rows:
        if act not in names:
            names.append(act)
    codes = {name: i for i, name in enumerate(names)}
    rooms = [[-1 if place is None else index[f"h1:{place}"] for place, _ in rows]]
    acts = [[codes[act] for _, act in rows]]
    run = _Run(spec, manifest, household, local_midnight_utc_ms(start, TZ), rooms, acts, names,
               policy_order=policy_order or [Media])
    run.run(total)
    run.finish()
    return run, manifest, member


def _label_for(run, event):
    return next(lb for lb in run.labels if lb.event_id == event.event_id)


def test_policy_is_registered_after_shading():
    names = [cls.name for cls in POLICY_ORDER]
    assert names[-4:] == ["media", "utility", "sensors", "notice"]
    assert names.index("shading") < names.index("media")


def test_tv_session_shape(monkeypatch):
    monkeypatch.setattr(media_mod, "PAUSE_P", 0.0)  # no pauses: the shape is the test
    seed = _seed_with("villa", lambda m: _has_in_room(m, "h1:living", "media.tv"))
    rows = _rows(("bedroom_1", "sleep", 420), ("living", "living", 30), ("living", "tv", 60),
                 ("living", "living", 10), ("bedroom_1", "sleep", 100))
    run, manifest, _ = _run("villa", seed, rows, tv_source_pref=0.0, volume_pref=0.35)
    tv = next(ld.load_id for ld in manifest.loads if ld.room_id == "h1:living"
              and ld.load_class == "media.tv")
    cmds = [e for e in run.events if e.kind == EventKind.command and e.load_id == tv]
    actions = [_label_for(run, e).device_action for e in cmds]
    assert actions[0] == "turn_on"
    assert actions[-1] == "turn_off"
    assert "select_source" in actions and "set_volume" in actions
    assert actions.index("select_source") < actions.index("set_volume") < len(actions) - 1
    on_ts = cmds[0].ts_utc
    off_ts = cmds[-1].ts_utc
    assert on_ts < off_ts
    assert [_label_for(run, e).device_action for e in cmds if e.capability == "playback"] == ["play", "stop"]


def test_stb_standby_not_waste():
    seed = _seed_with("villa", lambda m: _has_in_room(m, "h1:living", "media.stb_dth")
                      and _has_in_room(m, "h1:living", "media.tv"))
    rows = _rows(("bedroom_1", "sleep", 420), ("living", "living", 30), ("living", "tv", 60),
                 ("living", "living", 10), ("bedroom_1", "sleep", 100))
    run, manifest, _ = _run("villa", seed, rows, tv_source_pref=1.0)
    stb = next(ld.load_id for ld in manifest.loads if ld.load_class == "media.stb_dth")
    states = [e for e in run.events if e.kind == EventKind.state and e.load_id == stb
              and e.capability == "power"]
    assert [s.value for s in states] == [1.0, 0.0]  # on for the session, then standby (power 0)
    assert all(w.load_id != stb for w in run.gt_waste)
    assert get_load_class("media.stb_dth").waste_rule == "none"
    assert get_load_class("media.av_receiver").waste_rule == "none"


def test_music_only_with_speaker(monkeypatch):
    seed = _seed_with("villa", lambda m: any(ld.room_id == "h1:living" and ld.load_class in
                                             ("media.speaker", "media.soundbar") for ld in m.loads))
    manifest, _ = _members("villa", seed)
    bedroom_rows = _rows(("bedroom_1", "sleep", 420), ("bedroom_1", "wind_down", 30), ("bedroom_1", "sleep", 100))
    run, _, _ = _run("villa", seed, bedroom_rows, music_affinity=1.0)
    assert not [e for e in run.events if e.kind == EventKind.command
                and e.load_class.startswith("media.")]
    living_rows = _rows(("bedroom_1", "sleep", 420), ("living", "relax", 30), ("living", "living", 10),
                        ("bedroom_1", "sleep", 100))
    run, manifest, _ = _run("villa", seed, living_rows, music_affinity=1.0)
    audio = next(ld.load_id for ld in manifest.loads if ld.room_id == "h1:living"
                 and ld.load_class in ("media.speaker", "media.soundbar"))
    actions = [_label_for(run, e).device_action for e in run.events
               if e.kind == EventKind.command and e.load_id == audio]
    assert actions[0] == "turn_on"
    assert actions[-1] == "turn_off"
    assert actions[-2] == "stop"
    assert "play" in actions  # a fresh speaker starts stopped, so its first session plays


def test_music_continues_across_music_activities_in_one_room():
    seed = _seed_with("apartment_2bhk", lambda m: _has_in_room(m, "h1:kitchen", "media.speaker"))
    manifest, _ = _members("apartment_2bhk", seed)
    rows = _rows(("bedroom_1", "sleep", 420), ("kitchen", "cook", 30), ("kitchen", "dinner", 30),
                 ("living", "living", 10), ("bedroom_1", "sleep", 100))
    run, manifest, _ = _run("apartment_2bhk", seed, rows, music_affinity=1.0)
    speaker = next(ld.load_id for ld in manifest.loads if ld.room_id == "h1:kitchen"
                   and ld.load_class == "media.speaker")
    actions = [_label_for(run, e).device_action for e in run.events
               if e.kind == EventKind.command and e.load_id == speaker]
    assert actions.count("turn_on") == 1
    assert actions.count("stop") == 1
    assert actions[-1] == "turn_off"


def test_source_enums_valid():
    spec = HomeSpec(home_id="h1", home_seed=3, city="hyderabad", archetype="villa",
                    start_date=MONDAY, days=2, festivals=frozenset())
    result = simulate_home(spec, policy_order=[Media])
    allowed = {"turn_on", "turn_off", "select_source", "set_volume", "play", "pause", "stop",
               "open", "close"}
    for lb in result.labels:
        if lb.source_ref != "media":
            continue
        assert lb.device_action in allowed
        assert lb.intent == "entertainment"
    for e in result.events:
        if e.kind != EventKind.command or not e.load_class.startswith("media."):
            continue
        cls = get_load_class(e.load_class)
        cap = next((c for c in cls.capabilities if c.name == e.capability), None)
        if cap is not None and cap.values is not None:
            assert e.value in cap.values


def test_pause_only_while_tv_on(monkeypatch):
    monkeypatch.setattr(media_mod, "PAUSE_P", 1.0)
    seed = _seed_with("villa", lambda m: _has_in_room(m, "h1:living", "media.tv"))
    rows = _rows(("bedroom_1", "sleep", 420), ("living", "living", 20), ("living", "tv", 30),
                 ("living", "living", 30), ("bedroom_1", "sleep", 100))
    run, manifest, _ = _run("villa", seed, rows, tv_source_pref=0.0)
    tv = next(ld.load_id for ld in manifest.loads if ld.room_id == "h1:living"
              and ld.load_class == "media.tv")
    cmds = [e for e in run.events if e.kind == EventKind.command and e.load_id == tv]
    on_ts = next(e.ts_utc for e in cmds if _label_for(run, e).device_action == "turn_on")
    off_ts = next(e.ts_utc for e in cmds if _label_for(run, e).device_action == "turn_off")
    pauses = [e for e in cmds if e.capability == "playback"]
    assert pauses, "a pause every attended minute must produce playback events"
    assert all(on_ts <= e.ts_utc <= off_ts for e in pauses)
    first_pause = next(i for i, e in enumerate(pauses) if _label_for(run, e).device_action == "pause")
    assert _label_for(run, pauses[first_pause + 1]).device_action == "play"


def test_last_leaver_forgets_then_timer_switches_tv_off(monkeypatch):
    seed = _seed_with("villa", lambda m: _has_in_room(m, "h1:living", "media.tv"))
    rows = _rows(("bedroom_1", "sleep", 420), ("living", "living", 20), ("living", "tv", 30),
                 ("bedroom_1", "sleep", 300))
    run, manifest, member = _run("villa", seed, rows, forgetfulness=1.0)
    tv = next(ld.load_id for ld in manifest.loads if ld.room_id == "h1:living"
              and ld.load_class == "media.tv")
    waste = [w for w in run.gt_waste if w.load_id == tv]
    assert len(waste) == 1
    assert waste[0].person_id == member.person_id
    assert waste[0].end_ts_utc is not None
    gap_min = (waste[0].end_ts_utc - waste[0].start_ts_utc) / MIN
    assert media_mod.FORGET_OFF_MIN - 1 <= gap_min <= media_mod.FORGET_OFF_MAX + 3  # jitter and latency
    off = [e for e in run.events if e.kind == EventKind.state and e.load_id == tv
           and e.capability == "power" and e.value == 0.0]
    off_cmd = _label_for(run, next(e for e in run.events if e.kind == EventKind.command
                                   and e.event_id == off[-1].ref_id))
    assert off and off_cmd.trigger == "timer" and off_cmd.intent == "energy_saving"


def test_attentive_leaver_turns_tv_off_on_exit():
    seed = _seed_with("villa", lambda m: _has_in_room(m, "h1:living", "media.tv"))
    rows = _rows(("bedroom_1", "sleep", 420), ("living", "living", 20), ("living", "tv", 30),
                 ("bedroom_1", "sleep", 300))
    run, manifest, _ = _run("villa", seed, rows, forgetfulness=0.0)
    tv = next(ld.load_id for ld in manifest.loads if ld.room_id == "h1:living"
              and ld.load_class == "media.tv")
    assert not [w for w in run.gt_waste if w.load_id == tv]
    off = [e for e in run.events if e.kind == EventKind.command and e.load_id == tv
           and _label_for(run, e).device_action == "turn_off"]
    assert _label_for(run, off[-1]).trigger == "room_exit"


def test_theatre_on_weekend_evening_only(monkeypatch):
    monkeypatch.setattr(media_mod, "THEATRE_P", 1.0)
    seed = _seed_with("villa", lambda m: _has_in_room(m, "h1:home_theatre", "media.projector"))
    manifest, _ = _members("villa", seed)
    proj = next(ld.load_id for ld in manifest.loads if ld.load_class == "media.projector")
    awake_day = _rows(("bedroom_1", "sleep", 420), ("living", "living", 1020))  # awake in living all day
    # two days: a session that starts at 22:59 can run past midnight, so the off is on day two
    run, _, _ = _run("villa", seed, awake_day, start=SATURDAY, days=2)
    cmds = [e for e in run.events if e.kind == EventKind.command and e.load_id == proj]
    assert [_label_for(run, e).device_action for e in cmds][:1] == ["turn_on"]
    on_min = (cmds[0].ts_utc - local_midnight_utc_ms(SATURDAY, TZ)) / MIN
    assert 20 * 60 <= on_min < 23 * 60 + 1
    first_off = next(e for e in cmds if _label_for(run, e).device_action == "turn_off")
    off_min = (first_off.ts_utc - cmds[0].ts_utc) / MIN  # Saturday's session; Sunday has one too
    assert 119 <= off_min <= 182  # on and off commands each carry up to a minute of jitter
    assert all(_label_for(run, e).intent == "entertainment" and _label_for(run, e).trigger == "schedule"
               for e in cmds)
    run, _, _ = _run("villa", seed, awake_day, start=MONDAY, days=2)
    assert not [e for e in run.events if e.kind == EventKind.command and e.load_id == proj]


def test_theatre_needs_someone_awake(monkeypatch):
    monkeypatch.setattr(media_mod, "THEATRE_P", 1.0)
    seed = _seed_with("villa", lambda m: _has_in_room(m, "h1:home_theatre", "media.projector"))
    manifest, _ = _members("villa", seed)
    proj = next(ld.load_id for ld in manifest.loads if ld.load_class == "media.projector")
    asleep = _rows(("bedroom_1", "sleep", 1440))
    run, _, _ = _run("villa", seed, asleep, start=SATURDAY, days=1)
    assert not [e for e in run.events if e.kind == EventKind.command and e.load_id == proj]


def test_one_intent_per_load_capability_per_tick(monkeypatch):
    seen_duplicates = []

    class Probe(Media):
        def tick(self, ctx, rng):
            out = super().tick(ctx, rng)
            keys = [(i.load_id, i.capability) for i in out]
            if len(keys) != len(set(keys)):
                seen_duplicates.append((ctx.minute, keys))
            return out

    monkeypatch.setattr(media_mod, "THEATRE_P", 1.0)
    for seed in range(4):
        spec = HomeSpec(home_id="h1", home_seed=seed, city="hyderabad", archetype="villa",
                        start_date=SATURDAY, days=3, festivals=frozenset())
        simulate_home(spec, policy_order=[Probe])
    assert seen_duplicates == []


@pytest.mark.parametrize("field,bad", [("tv_source_pref", 1.5), ("volume_pref", 0.6),
                                       ("music_affinity", -0.1)])
def test_persona_media_prefs_are_validated(field, bad):
    persona = next(iter(_members("villa", 1)[1].members)).persona
    with pytest.raises(ValueError):
        replace(persona, **{field: bad})


def test_persona_media_prefs_have_defaults():
    persona = Persona(role="homemaker", wake_min=(360, 20), sleep_min=(1350, 25), leave_min=None,
                      return_min=None, forgetfulness=0.1, tech_affinity={"wall": 1.0},
                      light_threshold_lux=120)
    assert (persona.tv_source_pref, persona.volume_pref, persona.music_affinity) == (0.6, 0.35, 0.4)


def test_first_session_on_fresh_home_plays(monkeypatch):
    monkeypatch.setattr(media_mod, "PAUSE_P", 0.0)
    seed = _seed_with("villa", lambda m: _has_in_room(m, "h1:living", "media.tv"))
    rows = _rows(("bedroom_1", "sleep", 420), ("living", "living", 20), ("living", "tv", 30),
                 ("living", "living", 10), ("bedroom_1", "sleep", 100))
    run, manifest, _ = _run("villa", seed, rows, tv_source_pref=0.0, volume_pref=0.35)
    tv = next(ld.load_id for ld in manifest.loads if ld.room_id == "h1:living"
              and ld.load_class == "media.tv")
    actions = [_label_for(run, e).device_action for e in run.events
               if e.kind == EventKind.command and e.load_id == tv]
    assert actions[:4] == ["turn_on", "select_source", "set_volume", "play"]


def test_second_session_starts_from_stopped_and_plays_again(monkeypatch):
    monkeypatch.setattr(media_mod, "PAUSE_P", 0.0)
    seed = _seed_with("villa", lambda m: _has_in_room(m, "h1:living", "media.tv"))
    rows = _rows(("bedroom_1", "sleep", 420), ("living", "tv", 30), ("living", "living", 20),
                 ("living", "tv", 30), ("living", "living", 10), ("bedroom_1", "sleep", 100))
    run, manifest, _ = _run("villa", seed, rows, tv_source_pref=0.0, volume_pref=0.35)
    tv = next(ld.load_id for ld in manifest.loads if ld.room_id == "h1:living"
              and ld.load_class == "media.tv")
    actions = [_label_for(run, e).device_action for e in run.events
               if e.kind == EventKind.command and e.load_id == tv]
    assert actions.count("play") == 2
    assert actions.count("stop") == 2
    assert actions.count("turn_on") == 2 and actions.count("turn_off") == 2
    for i, a in enumerate(actions):
        if a == "turn_off":
            assert actions[i - 1] == "stop"  # playback is stopped before power off
