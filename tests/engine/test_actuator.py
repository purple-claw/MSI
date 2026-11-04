import pytest

from gaia.devices.state import HomeDevices
from gaia.engine.actuator import Actuator, Intent
from gaia.home.manifest import HomeManifest, Load, Room
from gaia.household.personas import PERSONAS
from gaia.interaction.choose import choose_source
from gaia.rng import rng_for
from gaia.schema.events import EventKind, Protocol, Source

HOME_SEED = 11
TICK = 1_700_000_000_000
PERSONS_P1 = PERSONAS["working_professional"]


def _manifest() -> HomeManifest:
    return HomeManifest(
        home_id="h1", archetype="test", city="pune",
        rooms=(Room("r1", "living", 1.0),),
        adjacency={"r1": ()},
        loads=(
            Load("l_tun", "r1", "light.tunable", Protocol.matter, True),
            Load("l_curtain", "r1", "shade.curtain", Protocol.zigbee, False),
            Load("s_temp", "r1", "sensor.temperature", Protocol.zigbee, False),
        ),
    )


class _Ids:
    def __init__(self) -> None:
        self.n = 0

    def __call__(self) -> str:
        self.n += 1
        return f"h1:{self.n:09d}"


def _streams() -> dict:
    return {
        "engine": rng_for(HOME_SEED, "engine"),
        "interaction": rng_for(HOME_SEED, "interaction"),
        "protocol": rng_for(HOME_SEED, "protocol"),
    }


def _actuator(ids=None, streams=None) -> Actuator:
    manifest = _manifest()
    streams = streams or _streams()
    return Actuator(
        home_id="h1", manifest=manifest, devices=HomeDevices(manifest), tz_offset_min=330,
        rng_engine=streams["engine"], rng_interaction=streams["interaction"],
        rng_protocol=streams["protocol"],
        personas_by_id={"p1": PERSONAS["working_professional"]},
        next_id=ids or _Ids(),
    )


def _intent(load_id="l_tun", capability="level", value=0.5, **overrides) -> Intent:
    fields = dict(
        load_id=load_id, capability=capability, value=value, person_id="p1",
        activity="living", intent="ambience", trigger="activity_start",
        device_action="set_level", policy="dim_for_tv", source=Source.wall,
    )
    fields.update(overrides)
    return Intent(**fields)


def _commands(act: Actuator):
    return [e for e in act.events if e.kind == EventKind.command]


def test_dedupe_first_wins():
    act = _actuator()
    act.apply([_intent(load_id="l_curtain", capability="position", value=0.5, device_action="set_position"),
               _intent(load_id="l_curtain", capability="position", value=0.8, device_action="set_position")],
              TICK)
    cmds = _commands(act)
    assert len(cmds) == 1
    assert cmds[0].value == 0.5


def test_redundant_intent_on_off_device_does_not_power_on():
    act = _actuator()
    # Fresh tunable is off at level 0, so level 0 and power 0 are no-ops.
    act.apply([_intent(value=0.0)], TICK)
    act.apply([_intent(capability="power", value=0.0, device_action="turn_off")], TICK)
    assert act.events == []
    assert act.labels == []


def test_implicit_power_on():
    act = _actuator()
    act.apply([_intent(capability="cct", value=3000.0, device_action="set_cct")], TICK)
    cmds = _commands(act)
    assert [c.capability for c in cmds] == ["power", "cct"]
    assert len(act.labels) == 2
    power_label, cct_label = act.labels
    assert power_label.device_action == "turn_on"
    assert cct_label.device_action == "set_cct"
    assert power_label.intent == cct_label.intent == "ambience"
    assert power_label.person_id == cct_label.person_id == "p1"
    assert power_label.source_ref == cct_label.source_ref == "dim_for_tv"


def test_no_implicit_power_on_for_position_device():
    act = _actuator()
    act.apply([_intent(load_id="l_curtain", capability="position", value=0.5,
                       device_action="set_position")], TICK)
    assert [c.capability for c in _commands(act)] == ["position"]


def test_every_command_has_one_label():
    act = _actuator()
    act.apply([
        _intent(value=0.4),
        _intent(capability="cct", value=3000.0, device_action="set_cct"),
        _intent(load_id="l_curtain", capability="position", value=0.2, device_action="set_position"),
    ], TICK)
    cmd_ids = [e.event_id for e in _commands(act)]
    assert len(cmd_ids) == len(act.labels) == 4
    assert [label.event_id for label in act.labels] == cmd_ids


def test_clamped_flag_propagates():
    act = _actuator()
    act.apply([_intent(load_id="l_curtain", capability="position", value=5.0,
                       device_action="set_position")], TICK)
    (label,) = act.labels
    assert label.clamped is True
    assert _commands(act)[0].value == 1.0


def test_state_follows_command_and_is_serialised_per_load():
    act = _actuator()
    act.apply([_intent(value=0.3)], TICK)
    act.apply([_intent(value=0.6)], TICK)
    cmd, state = act.events[0], act.events[1]
    assert state.kind == EventKind.state
    assert state.ref_id == cmd.event_id
    assert state.ts_utc >= cmd.ts_utc + 30
    second_cmd = _commands(act)[1]
    assert second_cmd.ts_utc > state.ts_utc


def test_source_defaults_from_persona_when_unset():
    act = _actuator()
    act.apply([_intent(source=None)], TICK)
    assert _commands(act)[0].source in {Source.wall, Source.app, Source.voice}
    assert _commands(act)[0].actor == "p1"


def test_no_person_uses_system_source_and_actor():
    act = _actuator()
    act.apply([_intent(person_id=None, source=None, intent="system", activity="none")], TICK)
    cmd = _commands(act)[0]
    assert cmd.source == Source.system
    assert cmd.actor == "system"


def test_sensor_emits_even_when_unchanged():
    act = _actuator()
    act.emit_sensor("s_temp", "temperature", 24.0, TICK)
    act.emit_sensor("s_temp", "temperature", 24.0, TICK + 60_000)
    sensors = [e for e in act.events if e.kind == EventKind.sensor]
    assert len(sensors) == 2
    assert all(e.source == Source.system and e.actor == "system" for e in sensors)
    assert all(e.ref_id is None for e in sensors)
    assert len(act.labels) == 2
    label = act.labels[0]
    assert (label.intent, label.trigger, label.device_action) == ("system", "presence_tick", "none")
    assert label.event_id == sensors[0].event_id


def test_sensor_value_is_clamped():
    act = _actuator()
    act.emit_sensor("s_temp", "temperature", 99.0, TICK)
    (sensor,) = [e for e in act.events if e.kind == EventKind.sensor]
    assert sensor.value == 50.0
    assert act.labels[0].clamped is True


def _states(streams):
    return {name: rng.bit_generator.state for name, rng in streams.items()}


def test_redundant_intent_draws_nothing_from_any_stream():
    streams = _streams()
    act = _actuator(streams=streams)
    before = _states(streams)
    act.apply([_intent(value=0.0), _intent(capability="power", value=0.0, device_action="turn_off")], TICK)
    assert _states(streams) == before


def test_explicit_source_skips_the_interaction_draw():
    streams = _streams()
    act = _actuator(streams=streams)
    before = streams["interaction"].bit_generator.state
    act.apply([_intent(source=Source.app)], TICK)
    assert streams["interaction"].bit_generator.state == before
    assert _commands(act)[0].source == Source.app


def test_implicit_power_on_makes_exactly_one_source_draw():
    streams = _streams()
    act = _actuator(streams=streams)
    reference = rng_for(HOME_SEED, "interaction")
    choose_source(PERSONS_P1, reference)
    act.apply([_intent(capability="cct", value=3000.0, device_action="set_cct", source=None)], TICK)
    assert [c.capability for c in _commands(act)] == ["power", "cct"]
    assert streams["interaction"].bit_generator.state == reference.bit_generator.state


def test_same_seed_gives_identical_events_and_labels():
    def run():
        act = _actuator()
        act.apply([_intent(value=0.4), _intent(source=None)], TICK)
        act.apply([_intent(load_id="l_curtain", capability="position", value=0.2,
                           device_action="set_position")], TICK + 60_000)
        return act.events, act.labels

    assert run() == run()


def test_unknown_person_id_raises_before_any_change():
    act = _actuator()
    with pytest.raises(ValueError, match="person_id"):
        act.apply([_intent(), _intent(load_id="l_curtain", capability="position", value=0.5,
                                       device_action="set_position", person_id="nobody")], TICK)
    assert act.events == [] and act.labels == []


def test_first_intent_wins_even_when_it_is_redundant():
    act = _actuator()
    act.apply([_intent(value=0.0), _intent(value=0.5)], TICK)
    assert _commands(act) == []
