import pytest

from gaia.devices.state import HomeDevices
from gaia.engine.actuator import SENSOR_SOURCE_REF, Actuator, Intent
from gaia.home.manifest import HomeManifest, Load, Room
from gaia.household.personas import PERSONAS
from gaia.interaction.choose import choose_source
from gaia.rng import rng_for
from gaia.schema.events import EventKind, Protocol, Source

HOME_SEED = 11
TICK = 1_700_000_000_000
POLICY = "dim_for_tv"
PERSON = PERSONAS["working_professional"]


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


def _actuator(on_state=None) -> Actuator:
    manifest = _manifest()
    return Actuator(
        home_id="h1", manifest=manifest, devices=HomeDevices(manifest), tz_offset_min=330,
        home_seed=HOME_SEED, personas_by_id={"p1": PERSON}, on_state=on_state,
    )


def _intent(load_id="l_tun", capability="level", value=0.5, **overrides) -> Intent:
    fields = dict(
        load_id=load_id, capability=capability, value=value, person_id="p1",
        activity="living", intent="ambience", trigger="activity_start",
        device_action="set_level", policy=POLICY, source=Source.wall,
    )
    fields.update(overrides)
    return Intent(**fields)


def _commands(act: Actuator):
    return [e for e in act.events if e.kind == EventKind.command]


def _states(act: Actuator, policy: str):
    s = act.streams_for(policy)
    return (s.engine.bit_generator.state, s.interaction.bit_generator.state,
            s.protocol.bit_generator.state)


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
    assert power_label.source_ref == cct_label.source_ref == POLICY


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


def test_redundant_intent_draws_nothing_from_any_stream():
    act = _actuator()
    before = _states(act, POLICY)
    act.apply([_intent(value=0.0), _intent(capability="power", value=0.0, device_action="turn_off")], TICK)
    assert _states(act, POLICY) == before


def test_explicit_source_skips_the_interaction_draw():
    act = _actuator()
    before = act.streams_for(POLICY).interaction.bit_generator.state
    act.apply([_intent(source=Source.app)], TICK)
    assert act.streams_for(POLICY).interaction.bit_generator.state == before
    assert _commands(act)[0].source == Source.app


def test_implicit_power_on_makes_exactly_one_source_draw():
    act = _actuator()
    reference = rng_for(HOME_SEED, f"actuator/{POLICY}/interaction")
    choose_source(PERSON, reference)
    act.apply([_intent(capability="cct", value=3000.0, device_action="set_cct", source=None)], TICK)
    assert [c.capability for c in _commands(act)] == ["power", "cct"]
    assert act.streams_for(POLICY).interaction.bit_generator.state == reference.bit_generator.state


def test_same_seed_gives_identical_events_and_labels():
    def run():
        act = _actuator()
        act.apply([_intent(value=0.4), _intent(source=None)], TICK)
        act.apply([_intent(load_id="l_curtain", capability="position", value=0.2,
                           device_action="set_position")], TICK + 60_000)
        return act.events, act.labels

    assert run() == run()


def test_each_policy_draws_from_its_own_streams():
    act = _actuator()
    other = _states(act, "other_policy")
    act.apply([_intent(source=None)], TICK)
    assert _states(act, "other_policy") == other


def test_event_ids_count_per_policy():
    act = _actuator()
    act.apply([_intent(value=0.4)], TICK)
    act.apply([_intent(load_id="l_curtain", capability="position", value=0.2,
                       device_action="set_position", policy="shade_policy")], TICK)
    ids = [e.event_id for e in act.events]
    assert ids[0] == "h1:dim_for_tv:000000000" and ids[1] == "h1:dim_for_tv:000000001"
    assert "h1:shade_policy:000000000" in ids


def test_on_state_callback_fires_for_each_state_event_only():
    seen = []
    act = _actuator(on_state=lambda load, cap, ts: seen.append((load, cap)))
    act.apply([_intent(capability="cct", value=3000.0, device_action="set_cct")], TICK)
    act.emit_sensor("s_temp", "temperature", 24.0, TICK)
    assert seen == [("l_tun", "power"), ("l_tun", "cct")]


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
