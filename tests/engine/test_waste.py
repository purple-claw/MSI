from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from gaia.engine.waste import WasteTracker
from gaia.home.manifest import HomeManifest, Load, Room
from gaia.schema.events import Protocol

MIN = 60_000
T0 = 1_700_000_000_000


@dataclass
class _Ctx:
    """Stand-in for TickContext: the tracker only reads tick_ms, awake_in and left."""
    tick_ms: int
    awake_rooms: tuple = ()
    asleep_rooms: tuple = ()
    leavers: tuple = ()  # (person_id, room_id)

    def any_awake_in(self, room_id):
        return room_id in self.awake_rooms

    def any_in(self, room_id):
        return room_id in self.awake_rooms or room_id in self.asleep_rooms

    def left(self):
        return [(SimpleNamespace(person_id=p), r) for p, r in self.leavers]


def _manifest() -> HomeManifest:
    return HomeManifest(
        home_id="h1", archetype="test", city="pune",
        tier="basic", lighting_system="matter_wifi",
        rooms=(Room("r1", "bedroom", 0.02),),
        adjacency={"r1": ()},
        loads=(
            Load("l_main", "r1", "light.onoff", Protocol.knx, "knx_dali", "main", "bulb"),   # unattended
            Load("l_fan", "r1", "fan.exhaust", Protocol.knx, "knx_dali", "none"),   # unattended:15
            Load("l_chim", "r1", "kitchen.chimney", Protocol.knx, "knx_dali", "none"),  # overrun:10
            Load("l_shade", "r1", "shade.curtain", Protocol.knx, "knx_dali", "none"),  # none
        ),
    )


def test_unattended_opens_on_empty_room_and_closes_on_awake_entry():
    tracker = WasteTracker(_manifest())
    tracker.on_state("l_main", T0, True)
    tracker.on_tick(_Ctx(T0, leavers=(("p1", "r1"),)))  # p1 leaves, light stays on
    tracker.on_tick(_Ctx(T0 + 5 * MIN))
    tracker.on_tick(_Ctx(T0 + 6 * MIN, awake_rooms=("r1",)))  # awake entry closes it
    (w,) = tracker.finish()
    assert (w.load_id, w.load_class, w.person_id) == ("l_main", "light.onoff", "p1")
    assert (w.start_ts_utc, w.end_ts_utc) == (T0, T0 + 6 * MIN)


def test_switch_off_closes_at_the_state_timestamp():
    tracker = WasteTracker(_manifest())
    tracker.on_state("l_main", T0, True)
    tracker.on_tick(_Ctx(T0, leavers=(("p1", "r1"),)))
    tracker.on_state("l_main", T0 + 3 * MIN + 500, False)
    (w,) = tracker.finish()
    assert w.end_ts_utc == T0 + 3 * MIN + 500


def test_no_leaver_gives_null_person():
    tracker = WasteTracker(_manifest())
    tracker.on_state("l_main", T0, True)
    tracker.on_tick(_Ctx(T0))
    (w,) = tracker.finish()
    assert w.person_id is None and w.end_ts_utc is None


def test_intervals_under_a_minute_are_dropped():
    tracker = WasteTracker(_manifest())
    tracker.on_state("l_main", T0, True)
    tracker.on_tick(_Ctx(T0, leavers=(("p1", "r1"),)))
    tracker.on_state("l_main", T0 + MIN - 1, False)
    assert tracker.finish() == []


def test_awake_occupied_room_never_opens():
    tracker = WasteTracker(_manifest())
    tracker.on_state("l_main", T0, True)
    tracker.on_tick(_Ctx(T0, awake_rooms=("r1",)))
    assert tracker.finish() == []


def test_overrun_opens_n_minutes_after_on_whatever_the_occupancy():
    tracker = WasteTracker(_manifest())
    tracker.on_state("l_chim", T0, True)
    tracker.on_tick(_Ctx(T0 + 9 * MIN, leavers=(("p1", "r1"),)))
    tracker.on_tick(_Ctx(T0 + 11 * MIN, awake_rooms=("r1",)))  # occupancy is irrelevant to overrun
    tracker.on_state("l_chim", T0 + 15 * MIN, False)
    (w,) = tracker.finish()
    assert (w.load_class, w.person_id, w.start_ts_utc, w.end_ts_utc) == (
        "kitchen.chimney", None, T0 + 10 * MIN, T0 + 15 * MIN)


def test_overrun_on_short_run_is_not_wasteful():
    tracker = WasteTracker(_manifest())
    tracker.on_state("l_chim", T0, True)
    tracker.on_tick(_Ctx(T0 + 5 * MIN))
    tracker.on_state("l_chim", T0 + 10 * MIN, False)
    assert tracker.finish() == []


def test_grace_waste_starts_at_last_attended_plus_n():
    tracker = WasteTracker(_manifest())
    tracker.on_state("l_fan", T0, True)
    tracker.on_tick(_Ctx(T0 + MIN, awake_rooms=("r1",)))  # last attended: minute 1
    tracker.on_tick(_Ctx(T0 + 10 * MIN, leavers=(("p1", "r1"),)))  # gone; 9 minutes unattended, not yet waste
    tracker.on_tick(_Ctx(T0 + 20 * MIN))  # 19 minutes unattended: waste from minute 1 + 15 = 16
    tracker.on_state("l_fan", T0 + 30 * MIN, False)
    (w,) = tracker.finish()
    assert (w.load_class, w.person_id, w.start_ts_utc, w.end_ts_utc) == (
        "fan.exhaust", "p1", T0 + 16 * MIN, T0 + 30 * MIN)


def test_grace_waste_closes_when_someone_comes_back():
    tracker = WasteTracker(_manifest())
    tracker.on_state("l_fan", T0, True)
    tracker.on_tick(_Ctx(T0 + MIN, awake_rooms=("r1",)))
    tracker.on_tick(_Ctx(T0 + 10 * MIN, leavers=(("p1", "r1"),)))
    tracker.on_tick(_Ctx(T0 + 20 * MIN))
    tracker.on_tick(_Ctx(T0 + 25 * MIN, awake_rooms=("r1",)))
    (w,) = tracker.finish()
    assert (w.start_ts_utc, w.end_ts_utc) == (T0 + 16 * MIN, T0 + 25 * MIN)


def test_grace_gap_within_n_minutes_is_not_waste():
    tracker = WasteTracker(_manifest())
    tracker.on_state("l_fan", T0, True)
    tracker.on_tick(_Ctx(T0 + MIN, awake_rooms=("r1",)))
    tracker.on_tick(_Ctx(T0 + 10 * MIN, leavers=(("p1", "r1"),)))
    tracker.on_state("l_fan", T0 + 15 * MIN, False)  # off at minute 15: the gap is 14 minutes
    assert tracker.finish() == []


def test_parse_rule_grace_and_overrun_forms():
    from gaia.engine.waste import parse_rule
    assert parse_rule("unattended") == ("unattended", 0)
    assert parse_rule("unattended:15") == ("unattended", 15)
    assert parse_rule("overrun:10") == ("overrun", 10)
    assert parse_rule("none") == ("none", 0)
    for bad in ("unattended:0", "unattended:", "overrun:0", "sometimes"):
        with pytest.raises(ValueError):
            parse_rule(bad)


def test_none_rule_never_records_waste():
    tracker = WasteTracker(_manifest())
    tracker.on_state("l_shade", T0, True)
    tracker.on_tick(_Ctx(T0 + 60 * MIN))
    tracker.on_state("l_shade", T0 + 61 * MIN, False)
    assert tracker.finish() == []


def test_pending_off_closes_at_the_awake_arrival_tick():
    # The off state lands 40 s into minute 5, after the person has walked in at minute 5.
    w = WasteTracker(_manifest())
    w.on_state("l_main", T0, True)
    w.on_tick(_Ctx(tick_ms=T0))  # empty room: waste opens
    w.on_state("l_main", T0 + 5 * MIN + 40_000, False)
    w.on_tick(_Ctx(tick_ms=T0 + 5 * MIN, awake_rooms=("r1",)))
    (iv,) = w.finish()
    assert (iv.start_ts_utc, iv.end_ts_utc) == (T0, T0 + 5 * MIN)


def test_pending_off_closes_at_its_timestamp_when_nobody_arrives():
    w = WasteTracker(_manifest())
    w.on_state("l_main", T0, True)
    w.on_tick(_Ctx(tick_ms=T0))
    w.on_state("l_main", T0 + 3 * MIN + 20_000, False)
    w.on_tick(_Ctx(tick_ms=T0 + 3 * MIN))  # off not reached yet: still open
    w.on_tick(_Ctx(tick_ms=T0 + 4 * MIN))  # off reached: closes at the off timestamp
    (iv,) = w.finish()
    assert iv.end_ts_utc == T0 + 3 * MIN + 20_000


def test_night_circuit_load_is_never_waste():
    manifest = HomeManifest(
        home_id="h1", archetype="test", city="pune", tier="premium", lighting_system="knx_dali",
        rooms=(Room("r1", "passage", 0.005),), adjacency={"r1": ()},
        loads=(Load("l_night", "r1", "light.onoff", Protocol.knx, "knx_dali", "night", "step"),),
    )
    w = WasteTracker(manifest)
    w.on_state("l_night", T0, True)
    w.on_tick(_Ctx(tick_ms=T0))
    w.on_tick(_Ctx(tick_ms=T0 + 10 * MIN))
    assert w.finish() == []


def _sleeper_manifest() -> HomeManifest:
    return HomeManifest(
        home_id="h1", archetype="test", city="pune",
        tier="basic", lighting_system="matter_wifi",
        rooms=(Room("r1", "bedroom", 0.02),),
        adjacency={"r1": ()},
        loads=(
            Load("l_ac", "r1", "climate.split_ac", Protocol.knx, "knx_dali", "none"),
            Load("l_main", "r1", "light.onoff", Protocol.knx, "knx_dali", "main", "bulb"),
        ),
    )


def test_sleeping_occupant_counts_as_attended_for_climate():
    tracker = WasteTracker(_sleeper_manifest())
    tracker.on_state("l_ac", T0, True)
    tracker.on_tick(_Ctx(T0, asleep_rooms=("r1",)))
    tracker.on_tick(_Ctx(T0 + 10 * MIN, asleep_rooms=("r1",)))
    assert tracker.finish() == []


def test_sleeping_occupant_does_not_count_as_attended_for_lights():
    tracker = WasteTracker(_sleeper_manifest())
    tracker.on_state("l_main", T0, True)
    tracker.on_tick(_Ctx(T0, asleep_rooms=("r1",)))
    tracker.on_tick(_Ctx(T0 + 10 * MIN, asleep_rooms=("r1",)))
    (w,) = tracker.finish()
    assert (w.load_id, w.start_ts_utc) == ("l_main", T0)


def test_climate_waste_still_opens_when_nobody_is_in_the_room():
    tracker = WasteTracker(_sleeper_manifest())
    tracker.on_state("l_ac", T0, True)
    tracker.on_tick(_Ctx(T0, leavers=(("p1", "r1"),)))
    tracker.on_tick(_Ctx(T0 + 2 * MIN))
    (w,) = tracker.finish()
    assert (w.load_id, w.person_id) == ("l_ac", "p1")
    assert w.start_ts_utc == T0


def test_grace_is_measured_from_the_last_arrival_after_a_second_leave():
    tracker = WasteTracker(_manifest())
    tracker.on_state("l_fan", T0, True)
    tracker.on_tick(_Ctx(T0 + MIN, awake_rooms=("r1",)))
    tracker.on_tick(_Ctx(T0 + 10 * MIN, leavers=(("p1", "r1"),)))
    tracker.on_tick(_Ctx(T0 + 12 * MIN, awake_rooms=("r1",)))  # re-entry: last attended moves to 12
    tracker.on_tick(_Ctx(T0 + 20 * MIN, leavers=(("p2", "r1"),)))  # second leave
    tracker.on_tick(_Ctx(T0 + 30 * MIN))  # 18 minutes after the last attendance: waste from 12 + 15 = 27
    tracker.on_state("l_fan", T0 + 40 * MIN, False)
    (w,) = tracker.finish()
    assert (w.person_id, w.start_ts_utc, w.end_ts_utc) == ("p2", T0 + 27 * MIN, T0 + 40 * MIN)
