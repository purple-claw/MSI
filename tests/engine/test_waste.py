from dataclasses import dataclass
from types import SimpleNamespace

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
    leavers: tuple = ()  # (person_id, room_id)

    def any_awake_in(self, room_id):
        return room_id in self.awake_rooms

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
            Load("l_fan", "r1", "fan.exhaust", Protocol.knx, "knx_dali", "none"),   # overrun:15
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


def test_overrun_opens_after_n_minutes_on_with_no_person():
    tracker = WasteTracker(_manifest())
    tracker.on_state("l_fan", T0, True)
    tracker.on_tick(_Ctx(T0 + 14 * MIN, leavers=(("p1", "r1"),)))
    tracker.on_tick(_Ctx(T0 + 16 * MIN, awake_rooms=("r1",)))  # occupancy is irrelevant to overrun
    tracker.on_state("l_fan", T0 + 20 * MIN, False)
    (w,) = tracker.finish()
    assert (w.load_class, w.person_id, w.start_ts_utc, w.end_ts_utc) == (
        "fan.exhaust", None, T0 + 15 * MIN, T0 + 20 * MIN)


def test_overrun_on_short_run_is_not_wasteful():
    tracker = WasteTracker(_manifest())
    tracker.on_state("l_fan", T0, True)
    tracker.on_tick(_Ctx(T0 + 5 * MIN))
    tracker.on_state("l_fan", T0 + 10 * MIN, False)
    assert tracker.finish() == []


def test_none_rule_never_records_waste():
    tracker = WasteTracker(_manifest())
    tracker.on_state("l_shade", T0, True)
    tracker.on_tick(_Ctx(T0 + 60 * MIN))
    tracker.on_state("l_shade", T0 + 61 * MIN, False)
    assert tracker.finish() == []
