from datetime import date

from hypothesis import given, settings
from hypothesis import strategies as st

from gaia.engine.sim import HomeSpec, simulate_home
from gaia.schema.events import EventKind

ARCHETYPES = ["apartment_1rk", "apartment_1bhk", "apartment_2bhk", "apartment_3bhk", "apartment_4bhk"]


@st.composite
def _specs(draw):
    return HomeSpec(home_id="h1", home_seed=draw(st.integers(0, 10**6)), city="hyderabad",
                    archetype=draw(st.sampled_from(ARCHETYPES)), start_date=date(2026, 11, 2),
                    days=3, festivals=frozenset())


@settings(max_examples=25, deadline=None)
@given(_specs())
def test_ts_monotonic(spec):
    ts = [e.ts_utc for e in simulate_home(spec).events]
    assert all(a <= b for a, b in zip(ts, ts[1:]))


@settings(max_examples=25, deadline=None)
@given(_specs())
def test_every_state_has_command(spec):
    events = simulate_home(spec).events
    ts_of = {e.event_id: e.ts_utc for e in events if e.kind == EventKind.command}
    for e in events:
        if e.kind == EventKind.state:
            assert e.ref_id in ts_of and ts_of[e.ref_id] < e.ts_utc


@settings(max_examples=25, deadline=None)
@given(_specs())
def test_no_redundant_state(spec):
    last: dict[str, float] = {}
    for e in simulate_home(spec).events:
        if e.kind != EventKind.state:
            continue
        if e.load_id in last:
            assert last[e.load_id] != e.value
        last[e.load_id] = e.value


@settings(max_examples=25, deadline=None)
@given(_specs())
def test_gt_cause_covers_commands(spec):
    r = simulate_home(spec)
    commands = [e.event_id for e in r.events if e.kind == EventKind.command]
    causes = [c.event_id for c in r.gt_cause]
    assert sorted(commands) == sorted(causes) and len(set(causes)) == len(causes)


@settings(max_examples=25, deadline=None)
@given(_specs())
def test_waste_intervals_valid(spec):
    for w in simulate_home(spec).gt_waste:
        assert w.end_ts_utc is None or w.end_ts_utc > w.start_ts_utc
