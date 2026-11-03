from datetime import date

from gaia.engine.sim import HomeSpec, simulate_home
from gaia.schema.events import EventKind


def _spec(seed=1, days=7):
    return HomeSpec(home_id="h1", home_seed=seed, city="hyderabad", archetype="apartment_2bhk",
                    start_date=date(2026, 11, 2), days=days, festivals=frozenset())


def test_produces_events_and_context():
    r = simulate_home(_spec())
    assert len(r.events) > 50 and len(r.context) == 7 * 24 * 4


def test_deterministic():
    a, b = simulate_home(_spec(5)), simulate_home(_spec(5))
    assert a.events == b.events and a.gt_cause == b.gt_cause


def test_lights_on_after_sunset_not_noon():
    r = simulate_home(_spec(days=14))
    on_hours_ist = [((e.ts_utc // 60000 + 330) // 60) % 24 for e in r.events
                    if e.kind == EventKind.command and e.value == 1.0 and e.room_type == "living"]
    assert sum(17 <= h <= 23 for h in on_hours_ist) > sum(11 <= h <= 14 for h in on_hours_ist)
