from gaia.schema.events import Event, EventKind, Source, Protocol, Domain, events_to_table, EVENT_ARROW_SCHEMA
def _ev(**kw):
    base = dict(ts_utc=1_790_000_000_000, tz_offset_min=330, home_id="h1", event_id="e1", kind=EventKind.command,
                room_id="r1", room_type="living", load_id="l1", load_class="light.onoff", capability="power",
                value=1.0, prev_value=0.0, source=Source.wall, actor="p1", protocol=Protocol.knx, ref_id=None,
                domain=Domain.synthetic)
    base.update(kw); return Event(**base)
def test_table_matches_schema():
    t = events_to_table([_ev(), _ev(event_id="e2", kind=EventKind.state, ref_id="e1")])
    assert t.schema.equals(EVENT_ARROW_SCHEMA)
    assert t.column("kind").to_pylist() == ["command", "state"]
    assert t.column("value").to_pylist() == ["1.0", "1.0"]
def test_event_is_frozen():
    import dataclasses, pytest
    with pytest.raises(dataclasses.FrozenInstanceError):
        _ev().value = 0.0
def test_numpy_float_values_normalised():
    import numpy as np
    t = events_to_table([_ev(value=np.float64(1.0), prev_value=np.float64(0.5))])
    assert t.column("value").to_pylist() == ["1.0"]
    assert t.column("prev_value").to_pylist() == ["0.5"]
