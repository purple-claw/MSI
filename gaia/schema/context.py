"""Parquet schemas for per-home context rows and ground-truth labels (cause, waste)."""

import pyarrow as pa

CONTEXT_ARROW_SCHEMA = pa.schema(
    [
        pa.field("ts_utc", pa.int64(), nullable=False),
        pa.field("home_id", pa.string(), nullable=False),
        pa.field("outdoor_lux", pa.float64(), nullable=False),
        pa.field("sun_altitude_deg", pa.float64(), nullable=False),
    ]
)

GT_CAUSE_ARROW_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("cause", pa.string(), nullable=False),
        pa.field("person_id", pa.string(), nullable=True),
        pa.field("activity", pa.string(), nullable=True),
    ]
)

GT_WASTE_ARROW_SCHEMA = pa.schema(
    [
        pa.field("load_id", pa.string(), nullable=False),
        pa.field("person_id", pa.string(), nullable=False),
        pa.field("start_ts_utc", pa.int64(), nullable=False),
        pa.field("end_ts_utc", pa.int64(), nullable=True),
    ]
)

GT_PRESENCE_ARROW_SCHEMA = pa.schema(
    [
        pa.field("person_id", pa.string(), nullable=False),
        pa.field("room_id", pa.string(), nullable=True),
        pa.field("start_ts_utc", pa.int64(), nullable=False),
        pa.field("end_ts_utc", pa.int64(), nullable=False),
        pa.field("asleep", pa.bool_(), nullable=False),
    ]
)
