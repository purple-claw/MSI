"""Local calendar in a fixed UTC offset (IST by default) and day typing.

All arithmetic is integer milliseconds, so day boundaries are exact.
"""

from datetime import date, datetime, timedelta, timezone
from enum import Enum

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MS = timedelta(milliseconds=1)


class DayType(str, Enum):
    workday = "workday"
    weekend = "weekend"
    festival = "festival"


def local_date(ts_utc_ms: int, tz_offset_min: int) -> date:
    """Return the local calendar date of a UTC timestamp."""
    local = _EPOCH + timedelta(milliseconds=ts_utc_ms, minutes=tz_offset_min)
    return local.date()


def local_midnight_utc_ms(d: date, tz_offset_min: int) -> int:
    """Return the UTC ms at which local day ``d`` begins."""
    local_midnight = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return (local_midnight - _EPOCH - timedelta(minutes=tz_offset_min)) // _MS


def day_type(d: date, festivals: frozenset[date]) -> DayType:
    """Festival wins; otherwise Saturday and Sunday are weekend, the rest workday."""
    if d in festivals:
        return DayType.festival
    if d.weekday() >= 5:
        return DayType.weekend
    return DayType.workday
