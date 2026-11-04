"""Waste tracking: a load is wasting energy while its ``waste_rule`` says so.

The rule comes from the load-class registry:
- ``unattended``: on while no awake member is in its room. It opens at the first such minute
  and closes at an awake entry or at the off state.
- ``overrun:N``: on for more than N minutes. It opens N minutes after the on state and closes
  at the off state. Occupancy does not matter.
- ``none``: never wasteful.

Every interval has a start, and an end unless the run ends while it is open. Intervals
shorter than MIN_WASTE_MS are dropped. The person is the last one to leave the load's room,
or None when nobody has left it.
"""

import re
from dataclasses import dataclass

from gaia.home.manifest import HomeManifest
from gaia.registry.registry import get_load_class

MS_PER_MIN = 60_000
MIN_WASTE_MS = MS_PER_MIN  # shorter waste intervals are not emitted as ground truth
_RULE_RE = re.compile(r"(unattended|none|overrun:([1-9][0-9]*))")


@dataclass(frozen=True)
class GtWaste:
    load_id: str
    load_class: str
    person_id: str | None
    start_ts_utc: int
    end_ts_utc: int | None


def parse_rule(rule: str) -> tuple[str, int]:
    """Return (kind, minutes): kind is unattended, overrun or none; minutes is 0 unless overrun."""
    match = _RULE_RE.fullmatch(rule)
    if match is None:
        raise ValueError(f"unknown waste_rule: {rule!r}")
    if match.group(2) is not None:
        return "overrun", int(match.group(2))
    return match.group(1), 0


class WasteTracker:
    def __init__(self, manifest: HomeManifest) -> None:
        self._loads = {ld.load_id: ld for ld in manifest.loads}
        self._rules = {ld.load_id: parse_rule(get_load_class(ld.load_class).waste_rule)
                       for ld in manifest.loads}
        self._on: dict[str, int] = {}  # on loads -> ts of their on state (insertion ordered)
        self._open: dict[str, tuple[str | None, int]] = {}  # load -> (person, start)
        self._last_leaver: dict[str, str] = {}  # room id -> person id
        self._intervals: list[GtWaste] = []

    def on_state(self, load_id: str, ts_ms: int, is_on: bool) -> None:
        """Report a load's state as it is emitted. ``ts_ms`` is the state event time."""
        if is_on:
            self._on.setdefault(load_id, ts_ms)
        else:
            self._on.pop(load_id, None)
            self._close(load_id, ts_ms)

    def on_tick(self, ctx) -> None:
        """Advance the rules for one minute. ``ctx`` is a TickContext (or a duck-typed stand-in)."""
        for person, room_id in ctx.left():
            self._last_leaver[room_id] = person.person_id
        for load_id, on_ts in list(self._on.items()):
            load = self._loads[load_id]
            kind, minutes = self._rules[load_id]
            if kind == "unattended":
                if load_id in self._open:
                    if ctx.awake_in(load.room_id):
                        self._close(load_id, ctx.tick_ms)
                elif not ctx.awake_in(load.room_id):
                    self._open_at(load_id, ctx.tick_ms, load.room_id)
            elif kind == "overrun" and load_id not in self._open:
                due = on_ts + minutes * MS_PER_MIN
                if ctx.tick_ms >= due:
                    self._open_at(load_id, due, load.room_id)

    def finish(self) -> list[GtWaste]:
        """Close the run: open intervals keep no end. Short intervals are dropped."""
        for load_id in list(self._open):
            self._close(load_id, None)
        kept = [w for w in self._intervals
                if w.end_ts_utc is None or w.end_ts_utc - w.start_ts_utc >= MIN_WASTE_MS]
        return sorted(kept, key=lambda w: (w.start_ts_utc, w.load_id))

    # --- internals ----------------------------------------------------------

    def _open_at(self, load_id: str, start_ms: int, room_id: str) -> None:
        self._open[load_id] = (self._last_leaver.get(room_id), start_ms)

    def _close(self, load_id: str, end_ms: int | None) -> None:
        if load_id not in self._open:
            return
        person, start = self._open.pop(load_id)
        load = self._loads[load_id]
        self._intervals.append(GtWaste(load_id, load.load_class, person, start, end_ms))
