"""Per-person timelines: stitch planned days into one contiguous per-minute grid.

Minutes count from local midnight of the start date. Each day's plan runs from its
start up to its final sleep. That sleep runs on through the next day's leading sleep
(which is dropped) until the next day's wake. The last day's sleep is clipped at the end.
"""

from collections.abc import Callable

import numpy as np

from gaia.behaviour.day_plan import Activity

MINUTES_PER_DAY = 1440
SLEEP = "sleep"

Segment = tuple[int, int, str, int]  # (start_min, end_min, activity name, room index or -1)


def stitch_days(plans: list[list[Activity]], resolve: Callable[[str | None], int]) -> list[Segment]:
    """Return absolute-minute segments for one person; ends may exceed the range (clip later)."""
    days = len(plans)
    total = days * MINUTES_PER_DAY
    segments: list[Segment] = []
    for d, plan in enumerate(plans):
        base = d * MINUTES_PER_DAY
        first = 0 if d == 0 else 1  # next day's leading sleep is covered by this day's final sleep
        for act in plan[first:-1]:
            segments.append((base + act.start_min, base + act.end_min, act.name, resolve(act.room)))
        final = plan[-1]
        if d == days - 1:
            end = total
        else:
            end = (d + 1) * MINUTES_PER_DAY + plans[d + 1][0].end_min
        segments.append((base + final.start_min, end, final.name, resolve(final.room)))
    return segments


def fill_grid(
    segments: list[Segment], total: int, act_codes: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Expand segments into per-minute arrays: room index (-1 away) and activity code.

    ``act_codes`` is shared across persons; new names get the next free code.
    """
    room = np.full(total, -1, dtype=np.int16)
    act = np.zeros(total, dtype=np.int16)
    for start, end, name, room_idx in segments:
        lo, hi = max(0, start), min(total, end)
        if lo >= hi:
            continue
        room[lo:hi] = room_idx
        act[lo:hi] = act_codes.setdefault(name, len(act_codes))
    return room, act
