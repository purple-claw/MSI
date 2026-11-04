"""Daily activity plans: a contiguous cover of one local day for a persona.

Anchor times (wake, leave, return, sleep) are sampled from the persona. Each
template block is then placed in order, so every activity starts where the previous
one ends. Toilet visits are then cut into awake home blocks, after all anchor draws.
Plans are deterministic for a given rng state.
"""

from dataclasses import dataclass

import numpy as np

from gaia.household.personas import Persona
from gaia.world.calendar import DayType

MIN_BLOCK = 5
DAY = 1440
WAKE_UP = 5  # awake in bed after the leading sleep, before the bath or cook
WIND_DOWN = 15  # awake in bed right before the final sleep
TOILET = "toilet"
TOILET_VISITS = (2, 5)  # visits per awake day, inclusive
TOILET_MIN, TOILET_MAX = MIN_BLOCK, 6  # visit length in minutes, inclusive: no shorter than a block
# Blocks a toilet visit never splits: sleeps and the bath, the wake-up and wind-down bookends, fixed
# rest, meals (a meal is one sitting) and cooking.
_NOT_HOST = frozenset({"sleep", "away", "bath", TOILET, "wake_up", "wind_down", "rest",
                       "breakfast", "lunch", "dinner", "cook"})


@dataclass(frozen=True)
class Activity:
    name: str
    room: str | None  # symbolic slot: own_bedroom, own_bathroom, living, kitchen, or None (away)
    start_min: int
    end_min: int


@dataclass(frozen=True)
class _Block:
    name: str
    room: str | None
    clock: int | None = None  # earliest start (fixed clock or sampled anchor)
    dur: int | None = None  # fixed duration; None means "until"
    until: int | None = None  # end target for "until" blocks


def _sample(rng: np.random.Generator, spec: tuple[float, float]) -> int:
    mean, sd = spec
    return int(round(rng.normal(mean, sd)))


def _professional_workday(wake: int, leave: int, ret: int, sleep: int, wake_next: int) -> list[_Block]:
    return [
        _Block("sleep", "own_bedroom", until=wake),
        _Block("wake_up", "own_bedroom", dur=WAKE_UP),
        _Block("bath", "own_bathroom", dur=20),
        _Block("breakfast", "kitchen", dur=20),
        _Block("living", "living", until=leave),
        _Block("away", None, clock=leave, until=ret),
        _Block("relax", "living", until=ret + 60),
        _Block("dinner", "kitchen", clock=ret + 60, dur=30),
        _Block("tv", "living", until=sleep - WIND_DOWN),
        _Block("wind_down", "own_bedroom", clock=sleep - WIND_DOWN, dur=WIND_DOWN),
        _Block("sleep", "own_bedroom", clock=sleep, until=DAY + wake_next),
    ]


def _professional_leisure(wake: int, sleep: int, wake_next: int) -> list[_Block]:
    return [
        _Block("sleep", "own_bedroom", until=wake),
        _Block("wake_up", "own_bedroom", dur=WAKE_UP),
        _Block("bath", "own_bathroom", dur=20),
        _Block("breakfast", "kitchen", dur=20),
        _Block("relax", "living", until=780),
        _Block("lunch", "kitchen", clock=780, dur=30),
        _Block("tv", "living", until=sleep - WIND_DOWN),
        _Block("wind_down", "own_bedroom", clock=sleep - WIND_DOWN, dur=WIND_DOWN),
        _Block("sleep", "own_bedroom", clock=sleep, until=DAY + wake_next),
    ]


def _homemaker(wake: int, sleep: int, wake_next: int) -> list[_Block]:
    return [
        _Block("sleep", "own_bedroom", until=wake),
        _Block("wake_up", "own_bedroom", dur=WAKE_UP),
        _Block("bath", "own_bathroom", dur=20),
        _Block("cook", "kitchen", dur=60),
        _Block("living", "living", until=720),
        _Block("cook", "kitchen", clock=720, dur=60),
        _Block("lunch", "living", until=840),
        _Block("rest", "own_bedroom", clock=840, dur=60),
        _Block("living", "living", until=1170),
        _Block("cook", "kitchen", clock=1170, dur=60),
        _Block("tv", "living", until=sleep - WIND_DOWN),
        _Block("wind_down", "own_bedroom", clock=sleep - WIND_DOWN, dur=WIND_DOWN),
        _Block("sleep", "own_bedroom", clock=sleep, until=DAY + wake_next),
    ]


def _place(blocks: list[_Block]) -> list[Activity]:
    """Lay blocks end to end. A block waiting for its clock absorbs the gap before it."""
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for block in blocks:
        start = cursor if block.clock is None else max(block.clock, cursor)
        if starts and start > cursor:
            ends[-1] = start  # gap before a clock-anchored block goes to the previous block
        if block.dur is not None:
            end = start + block.dur
        else:
            end = max(block.until, start + MIN_BLOCK)
        starts.append(start)
        ends.append(end)
        cursor = end
    return [
        Activity(b.name, b.room, s, e) for b, s, e in zip(blocks, starts, ends)
    ]


def plan_day(persona: Persona, day_type: DayType, rng: np.random.Generator) -> list[Activity]:
    wake = _sample(rng, _wake_spec(persona, day_type))
    away_day = persona.role == "working_professional" and day_type == DayType.workday

    if persona.role == "homemaker":
        sleep = _sample(rng, persona.sleep_min)
        blocks = _homemaker(wake, sleep, wake)
    elif persona.role == "working_professional":
        sleep = _sample(rng, persona.sleep_min)
        if away_day:
            leave = _sample(rng, persona.leave_min)
            ret = _sample(rng, persona.return_min)
            blocks = _professional_workday(wake, leave, ret, sleep, wake)
        else:
            blocks = _professional_leisure(wake, sleep, wake)
    else:
        raise ValueError(f"unknown persona role: {persona.role!r}")
    return _insert_toilets(_place(blocks), rng)


def _insert_toilets(acts: list[Activity], rng: np.random.Generator) -> list[Activity]:
    """Cut 2-5 toilet visits out of awake home blocks. A visit takes a uniform position over every
    valid split (see _split_count), so no part is shorter than MIN_BLOCK. A visit with no host
    long enough is skipped."""
    count = int(rng.integers(TOILET_VISITS[0], TOILET_VISITS[1] + 1))
    for _ in range(count):
        dur = int(rng.integers(TOILET_MIN, TOILET_MAX + 1))
        hosts = [(i, _split_count(a.end_min - a.start_min - dur)) for i, a in enumerate(acts)
                 if a.room is not None and a.name not in _NOT_HOST and not _beside_visit(acts, i)]
        hosts = [(i, n) for i, n in hosts if n]
        total = sum(n for _, n in hosts)
        if not total:
            continue
        pick = int(rng.integers(total))
        for i, n in hosts:
            if pick < n:
                break
            pick -= n
        host = acts[i]
        rest = host.end_min - host.start_min - dur
        before = _split_at(rest, pick)
        cut = host.start_min + before
        new = [Activity(TOILET, "own_bathroom", cut, cut + dur)]
        if before:
            new.insert(0, Activity(host.name, host.room, host.start_min, cut))
        if rest - before:
            new.append(Activity(host.name, host.room, cut + dur, host.end_min))
        acts[i:i + 1] = new
    return acts


def _beside_visit(acts: list[Activity], i: int) -> bool:
    """True when a toilet visit touches block i, so cutting block i would join two visits."""
    return any(0 <= j < len(acts) and acts[j].name == TOILET for j in (i - 1, i + 1))


def _split_count(rest: int) -> int:
    """How many valid 'minutes before the visit' a host with ``rest`` minutes left over has.
    Valid: 0 or >= MIN_BLOCK before, and 0 or >= MIN_BLOCK after. Sorted: 0, 5..rest-5, rest."""
    if rest == 0:
        return 1
    if rest < MIN_BLOCK:
        return 0
    return 2 + max(0, rest - 2 * MIN_BLOCK + 1)


def _split_at(rest: int, index: int) -> int:
    """The index-th valid 'minutes before the visit' for ``rest`` (see _split_count)."""
    if index == 0:
        return 0
    if index == _split_count(rest) - 1:
        return rest
    return index + MIN_BLOCK - 1


def _wake_spec(persona: Persona, day_type: DayType) -> tuple[float, float]:
    mean, sd = persona.wake_min
    if persona.role == "working_professional" and day_type != DayType.workday:
        mean += 60
    return mean, sd
