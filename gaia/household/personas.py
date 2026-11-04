"""Personas (daily-routine parameters) and household construction.

Time-of-day fields are (mean, sd) in minutes after local midnight. Sleep means may
exceed 1440 to mean after midnight.
"""

from dataclasses import dataclass, field

import numpy as np

from gaia.home.archetypes import ArchetypeSpec
from gaia.home.manifest import HomeManifest


@dataclass(frozen=True)
class Persona:
    role: str
    wake_min: tuple[float, float]
    sleep_min: tuple[float, float]
    leave_min: tuple[float, float] | None
    return_min: tuple[float, float] | None
    forgetfulness: float
    # dict is unhashable; excluded from hash, still compared for equality.
    tech_affinity: dict[str, float] = field(hash=False)
    light_threshold_lux: float
    # Lighting preferences (assumption: defaults until calibrated from ETS).
    cct_pref_offset_k: float = 0.0  # added to the circadian CCT, within +-500 K
    daylight_off_p: float = 0.6  # chance an awake occupant switches the main light off in daylight
    dim_pref: float = 1.0  # multiplier on full-brightness levels, within 0.8-1.0
    # Climate preferences (assumption: defaults until calibrated from ETS).
    ac_affinity: float = 0.7  # chance of using the AC, scaled by the heat index
    ac_setpoint: float = 24.0  # C, the BEE default; the policy sets it when the AC goes on
    fan_affinity: float = 0.8  # chance of switching a room's ceiling fan on when occupied in the heat
    # Media preferences (assumption: defaults until calibrated from ETS).
    tv_source_pref: float = 0.6  # chance the TV takes the set-top box as its source when the room has one
    volume_pref: float = 0.35  # volume set on the TV, streamer, speaker and receiver, within 0.2-0.5
    music_affinity: float = 0.4  # chance of starting music at a cook, dinner, relax or wind_down
    # Utilities (assumption: defaults until calibrated from ETS).
    notice_rate: float = 0.03  # chance per minute that the member switches off a wasted load outside their room
    geyser_lead_min: float = 20.0  # minutes before a bath the geyser goes on: mean, drawn 15-30 per day
    hot_water_always: bool = False  # geyser on before every bath, not only in cool weather

    def __post_init__(self) -> None:
        if not -500.0 <= self.cct_pref_offset_k <= 500.0:
            raise ValueError(f"cct_pref_offset_k out of [-500, 500]: {self.cct_pref_offset_k!r}")
        if not 0.0 <= self.daylight_off_p <= 1.0:
            raise ValueError(f"daylight_off_p out of [0, 1]: {self.daylight_off_p!r}")
        if not 0.8 <= self.dim_pref <= 1.0:
            raise ValueError(f"dim_pref out of [0.8, 1.0]: {self.dim_pref!r}")
        if not 0.0 <= self.ac_affinity <= 1.0:
            raise ValueError(f"ac_affinity out of [0, 1]: {self.ac_affinity!r}")
        if not 20.0 <= self.ac_setpoint <= 27.0:
            raise ValueError(f"ac_setpoint out of [20, 27]: {self.ac_setpoint!r}")
        if not 0.0 <= self.fan_affinity <= 1.0:
            raise ValueError(f"fan_affinity out of [0, 1]: {self.fan_affinity!r}")
        if not 0.0 <= self.tv_source_pref <= 1.0:
            raise ValueError(f"tv_source_pref out of [0, 1]: {self.tv_source_pref!r}")
        if not 0.2 <= self.volume_pref <= 0.5:
            raise ValueError(f"volume_pref out of [0.2, 0.5]: {self.volume_pref!r}")
        if not 0.0 <= self.music_affinity <= 1.0:
            raise ValueError(f"music_affinity out of [0, 1]: {self.music_affinity!r}")
        if not 0.0 <= self.notice_rate <= 1.0:
            raise ValueError(f"notice_rate out of [0, 1]: {self.notice_rate!r}")
        if not 15.0 <= self.geyser_lead_min <= 30.0:
            raise ValueError(f"geyser_lead_min out of [15, 30]: {self.geyser_lead_min!r}")


PERSONAS: dict[str, Persona] = {
    "working_professional": Persona(
        role="working_professional",
        wake_min=(405, 25),
        sleep_min=(1395, 30),
        leave_min=(525, 20),
        return_min=(1140, 40),
        forgetfulness=0.15,
        tech_affinity={"wall": 0.6, "app": 0.3, "voice": 0.1},
        light_threshold_lux=150,
    ),
    "homemaker": Persona(
        role="homemaker",
        wake_min=(360, 20),
        sleep_min=(1350, 25),
        leave_min=None,
        return_min=None,
        forgetfulness=0.08,
        tech_affinity={"wall": 0.8, "app": 0.1, "voice": 0.1},
        light_threshold_lux=120,
    ),
}


@dataclass(frozen=True)
class Member:
    person_id: str
    persona: Persona
    bedroom: str  # room name from the manifest, e.g. "bedroom_1" or "living"
    bathroom: str  # room name from the manifest, e.g. "bathroom_2"


@dataclass(frozen=True)
class Household:
    members: tuple[Member, ...]


def _bedroom_for(index: int, bedrooms: int) -> str:
    """Members 0 and 1 share bedroom_1; member k>=2 takes bedroom_k, then round-robin."""
    if bedrooms == 0:
        return "living"
    if index <= 1:
        return "bedroom_1"
    if index <= bedrooms:
        return f"bedroom_{index}"
    overflow = list(range(2, bedrooms + 1)) or [1]
    return f"bedroom_{overflow[(index - bedrooms - 1) % len(overflow)]}"


def make_household(manifest: HomeManifest, spec: ArchetypeSpec, rng: np.random.Generator) -> Household:
    lo, hi = spec.household_size
    size = int(rng.integers(lo, hi + 1))
    members: list[Member] = []
    for n in range(size):
        if n == 0 or rng.random() >= 0.5:
            persona = PERSONAS["working_professional"]
        else:
            persona = PERSONAS["homemaker"]
        members.append(
            Member(
                person_id=f"{manifest.home_id}:p{n}",
                persona=persona,
                bedroom=_bedroom_for(n, spec.bedrooms),
                bathroom=f"bathroom_{(n % spec.bathrooms) + 1}",
            )
        )
    return Household(members=tuple(members))
