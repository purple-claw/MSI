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
