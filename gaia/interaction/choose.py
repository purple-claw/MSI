"""Interaction choice: sample a control channel from tech affinity, and forgetfulness."""

import numpy as np

from gaia.household.personas import Persona
from gaia.schema.events import Source

_CHANNEL_SOURCE: dict[str, Source] = {
    "wall": Source.wall,
    "app": Source.app,
    "voice": Source.voice,
}


def choose_source(persona: Persona, rng: np.random.Generator) -> Source:
    keys = sorted(persona.tech_affinity)
    weights = np.array([persona.tech_affinity[k] for k in keys], dtype=float)
    probs = weights / weights.sum()
    idx = int(rng.choice(len(keys), p=probs))
    return _CHANNEL_SOURCE[keys[idx]]


def forgets(persona: Persona, rng: np.random.Generator) -> bool:
    return bool(rng.random() < persona.forgetfulness)
