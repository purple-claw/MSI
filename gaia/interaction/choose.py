"""Interaction choice: sample a control channel from tech affinity, and forgetfulness."""

from functools import lru_cache

import numpy as np

from gaia.household.personas import Persona
from gaia.schema.events import Source

_CHANNEL_SOURCE: dict[str, Source] = {
    "wall": Source.wall,
    "app": Source.app,
    "voice": Source.voice,
}


@lru_cache(maxsize=256)
def _table(affinity: tuple[tuple[str, float], ...]) -> tuple[tuple[str, ...], np.ndarray]:
    """Sorted channel keys and the cumulative distribution of their normalised probabilities."""
    table = dict(affinity)
    keys = tuple(sorted(table))
    weights = np.array([table[k] for k in keys], dtype=float)
    probs = weights / weights.sum()
    cdf = probs.cumsum()
    cdf /= cdf[-1]
    return keys, cdf


def choose_source(persona: Persona, rng: np.random.Generator) -> Source:
    """Draw one channel. Same draw as ``rng.choice(n, p=probs)``: one uniform per call, mapped
    through the cumulative distribution. ``rng.choice`` re-validates ``p`` on every call, which
    costs several times the draw itself."""
    keys, cdf = _table(tuple(persona.tech_affinity.items()))
    idx = int(cdf.searchsorted(rng.random(), side="right"))
    return _CHANNEL_SOURCE[keys[idx]]


def forgets(persona: Persona, rng: np.random.Generator) -> bool:
    return bool(rng.random() < persona.forgetfulness)
