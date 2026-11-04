"""Policy protocol: a policy reads one TickContext and proposes intents for the actuator.

A policy owns its own random stream (``rng``, derived from the home seed and the policy name)
and must draw only from it. ``finish`` is optional: the engine calls it once, after the last
tick, with the final context.
"""

from typing import Protocol

import numpy as np

from gaia.engine.actuator import Intent
from gaia.engine.context import TickContext


class Policy(Protocol):
    name: str

    def tick(self, ctx: TickContext, rng: np.random.Generator) -> list[Intent]: ...
