"""Policy protocol: a policy reads one TickContext and proposes intents for the actuator.

A policy owns its own random stream (``rng``, derived from the home seed and the policy name)
and must draw only from it. Policy names must be unique within a run.
"""

from typing import Protocol

import numpy as np

from gaia.engine.actuator import Intent, SensorReading
from gaia.engine.context import TickContext


class Policy(Protocol):
    name: str

    def tick(self, ctx: TickContext, rng: np.random.Generator) -> list[Intent | SensorReading]: ...

    def finish(self, ctx: TickContext) -> None:
        """Optional: the engine calls this once after the last tick, with the final context.

        A policy without it needs no finish step; the engine checks for the attribute.
        """
