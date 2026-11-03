"""Command-to-state latency per control protocol."""

import numpy as np

from gaia.schema.events import Protocol

_LATENCY_MS: dict[Protocol, tuple[int, int]] = {
    Protocol.knx: (30, 80),
    Protocol.zigbee: (50, 150),
    Protocol.matter: (40, 120),
    Protocol.wifi: (80, 300),
    Protocol.other: (50, 200),
}


def latency_ms(protocol: Protocol, rng: np.random.Generator) -> int:
    """Uniform integer latency in milliseconds, bounds inclusive."""
    lo, hi = _LATENCY_MS[protocol]
    return int(rng.integers(lo, hi + 1))
