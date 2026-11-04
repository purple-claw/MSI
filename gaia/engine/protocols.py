"""Command-to-state latency per control protocol."""

import numpy as np

from gaia.schema.events import Protocol

# (min, max) ms. The knx..other rows predate G2a. dali, lutron, ip and ir were added with the
# room inventory (Task 5) and are assumptions with no source yet; calibrate them later.
_LATENCY_MS: dict[Protocol, tuple[int, int]] = {
    Protocol.knx: (30, 80),
    Protocol.zigbee: (50, 150),
    Protocol.matter: (40, 120),
    Protocol.wifi: (80, 300),
    Protocol.other: (50, 200),
    Protocol.dali: (50, 150),  # assumption
    Protocol.lutron: (50, 200),  # assumption
    Protocol.ip: (80, 300),  # assumption
    Protocol.ir: (100, 400),  # assumption; IR is one-way, so this is the simulated ack time
}


def latency_ms(protocol: Protocol, rng: np.random.Generator) -> int:
    """Uniform integer latency in milliseconds, bounds inclusive."""
    lo, hi = _LATENCY_MS[protocol]
    return int(rng.integers(lo, hi + 1))
