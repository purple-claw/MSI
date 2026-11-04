"""Output fingerprint for the performance work: the simulator must not change any output byte.

For each (seed, archetype) pair the digest covers every emitted stream of one home: events,
labels, gt_waste, gt_presence and context, each record serialised as sorted-key JSON in the
order the engine emitted it. The expected digests were computed from the code before the
performance pass and must never be edited to match new code.
"""

import dataclasses
import hashlib
import json
from datetime import date

import pytest

from gaia.engine.sim import HomeSpec, SimResult, simulate_home

SEEDS = (1, 2, 3)
ARCHETYPES = ("apartment_1rk", "apartment_3bhk", "villa")
DAYS = 14

EXPECTED: dict[tuple[int, str], str] = {
    (1, "apartment_1rk"): "c911f0cc40bd0330d748d20ac9141b10da914e80772446bd80f6f431a241d567",
    (2, "apartment_1rk"): "5743b2dc83a29d1e21d9c6b253a98f8243fbf1ab6eb5b0cba0994092e01aa27b",
    (3, "apartment_1rk"): "07814f881c1a7b16f6484717773744b9d8ab4791ef9e98d37071c3eea5b2ad72",
    (1, "apartment_3bhk"): "eed0aace546b6e2fee8b951042f361c1b4adac1282654fe2414e94392a641e33",
    (2, "apartment_3bhk"): "452e2a7b01806a2c9041482b4028d2d165cd399ad0e6fd0c066ebbb06dbee459",
    (3, "apartment_3bhk"): "2dbf4458f9ed27dc7ff1ae32dd10bc0f0541c699fa401acb8e6af75b76c45d5e",
    (1, "villa"): "9634ccf62d0bfcd862bc180fdf57df4c8bbf7b3a3e6d7010808476b705122b73",
    (2, "villa"): "3ff18bee3391c92d1a976817d61a201fe7768b8be1de931ff063a35847e625b7",
    (3, "villa"): "3b61a7411e6b8d74dcc79407892a13fd9c42b63f50f9bcfe0e61710822b85ac2",
}


def _spec(seed: int, archetype: str) -> HomeSpec:
    return HomeSpec(home_id=f"h{seed}", home_seed=seed, city="hyderabad", archetype=archetype,
                    start_date=date(2026, 1, 1), days=DAYS, festivals=frozenset())


def _lines(records) -> list[str]:
    return [json.dumps(dataclasses.asdict(r), sort_keys=True, default=str) for r in records]


def fingerprint(result: SimResult) -> str:
    h = hashlib.sha256()
    for name, records in (("events", result.events), ("labels", result.labels),
                          ("gt_waste", result.gt_waste), ("gt_presence", result.gt_presence),
                          ("context", result.context)):
        h.update(f"[{name}:{len(records)}]\n".encode())
        for line in _lines(records):
            h.update(line.encode())
            h.update(b"\n")
    return h.hexdigest()


@pytest.mark.parametrize("archetype", ARCHETYPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_output_fingerprint_unchanged(seed: int, archetype: str) -> None:
    digest = fingerprint(simulate_home(_spec(seed, archetype)))
    assert digest == EXPECTED[(seed, archetype)]
