import numpy as np

from gaia.household.personas import PERSONAS
from gaia.interaction.choose import choose_source, forgets
from gaia.schema.events import Source

N = 10_000


def test_choose_source_distribution():
    rng = np.random.default_rng(1234)
    persona = PERSONAS["working_professional"]
    draws = [choose_source(persona, rng) for _ in range(N)]
    wall_share = sum(1 for s in draws if s is Source.wall) / N
    assert 0.57 <= wall_share <= 0.63
    assert set(draws) <= {Source.wall, Source.app, Source.voice}


def test_choose_source_deterministic_per_rng_state():
    persona = PERSONAS["homemaker"]
    a = [choose_source(persona, np.random.default_rng(7)) for _ in range(50)]
    b = [choose_source(persona, np.random.default_rng(7)) for _ in range(50)]
    assert a == b


def test_forgets_rate():
    rng = np.random.default_rng(99)
    persona = PERSONAS["homemaker"]
    rate = sum(1 for _ in range(N) if forgets(persona, rng)) / N
    assert 0.06 <= rate <= 0.10
