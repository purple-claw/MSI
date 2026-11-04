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


def test_choose_source_matches_rng_choice_draw_for_draw():
    # The inverse-CDF draw must equal rng.choice(n, p=probs) and consume the same stream.
    persona = PERSONAS["working_professional"]
    keys = sorted(persona.tech_affinity)
    weights = np.array([persona.tech_affinity[k] for k in keys], dtype=float)
    probs = weights / weights.sum()
    a = np.random.default_rng(42)
    b = np.random.default_rng(42)
    expected = [keys[int(a.choice(len(keys), p=probs))] for _ in range(2000)]
    got = [choose_source(persona, b).value for _ in range(2000)]
    assert got == [{"wall": "wall", "app": "app", "voice": "voice"}[k] for k in expected]
    assert a.random() == b.random()
