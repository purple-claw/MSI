from gaia.rng import derive_seed, rng_for
def test_derive_seed_stable_and_distinct():
    assert derive_seed(42, "world") == derive_seed(42, "world")
    assert derive_seed(42, "world") != derive_seed(42, "household")
    assert 0 <= derive_seed(1, "x") < 2**63
def test_rng_for_reproducible():
    assert rng_for(7, "behaviour").random() == rng_for(7, "behaviour").random()
