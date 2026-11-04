from gaia.world.cities import CITIES, City, bengaluru, chennai, delhi, hyderabad, kolkata, mumbai, pune


def test_presets_present():
    assert set(CITIES) == {"hyderabad", "bengaluru", "chennai", "mumbai", "delhi", "kolkata", "pune"}


def test_preset_coordinates():
    assert hyderabad == City("hyderabad", 17.3850, 78.4867)
    assert bengaluru == City("bengaluru", 12.9716, 77.5946)
    assert chennai == City("chennai", 13.0827, 80.2707)
    assert mumbai == City("mumbai", 19.0760, 72.8777)
    assert delhi == City("delhi", 28.6139, 77.2090)
    assert kolkata == City("kolkata", 22.5726, 88.3639)
    assert pune == City("pune", 18.5204, 73.8567)


def test_default_tz_is_ist():
    assert all(c.tz_offset_min == 330 for c in CITIES.values())


def test_city_is_frozen():
    import dataclasses
    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        hyderabad.lat = 0.0
