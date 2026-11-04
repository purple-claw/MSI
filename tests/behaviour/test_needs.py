import numpy as np

from gaia.behaviour.needs import indoor_lux, light_needed
from gaia.home.manifest import Room
from gaia.household.personas import PERSONAS

ROOM = Room(room_id="living", room_type="living", window_factor=0.5)


def test_indoor_lux_scales_by_window_factor():
    assert indoor_lux(1000.0, ROOM) == 500.0


def test_no_light_when_sleeping():
    persona = PERSONAS["working_professional"]
    assert light_needed(0.0, persona, "sleep") is False


def test_dark_room_needs_light():
    persona = PERSONAS["working_professional"]
    assert light_needed(10.0, persona, "relax") is True


def test_bright_room_needs_no_light():
    persona = PERSONAS["working_professional"]
    assert light_needed(persona.light_threshold_lux + 1, persona, "relax") is False
