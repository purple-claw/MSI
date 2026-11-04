"""Light need: indoor illuminance from outdoor lux, and whether a persona wants light."""

from gaia.home.manifest import Room
from gaia.household.personas import Persona


def indoor_lux(outdoor_lux: float, room: Room) -> float:
    return outdoor_lux * room.window_factor


def light_needed(indoor: float, persona: Persona, activity_name: str) -> bool:
    if activity_name == "sleep":
        return False
    return indoor < persona.light_threshold_lux
