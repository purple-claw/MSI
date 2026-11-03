"""City presets used to place homes in the sun and calendar model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float
    tz_offset_min: int = 330


hyderabad = City("hyderabad", 17.3850, 78.4867)
bengaluru = City("bengaluru", 12.9716, 77.5946)
chennai = City("chennai", 13.0827, 80.2707)
mumbai = City("mumbai", 19.0760, 72.8777)
delhi = City("delhi", 28.6139, 77.2090)

CITIES: dict[str, City] = {
    c.name: c for c in (hyderabad, bengaluru, chennai, mumbai, delhi)
}
