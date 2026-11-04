"""Device state model: per-load capability values with range clamping and redundant-set detection.

``DeviceState`` holds the stored value of every stateful capability of one load class.
``HomeDevices`` builds one ``DeviceState`` per load in a home manifest.
"""

import math
from numbers import Real

from gaia.home.manifest import HomeManifest
from gaia.registry.registry import Capability, LoadClass, get_load_class

Value = float | str
_EPS = 1e-9
_CELSIUS_INITIAL = 24.0


def _initial(cap: Capability) -> Value:
    if cap.kind == "kelvin":
        return float(round((cap.lo + cap.hi) / 2))
    if cap.kind == "celsius":
        return min(max(_CELSIUS_INITIAL, cap.lo), cap.hi)
    if cap.kind == "enum":
        return cap.values[0]
    if cap.kind == "count":
        return float(cap.lo)
    return 0.0


def _normalise(cap: Capability, value) -> tuple[Value, bool]:
    if cap.kind == "enum":
        if not isinstance(value, str) or value not in cap.values:
            raise ValueError(f"{cap.name!r} must be one of {list(cap.values)}: {value!r}")
        return value, False
    if not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{cap.name!r} needs a finite number: {value!r}")
    number = float(value)
    if cap.kind == "binary":
        snapped = 1.0 if number >= 0.5 else 0.0
        return snapped, number not in (0.0, 1.0)
    if cap.kind == "count":
        number = float(round(number))
    clamped = False
    if cap.lo is not None and number < cap.lo:
        number, clamped = cap.lo, True
    elif cap.hi is not None and number > cap.hi:
        number, clamped = cap.hi, True
    return number, clamped


def _same(prev: Value, new: Value) -> bool:
    if isinstance(prev, str) or isinstance(new, str):
        return prev == new
    return abs(prev - new) < _EPS


class DeviceState:
    """Stored values for one load. Event capabilities hold no state."""

    def __init__(self, load_class: LoadClass) -> None:
        self._load_class = load_class
        self._caps = {cap.name: cap for cap in load_class.capabilities}
        self._values: dict[str, Value] = {
            cap.name: _initial(cap) for cap in load_class.capabilities if cap.kind != "event"
        }

    @property
    def load_class(self) -> LoadClass:
        return self._load_class

    def get(self, cap: str) -> Value | None:
        self._capability(cap)
        return self._values.get(cap)

    def set(self, cap: str, value) -> tuple[Value | None, Value | None, bool] | None:
        """Store ``value`` for ``cap``. Returns (new, prev, clamped), or None if nothing changed.

        Events are not states: they return (value, None, False) and store nothing.
        """
        capability = self._capability(cap)
        if capability.kind == "event":
            return (value, None, False)
        new, clamped = _normalise(capability, value)
        prev = self._values[cap]
        if _same(prev, new):
            return None
        self._values[cap] = new
        return (new, prev, clamped)

    def requires_power(self, cap: str) -> bool:
        """True when setting ``cap`` on a powered class needs the device switched on first."""
        capability = self._capability(cap)
        return self._has_power() and cap != "power" and capability.kind != "event"

    def is_on(self) -> bool:
        if self._has_power():
            return self._values["power"] == 1
        if "position" in self._caps:
            return self._values["position"] > 0
        return False

    def snapshot(self) -> dict[str, Value]:
        return dict(self._values)

    def _has_power(self) -> bool:
        return "power" in self._caps

    def _capability(self, name: str) -> Capability:
        try:
            return self._caps[name]
        except KeyError:
            raise ValueError(f"{self._load_class.name} has no capability {name!r}") from None


class HomeDevices:
    """One DeviceState per load, in manifest order."""

    def __init__(self, manifest: HomeManifest) -> None:
        self._devices: dict[str, DeviceState] = {}
        for load in manifest.loads:
            if load.load_id in self._devices:
                raise ValueError(f"duplicate load_id {load.load_id!r}")
            self._devices[load.load_id] = DeviceState(get_load_class(load.load_class))

    def get(self, load_id: str) -> DeviceState:
        try:
            return self._devices[load_id]
        except KeyError:
            raise KeyError(f"unknown load: {load_id!r}") from None

    def is_on(self, load_id: str) -> bool:
        return self.get(load_id).is_on()
