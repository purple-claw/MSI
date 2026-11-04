"""Load-class registry v1: data-driven device-class descriptions loaded from classes.yaml.

Each class says what it can do (capabilities), its rated power per state, where the number
came from (a URL, or the literal ``assumption``), its waste rule and the rooms it appears in.
"""

from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

import re

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

CapabilityKind = Literal[
    "binary", "level", "kelvin", "hue", "saturation", "position", "tilt",
    "celsius", "enum", "count", "event",
]
NORMALISED_KINDS = frozenset({"level", "position", "tilt", "hue", "saturation"})
RANGED_KINDS = frozenset({"kelvin", "celsius", "count"})
WASTE_RULE_RE = re.compile(r"unattended|none|overrun:[1-9][0-9]*")
POWER_STATES = frozenset({"on", "standby", "active"})
SENSOR_DOMAIN = "sensing"

FIXTURES: frozenset[str] = frozenset({
    "downlight", "profile", "cove", "pendant", "chandelier", "track", "wall_washer",
    "step", "landscape", "facade", "accent", "strip", "bulb",
})


@dataclass(frozen=True)
class Capability:
    name: str
    kind: str
    lo: float | None
    hi: float | None
    unit: str | None
    values: tuple[str, ...] | None


@dataclass(frozen=True)
class LoadClass:
    name: str
    domain: str
    matter_device_type: str | None
    capabilities: tuple[Capability, ...]
    power_w: dict[str, float]
    power_source: str
    waste_rule: str
    is_sensor: bool
    rooms: tuple[str, ...]


class _CapabilitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: CapabilityKind
    lo: float | None = None
    hi: float | None = None
    unit: str | None = None
    values: tuple[str, ...] | None = None

    @model_validator(mode="before")
    @classmethod
    def _implicit_range(cls, data):
        if not isinstance(data, dict):
            return data
        kind = data.get("kind")
        if kind == "binary" or kind in NORMALISED_KINDS:
            data = {**data, "lo": data.get("lo", 0), "hi": data.get("hi", 1)}
        return data

    @model_validator(mode="after")
    def _check_kind(self):
        if self.kind == "enum":
            if not self.values:
                raise ValueError(f"enum capability {self.name!r} needs values")
        elif self.values is not None:
            raise ValueError(f"only enum capabilities take values: {self.name!r}")
        if self.kind in RANGED_KINDS and (self.lo is None or self.hi is None):
            raise ValueError(f"{self.kind} capability {self.name!r} needs lo and hi")
        if self.kind == "event" and (self.lo is not None or self.hi is not None):
            raise ValueError(f"event capability {self.name!r} takes no range")
        if self.lo is not None and self.hi is not None and not self.lo < self.hi:
            raise ValueError(f"capability {self.name!r} needs lo < hi")
        return self


class _LoadClassSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str
    matter_device_type: str | None
    capabilities: tuple[_CapabilitySpec, ...]
    power_w: dict[str, float]
    power_source: str
    waste_rule: str
    rooms: tuple[str, ...]

    @model_validator(mode="after")
    def _check_fields(self):
        if not (self.power_source == "assumption" or self.power_source.startswith("https://")):
            raise ValueError(f"power_source must be a https URL or 'assumption': {self.power_source!r}")
        unknown = set(self.power_w) - POWER_STATES
        if unknown:
            raise ValueError(f"unknown power_w states: {sorted(unknown)}")
        if not WASTE_RULE_RE.fullmatch(self.waste_rule):
            raise ValueError(f"invalid waste_rule: {self.waste_rule!r}")
        return self


def _load() -> dict[str, LoadClass]:
    raw = yaml.safe_load((files("gaia.registry") / "classes.yaml").read_text(encoding="utf-8"))
    registry: dict[str, LoadClass] = {}
    for name, body in raw.items():
        try:
            spec = _LoadClassSpec.model_validate(body)
        except ValidationError as exc:
            raise ValueError(f"invalid load class {name!r}: {exc}") from exc
        registry[name] = LoadClass(
            name=name,
            domain=spec.domain,
            matter_device_type=spec.matter_device_type,
            capabilities=tuple(
                Capability(c.name, c.kind, c.lo, c.hi, c.unit, c.values) for c in spec.capabilities
            ),
            power_w=dict(spec.power_w),
            power_source=spec.power_source,
            waste_rule=spec.waste_rule,
            is_sensor=spec.domain == SENSOR_DOMAIN,
            rooms=spec.rooms,
        )
    return registry


REGISTRY: dict[str, LoadClass] = _load()


def get_load_class(name: str) -> LoadClass:
    """Return the load class for ``name``; raise KeyError naming the unknown class."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown load class: {name!r}") from None
