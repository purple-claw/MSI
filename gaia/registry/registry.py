"""Load-class registry v1: data-driven device-class descriptions loaded from classes.yaml.

Each class says what it can do (capabilities), its rated power per state, where the number
came from (a URL, or the literal ``assumption``), its waste rule and the rooms it appears in.
"""

import math
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

CapabilityKind = Literal[
    "binary", "level", "kelvin", "hue", "saturation", "position", "tilt",
    "celsius", "enum", "count", "event",
]
ZERO_ONE_KINDS = frozenset({"binary", "level", "position", "tilt", "hue", "saturation"})
RANGED_KINDS = frozenset({"kelvin", "celsius", "count"})
SEMANTICS_KINDS = frozenset({"position", "tilt"})
POWER_STATES = frozenset({"on", "standby", "active"})
SENSOR_DOMAIN = "sensing"
WASTE_RULE_RE = re.compile(r"unattended(?::[1-9][0-9]*)?|none|overrun:[1-9][0-9]*")

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
    semantics: str | None = None
    initial: float | str | None = None  # the stored value before any command; None: the default


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
    sleeper_counts_as_attended: bool = False


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate keys in any mapping instead of keeping the last one."""


def _construct_unique_mapping(loader, node, deep=False):
    seen: set = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping", node.start_mark,
                f"duplicate key {key!r}", key_node.start_mark,
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


class _CapabilitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    kind: CapabilityKind
    lo: float | None = None
    hi: float | None = None
    unit: str | None = None
    values: tuple[str, ...] | None = None
    semantics: str | None = None
    initial: float | str | None = None

    @model_validator(mode="before")
    @classmethod
    def _implicit_range(cls, data):
        if isinstance(data, dict) and data.get("kind") in ZERO_ONE_KINDS:
            data = {**data, "lo": data.get("lo", 0), "hi": data.get("hi", 1)}
        return data

    @model_validator(mode="after")
    def _check_kind(self):
        if self.kind in ZERO_ONE_KINDS and (self.lo, self.hi) != (0, 1):
            raise ValueError(f"{self.kind} capability {self.name!r} must have lo 0 and hi 1")
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
        if self.kind in SEMANTICS_KINDS and not self.semantics:
            raise ValueError(f"{self.kind} capability {self.name!r} needs semantics")
        if self.initial is not None:
            self._check_initial()
        return self

    def _check_initial(self):
        if self.kind == "event":
            raise ValueError(f"event capability {self.name!r} takes no initial value")
        if self.kind == "enum":
            if self.initial not in self.values:
                raise ValueError(f"enum capability {self.name!r} initial {self.initial!r} not in values")
            return
        if isinstance(self.initial, str) or not math.isfinite(self.initial):
            raise ValueError(f"capability {self.name!r} initial must be a finite number: {self.initial!r}")
        if self.lo is not None and self.initial < self.lo or self.hi is not None and self.initial > self.hi:
            raise ValueError(f"capability {self.name!r} initial {self.initial!r} outside [{self.lo}, {self.hi}]")


class _LoadClassSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str
    matter_device_type: str | None
    capabilities: tuple[_CapabilitySpec, ...]
    power_w: dict[str, float]
    power_source: str
    waste_rule: str
    rooms: tuple[str, ...]
    sleeper_counts_as_attended: bool = False

    @model_validator(mode="after")
    def _check_fields(self):
        if not self.capabilities:
            raise ValueError("capabilities must not be empty")
        if not self.rooms:
            raise ValueError("rooms must not be empty")
        names = [c.name for c in self.capabilities]
        dups = sorted({n for n in names if names.count(n) > 1})
        if dups:
            raise ValueError(f"duplicate capability names: {dups}")
        if not (self.power_source == "assumption" or self.power_source.startswith("https://")):
            raise ValueError(f"power_source must be a https URL or 'assumption': {self.power_source!r}")
        unknown = set(self.power_w) - POWER_STATES
        if unknown:
            raise ValueError(f"unknown power_w states: {sorted(unknown)}")
        if any(v < 0 for v in self.power_w.values()):
            raise ValueError("power_w values must be >= 0")
        if self.domain != SENSOR_DOMAIN and "on" not in self.power_w:
            raise ValueError("non-sensor class needs a power_w 'on' value")
        if not WASTE_RULE_RE.fullmatch(self.waste_rule):
            raise ValueError(f"invalid waste_rule: {self.waste_rule!r}")
        return self


def _build(name: str, spec: _LoadClassSpec) -> LoadClass:
    return LoadClass(
        name=name,
        domain=spec.domain,
        matter_device_type=spec.matter_device_type,
        capabilities=tuple(
            Capability(c.name, c.kind, c.lo, c.hi, c.unit, c.values, c.semantics, c.initial)
            for c in spec.capabilities
        ),
        power_w=dict(spec.power_w),
        power_source=spec.power_source,
        waste_rule=spec.waste_rule,
        is_sensor=spec.domain == SENSOR_DOMAIN,
        rooms=spec.rooms,
        sleeper_counts_as_attended=spec.sleeper_counts_as_attended,
    )


def load_registry(text: str) -> dict[str, LoadClass]:
    """Parse and validate registry YAML text. Raises ValueError naming the offending class."""
    try:
        raw = yaml.load(text, Loader=_StrictLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid registry YAML: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise ValueError("registry YAML must be a non-empty mapping of class names to specs")
    registry: dict[str, LoadClass] = {}
    for name, body in raw.items():
        if not isinstance(name, str):
            raise ValueError(f"load class name must be a string: {name!r}")
        try:
            spec = _LoadClassSpec.model_validate(body)
        except ValidationError as exc:
            raise ValueError(f"invalid load class {name!r}: {exc}") from exc
        registry[name] = _build(name, spec)
    return registry


REGISTRY: dict[str, LoadClass] = load_registry(
    (files("gaia.registry") / "classes.yaml").read_text(encoding="utf-8")
)


def get_load_class(name: str) -> LoadClass:
    """Return the load class for ``name``; raise KeyError naming the unknown class."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown load class: {name!r}") from None
