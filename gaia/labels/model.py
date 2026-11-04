"""Hierarchical labels: one record per ground-truth command, naming who meant it, why, and how.

Vocabulary lives in ``schema.yaml``. ``validate_label`` rejects any value outside it.
"""

from dataclasses import dataclass
from importlib.resources import files

import yaml

from gaia.schema.context import LABEL_ARROW_SCHEMA

__all__ = ["ENUMS", "LABEL_ARROW_SCHEMA", "SCHEMA_VERSION", "Label", "validate_label"]

_RAW = yaml.safe_load((files("gaia.labels") / "schema.yaml").read_text(encoding="utf-8"))
SCHEMA_VERSION: int = _RAW["schema_version"]
_ENUM_FIELDS = ("activity", "intent", "trigger", "device_action", "provenance")


def _load_enums(raw: dict) -> dict[str, frozenset[str]]:
    enums: dict[str, frozenset[str]] = {}
    for name in _ENUM_FIELDS:
        values = raw[name]
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate values in labels schema enum {name!r}")
        enums[name] = frozenset(values)
    return enums


ENUMS: dict[str, frozenset[str]] = _load_enums(_RAW)


@dataclass(frozen=True)
class Label:
    event_id: str
    person_id: str | None
    room_id: str | None
    activity: str
    intent: str
    trigger: str
    device_action: str
    source_ref: str
    schema_version: int
    clamped: bool
    provenance: str = "sim_truth"
    confidence: float = 1.0


def validate_label(label: Label) -> None:
    """Raise ValueError if any enum value, the version or the confidence is not allowed."""
    for name, allowed in ENUMS.items():
        value = getattr(label, name)
        if value not in allowed:
            raise ValueError(f"label {name} {value!r} is not in schema v{SCHEMA_VERSION}")
    if label.schema_version != SCHEMA_VERSION:
        raise ValueError(f"label schema_version {label.schema_version} != {SCHEMA_VERSION}")
    if not 0.0 <= label.confidence <= 1.0:
        raise ValueError(f"label confidence out of [0, 1]: {label.confidence!r}")
