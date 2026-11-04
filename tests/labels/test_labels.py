import dataclasses

import numpy as np
import pytest

from gaia.behaviour.day_plan import plan_day
from gaia.household.personas import PERSONAS
from gaia.labels.model import ENUMS, Label, validate_label
from gaia.world.calendar import DayType


def _label(**overrides) -> Label:
    fields = dict(
        event_id="h1:000000001", person_id="p1", room_id="r1", activity="living",
        intent="illumination", trigger="darkness", device_action="turn_on",
        source_ref="lights_on_dark", schema_version=1, clamped=False,
    )
    fields.update(overrides)
    return Label(**fields)


def test_label_validation_accepts_schema_values():
    validate_label(_label())


def test_label_validation_rejects_unknown_intent():
    with pytest.raises(ValueError):
        validate_label(_label(intent="make_coffee"))


@pytest.mark.parametrize("field", ["activity", "intent", "trigger", "device_action", "provenance"])
def test_label_validation_rejects_each_enum(field):
    with pytest.raises(ValueError):
        validate_label(_label(**{field: "not_in_schema"}))


def test_label_defaults_are_sim_truth_with_full_confidence():
    label = _label()
    assert label.provenance == "sim_truth"
    assert label.confidence == 1.0


def test_label_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _label().intent = "system"


def test_activity_enum_covers_every_day_plan_activity():
    names = set()
    for persona in PERSONAS.values():
        for kind in (DayType.workday, DayType.weekend):
            for seed in range(40):
                names |= {a.name for a in plan_day(persona, kind, np.random.default_rng(seed))}
    assert names <= set(ENUMS["activity"])
    assert "none" in ENUMS["activity"]


def test_device_action_enum_has_none():
    assert "none" in ENUMS["device_action"]
