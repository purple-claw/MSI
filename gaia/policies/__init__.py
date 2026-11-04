"""Household policies. ``POLICY_ORDER`` lists the policies the engine runs, in order."""

from gaia.policies.base import Policy
from gaia.policies.climate import Climate
from gaia.policies.lighting import Lighting
from gaia.policies.shading import Shading

POLICY_ORDER: list[type[Policy]] = [Lighting, Climate, Shading]

__all__ = ["POLICY_ORDER", "Climate", "Lighting", "Policy", "Shading"]
