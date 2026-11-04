"""Household policies. ``POLICY_ORDER`` lists the policies the engine runs, in order."""

from gaia.policies.base import Policy
from gaia.policies.lighting import Lighting

POLICY_ORDER: list[type[Policy]] = [Lighting]

__all__ = ["POLICY_ORDER", "Lighting", "Policy"]
