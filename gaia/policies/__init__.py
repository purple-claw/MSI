"""Household policies. ``POLICY_ORDER`` lists the policies the engine runs, in order."""

from gaia.policies.base import Policy
from gaia.policies.lighting_basic import LightingBasic

POLICY_ORDER: list[type[Policy]] = [LightingBasic]

__all__ = ["POLICY_ORDER", "LightingBasic", "Policy"]
