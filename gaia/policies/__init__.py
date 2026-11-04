"""Household policies. ``POLICY_ORDER`` lists the policies the engine runs, in order."""

from gaia.policies.base import Policy
from gaia.policies.climate import Climate
from gaia.policies.lighting import Lighting
from gaia.policies.media import Media
from gaia.policies.notice import Notice
from gaia.policies.shading import Shading
from gaia.policies.utility import Utility

# Notice runs last: it sees every other policy's intents for the minute, and waste it switches
# off is settled before the next minute.
POLICY_ORDER: list[type[Policy]] = [Lighting, Climate, Shading, Media, Utility, Notice]

__all__ = ["POLICY_ORDER", "Climate", "Lighting", "Media", "Notice", "Policy", "Shading", "Utility"]
