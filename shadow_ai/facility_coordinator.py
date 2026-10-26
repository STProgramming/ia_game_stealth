
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .doctrine import FacilityAlert


class DeploymentReason(IntEnum):
    NONE = 0
    CONFIRMED_ESCALATION = 1
    MULTIPLE_GUARDS_DOWN = 2
    OBJECTIVE_BREACH = 3
    REPEATED_CONTACT = 4


@dataclass
class FacilitySecurityState:
    alert: FacilityAlert = FacilityAlert.CALM
    guards_active: int = 3
    guards_down: int = 0
    confirmed_theft: bool = False
    reinforcement_requested: bool = False
    repeated_contact_count: int = 0
    swat_deployed: bool = False


def swat_deployment_reason(
    state: FacilitySecurityState,
) -> DeploymentReason:
    """
    Game-level escalation rules.

    No hidden player information is used here. Deployment depends only on
    facility events already legitimately known to security.
    """
    if state.swat_deployed:
        return DeploymentReason.NONE

    if state.guards_down >= 2:
        return DeploymentReason.MULTIPLE_GUARDS_DOWN

    if (
        state.confirmed_theft
        and state.alert >= FacilityAlert.SEARCHING
        and state.reinforcement_requested
    ):
        return DeploymentReason.OBJECTIVE_BREACH

    if (
        state.alert == FacilityAlert.CONFIRMED
        and state.reinforcement_requested
    ):
        return DeploymentReason.CONFIRMED_ESCALATION

    if (
        state.repeated_contact_count >= 3
        and state.alert >= FacilityAlert.SEARCHING
    ):
        return DeploymentReason.REPEATED_CONTACT

    return DeploymentReason.NONE
