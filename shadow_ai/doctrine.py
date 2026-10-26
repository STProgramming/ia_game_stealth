from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class PlayerStance(IntEnum):
    CROUCHED = 0
    STANDING = 1


class FacilityAlert(IntEnum):
    CALM = 0
    SUSPICIOUS = 1
    SEARCHING = 2
    CONFIRMED = 3


class VisualBand(IntEnum):
    NONE = 0
    PERIPHERAL = 1
    OUTER = 2
    MID = 3
    CLOSE = 4
    IMMEDIATE = 5


class UnitKind(IntEnum):
    NORMAL_GUARD = 0
    SWAT = 1


class SwatRole(IntEnum):
    NONE = 0
    SHIELD = 1
    ASSAULT = 2
    UTILITY = 3
    HEAVY = 4
    LEADER = 5


@dataclass(frozen=True)
class UnitProfile:
    name: str
    kind: UnitKind
    role: SwatRole
    move_multiplier: float
    turn_multiplier: float
    armor: float
    ammo_multiplier: float
    vision_multiplier: float
    focus_multiplier: float
    has_flashlight: bool
    has_shield: bool
    smoke_count: int
    flash_count: int
    weapon: str


NORMAL_GUARD_PROFILE = UnitProfile(
    name='normal_guard',
    kind=UnitKind.NORMAL_GUARD,
    role=SwatRole.NONE,
    move_multiplier=1.00,
    turn_multiplier=1.00,
    armor=1.00,
    ammo_multiplier=1.00,
    vision_multiplier=1.00,
    focus_multiplier=1.00,
    has_flashlight=True,
    has_shield=False,
    smoke_count=0,
    flash_count=0,
    weapon='service_weapon',
)


SWAT_PROFILES = {
    SwatRole.SHIELD: UnitProfile(
        name='swat_shield',
        kind=UnitKind.SWAT,
        role=SwatRole.SHIELD,
        move_multiplier=0.68,
        turn_multiplier=0.78,
        armor=2.20,
        ammo_multiplier=1.35,
        vision_multiplier=1.08,
        focus_multiplier=1.15,
        has_flashlight=True,
        has_shield=True,
        smoke_count=1,
        flash_count=1,
        weapon='pistol',
    ),
    SwatRole.ASSAULT: UnitProfile(
        name='swat_assault',
        kind=UnitKind.SWAT,
        role=SwatRole.ASSAULT,
        move_multiplier=0.82,
        turn_multiplier=0.90,
        armor=1.75,
        ammo_multiplier=1.80,
        vision_multiplier=1.10,
        focus_multiplier=1.18,
        has_flashlight=True,
        has_shield=False,
        smoke_count=1,
        flash_count=1,
        weapon='carbine',
    ),
    SwatRole.UTILITY: UnitProfile(
        name='swat_utility',
        kind=UnitKind.SWAT,
        role=SwatRole.UTILITY,
        move_multiplier=0.80,
        turn_multiplier=0.90,
        armor=1.65,
        ammo_multiplier=1.55,
        vision_multiplier=1.10,
        focus_multiplier=1.18,
        has_flashlight=True,
        has_shield=False,
        smoke_count=3,
        flash_count=3,
        weapon='carbine',
    ),
    SwatRole.HEAVY: UnitProfile(
        name='swat_heavy',
        kind=UnitKind.SWAT,
        role=SwatRole.HEAVY,
        move_multiplier=0.64,
        turn_multiplier=0.76,
        armor=2.40,
        ammo_multiplier=2.00,
        vision_multiplier=1.05,
        focus_multiplier=1.12,
        has_flashlight=True,
        has_shield=False,
        smoke_count=1,
        flash_count=1,
        weapon='shotgun',
    ),
    SwatRole.LEADER: UnitProfile(
        name='swat_leader',
        kind=UnitKind.SWAT,
        role=SwatRole.LEADER,
        move_multiplier=0.78,
        turn_multiplier=0.88,
        armor=1.85,
        ammo_multiplier=1.70,
        vision_multiplier=1.15,
        focus_multiplier=1.22,
        has_flashlight=True,
        has_shield=False,
        smoke_count=2,
        flash_count=2,
        weapon='carbine',
    ),
}
