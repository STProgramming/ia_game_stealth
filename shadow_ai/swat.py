
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

from .doctrine import FacilityAlert, SWAT_PROFILES, SwatRole
from .facility_map import MapFeature


class SwatAction(IntEnum):
    HOLD = 0
    ADVANCE = 1
    COVER_TEAMMATE = 2
    FOLLOW_SHIELD = 3
    CHECK_CORNER = 4
    CLEAR_SECTOR = 5
    WATCH_ENTRY = 6
    REGROUP = 7
    USE_FLASH = 8
    USE_SMOKE = 9
    FALL_BACK = 10
    RELOAD = 11
    PURSUE = 12
    ARREST = 13


@dataclass
class SmokeCloud:
    center: tuple[float, float]
    radius: float = 2.25
    ttl: float = 5.0


@dataclass
class SwatUnitState:
    uid: str
    role: SwatRole
    pos: np.ndarray
    angle: float
    health: float = 1.0
    ammo: int = 0
    max_ammo: int = 0
    smoke_count: int = 0
    flash_count: int = 0
    alive: bool = True
    assigned_feature_id: Optional[int] = None
    assigned_sector: int = 0
    local_visual: bool = False
    local_visual_confidence: float = 0.0
    local_visual_rel: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    local_visible_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    local_sound: bool = False
    local_sound_rel: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    local_sound_confidence: float = 0.0
    last_seen_pos: Optional[np.ndarray] = None
    last_seen_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    last_seen_age: float = 999.0
    blinded_timer: float = 0.0
    last_action: SwatAction = SwatAction.HOLD

    @property
    def profile(self):
        return SWAT_PROFILES[self.role]


@dataclass
class TeamBlackboard:
    alert: FacilityAlert = FacilityAlert.CONFIRMED
    confirmed_intruder: bool = True
    contact_pos: Optional[np.ndarray] = None
    contact_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    contact_age: float = 999.0
    contact_reported: bool = False
    theft_known: bool = False
    player_compliant_known: bool = False
    cleared_feature_ids: set[int] = field(default_factory=set)
    radio_updates: int = 0
    deployment_reason: str = "high_alert"
    radio_cooldown: float = 0.0
    last_radio_pos: Optional[np.ndarray] = None

    def report_visual(
        self,
        pos: np.ndarray,
        velocity: np.ndarray,
        *,
        compliant: bool = False,
    ) -> None:
        previous_compliance = bool(self.player_compliant_known)
        was_reported = bool(self.contact_reported)

        moved_since_radio = float("inf")
        if self.last_radio_pos is not None:
            moved_since_radio = float(
                np.linalg.norm(pos - self.last_radio_pos)
            )

        # Refresh shared contact every legitimate sighting, but only count a
        # radio transmission when there is actually new tactical information.
        should_radio = (
            not was_reported
            or previous_compliance != bool(compliant)
            or (
                self.radio_cooldown <= 0.0
                and moved_since_radio >= 1.75
            )
        )

        self.confirmed_intruder = True
        self.contact_pos = pos.copy()
        self.contact_velocity = velocity.copy()
        self.contact_age = 0.0
        self.contact_reported = True
        self.player_compliant_known = bool(compliant)
        self.alert = FacilityAlert.CONFIRMED

        if should_radio:
            self.radio_updates += 1
            self.radio_cooldown = 2.0
            self.last_radio_pos = pos.copy()

    def tick(self, dt: float, ttl: float) -> None:
        self.radio_cooldown = max(0.0, self.radio_cooldown - dt)
        if self.contact_pos is not None:
            self.contact_age += dt
            if self.contact_age > ttl:
                self.contact_pos = None
                self.contact_velocity[:] = 0.0
                self.player_compliant_known = False


@dataclass
class SwatConfig:
    width: int = 32
    height: int = 20
    dt: float = 0.20
    max_world_ticks: int = 450
    radio_memory_seconds: float = 14.0
    vision_range: float = 10.0
    fov_degrees: float = 115.0
    close_identification_range: float = 1.5
    hearing_range: float = 6.5
    formation_target_distance: float = 1.8
    feature_reach_distance: float = 0.85
    arrest_distance: float = 1.20
    automatic_engagement_distance: float = 4.0
    smoke_radius: float = 2.25
    smoke_ttl: float = 5.0
    flash_radius: float = 2.60
    flash_effect_seconds: float = 2.4
    reload_fraction_threshold: float = 0.28
    player_walk_speed: float = 0.34
    player_sprint_speed: float = 0.56
    player_flash_speed_multiplier: float = 0.28
    player_wounded_speed_multiplier: float = 0.62
    player_neutralized_health: float = 0.12
    engagement_damage_close: float = 0.085
    engagement_damage_far: float = 0.035


ROLE_ORDER = (
    SwatRole.SHIELD,
    SwatRole.ASSAULT,
    SwatRole.UTILITY,
    SwatRole.HEAVY,
    SwatRole.LEADER,
)
