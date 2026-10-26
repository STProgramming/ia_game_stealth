
from __future__ import annotations

import math
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .doctrine import FacilityAlert, SWAT_PROFILES, SwatRole
from .facility_map import FacilityMap, MapFeature
from .swat import (
    ROLE_ORDER,
    SmokeCloud,
    SwatAction,
    SwatConfig,
    SwatUnitState,
    TeamBlackboard,
)


def _norm(v: np.ndarray) -> np.ndarray:
    mag = float(np.linalg.norm(v))
    if mag <= 1e-8:
        return np.zeros_like(v)
    return v / mag


class SwatTeamCore:
    """
    Physical five-unit SWAT simulation.

    Important epistemic rule:
    - unit observation never contains hidden live player coordinates;
    - blackboard contact is populated only by legitimate team observations;
    - utility actions target only local/blackboard/assigned semantic points.
    """

    def __init__(
        self,
        config: Optional[SwatConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.cfg = config or SwatConfig()
        self.map = FacilityMap(self.cfg.width, self.cfg.height)
        self.rng = np.random.default_rng(seed)
        self.roles = ROLE_ORDER
        self.agent_ids = [f"swat_{role.name.lower()}" for role in self.roles]
        self.units: dict[str, SwatUnitState] = {}
        self.blackboard = TeamBlackboard()
        self.smoke_clouds: list[SmokeCloud] = []
        self.player_pos = np.zeros(2, dtype=np.float32)
        self.player_velocity = np.zeros(2, dtype=np.float32)
        self.player_compliant = False
        self.player_flash_timer = 0.0
        self.player_health = 1.0
        self.player_neutralized = False
        self.player_captured = False
        self.player_escaped = False
        self.contact_ever_confirmed = False
        self.world_ticks = 0
        self.scenario = "entry_clear"
        self.last_team_reward = 0.0
        self.utility_effect_event = False
        self.contact_event = False
        self.capture_event = False
        self.feature_clear_event = False

    # ---------------------------- reset ----------------------------
    def reset_core(
        self,
        seed: Optional[int] = None,
        scenario: Optional[str] = None,
    ) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.world_ticks = 0
        self.scenario = scenario or str(
            self.rng.choice(
                [
                    "entry_clear",
                    "corner_clear",
                    "lost_contact",
                    "confirmed_visual",
                    "utility",
                    "arrest",
                    "objective_breach",
                ],
                p=[0.18,0.15,0.15,0.18,0.12,0.10,0.12],
            )
        )
        self.blackboard = TeamBlackboard(
            alert=FacilityAlert.CONFIRMED,
            confirmed_intruder=True,
            theft_known=self.scenario == "objective_breach",
            deployment_reason=self.scenario,
        )
        self.smoke_clouds = []
        self.player_compliant = self.scenario == "arrest"
        self.player_flash_timer = 0.0
        self.player_health = 1.0
        self.player_neutralized = False
        self.player_captured = False
        self.player_escaped = False
        self.contact_ever_confirmed = False
        self.last_team_reward = 0.0
        self.utility_effect_event = False
        self.contact_event = False
        self.capture_event = False
        self.feature_clear_event = False

        entries = [
            np.array([30.0, 1.0], dtype=np.float32),
            np.array([29.0, 1.0], dtype=np.float32),
            np.array([30.0, 2.0], dtype=np.float32),
            np.array([28.5, 1.0], dtype=np.float32),
            np.array([29.0, 2.0], dtype=np.float32),
        ]
        self.units = {}
        for idx, role in enumerate(self.roles):
            profile = SWAT_PROFILES[role]
            max_ammo = max(6, int(round(12 * profile.ammo_multiplier)))
            pos = entries[idx].copy()
            self.units[self.agent_ids[idx]] = SwatUnitState(
                uid=self.agent_ids[idx],
                role=role,
                pos=pos,
                angle=math.pi,
                ammo=max_ammo,
                max_ammo=max_ammo,
                smoke_count=profile.smoke_count,
                flash_count=profile.flash_count,
                assigned_sector=idx % 4,
            )

        self._assign_semantic_targets()

        if self.scenario == "confirmed_visual":
            self.player_pos = np.array([23.0, 4.0], dtype=np.float32)
            # Contact is known before entry only if a real facility report
            # existed; this scenario represents that explicit precondition.
            self.blackboard.contact_pos = np.array([23.0, 4.0], dtype=np.float32)
            self.blackboard.contact_velocity = np.array([-0.25, 0.0], dtype=np.float32)
            self.blackboard.contact_age = 0.8
            self.blackboard.contact_reported = True
        elif self.scenario == "lost_contact":
            self.player_pos = np.array([10.0, 15.5], dtype=np.float32)
            self.blackboard.contact_pos = np.array([15.0, 10.0], dtype=np.float32)
            self.blackboard.contact_velocity = np.array([-0.4, 0.15], dtype=np.float32)
            self.blackboard.contact_age = 4.0
            self.blackboard.contact_reported = True
        elif self.scenario == "arrest":
            self.player_pos = np.array([25.5, 3.0], dtype=np.float32)
            self.blackboard.contact_pos = self.player_pos.copy()
            self.blackboard.contact_age = 0.2
            self.blackboard.contact_reported = True
            self.blackboard.player_compliant_known = True
        elif self.scenario == "utility":
            self.player_pos = np.array([20.5, 8.5], dtype=np.float32)
            self.blackboard.contact_pos = np.array([20.0, 8.0], dtype=np.float32)
            self.blackboard.contact_age = 1.2
            self.blackboard.contact_reported = True
        elif self.scenario == "corner_clear":
            self.player_pos = np.array([16.0, 15.5], dtype=np.float32)
            self.blackboard.contact_pos = np.array([18.0, 13.0], dtype=np.float32)
            self.blackboard.contact_age = 3.0
            self.blackboard.contact_reported = True
        else:
            self.player_pos = np.array([8.0, 16.0], dtype=np.float32)

        self.contact_ever_confirmed = (
            self.blackboard.contact_pos is not None
        )
        self.player_velocity[:] = 0.0
        self._sense_all()

    def _assign_semantic_targets(self) -> None:
        used: set[int] = set()
        preferences = {
            SwatRole.SHIELD: {MapFeature.DOOR, MapFeature.ROOM_ENTRY},
            SwatRole.ASSAULT: {MapFeature.CORNER, MapFeature.ROOM_ENTRY},
            SwatRole.UTILITY: {MapFeature.INTERSECTION, MapFeature.LONG_CORRIDOR},
            SwatRole.HEAVY: {MapFeature.COVER_POINT, MapFeature.DOOR},
            SwatRole.LEADER: {MapFeature.INTERSECTION, MapFeature.ROOM_ENTRY},
        }
        for uid in self.agent_ids:
            unit = self.units[uid]
            feature = self.map.nearest_feature(
                unit.pos,
                allowed=preferences[unit.role],
                exclude_ids=used,
            )
            if feature is not None:
                unit.assigned_feature_id = feature.feature_id
                unit.assigned_sector = feature.sector_id
                used.add(feature.feature_id)

    # -------------------------- perception --------------------------
    def _segment_hits_smoke(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> bool:
        return not self.map.line_clear(a, b, self.smoke_clouds)

    def _unit_can_see_player(self, unit: SwatUnitState) -> bool:
        if not unit.alive or unit.blinded_timer > 0:
            return False
        rel = self.player_pos - unit.pos
        distance = float(np.linalg.norm(rel))
        max_range = self.cfg.vision_range * unit.profile.vision_multiplier
        if distance > max_range:
            return False
        if not self.map.line_clear(unit.pos, self.player_pos, self.smoke_clouds):
            return False
        if distance <= self.cfg.close_identification_range:
            return True
        facing = np.array(
            [math.cos(unit.angle), math.sin(unit.angle)],
            dtype=np.float32,
        )
        direction = rel / max(distance, 1e-6)
        dot = float(np.dot(facing, direction))
        return dot >= math.cos(math.radians(self.cfg.fov_degrees / 2))

    def _sense_unit(self, unit: SwatUnitState) -> None:
        unit.local_visual = False
        unit.local_visual_confidence = 0.0
        unit.local_visual_rel[:] = 0.0
        unit.local_visible_velocity[:] = 0.0
        unit.local_sound = False
        unit.local_sound_rel[:] = 0.0
        unit.local_sound_confidence = 0.0

        if self._unit_can_see_player(unit):
            rel = self.player_pos - unit.pos
            d = float(np.linalg.norm(rel))
            unit.local_visual = True
            unit.local_visual_confidence = 1.0
            unit.local_visual_rel = rel.astype(np.float32)
            unit.local_visible_velocity = self.player_velocity.copy()
            unit.last_seen_pos = self.player_pos.copy()
            unit.last_seen_velocity = self.player_velocity.copy()
            unit.last_seen_age = 0.0
            self.blackboard.report_visual(
                self.player_pos,
                self.player_velocity,
                compliant=self.player_compliant,
            )
            self.contact_ever_confirmed = True
            self.contact_event = True
            return

        if unit.last_seen_pos is not None:
            unit.last_seen_age += self.cfg.dt
            if unit.last_seen_age > self.cfg.radio_memory_seconds:
                unit.last_seen_pos = None

        speed = float(np.linalg.norm(self.player_velocity))
        if speed > 0.25:
            rel = self.player_pos - unit.pos
            distance = float(np.linalg.norm(rel))
            hearing = self.cfg.hearing_range * min(1.4, 0.7 + speed)
            if distance <= hearing:
                noise = self.rng.normal(0.0, 0.45, size=2).astype(np.float32)
                unit.local_sound = True
                unit.local_sound_rel = (rel + noise).astype(np.float32)
                unit.local_sound_confidence = float(
                    np.clip(1.0 - distance / max(hearing, 1e-6), 0.15, 0.9)
                )

    def _sense_all(self) -> None:
        for unit in self.units.values():
            self._sense_unit(unit)

    # -------------------------- semantics ---------------------------
    def _feature_for(self, unit: SwatUnitState):
        if unit.assigned_feature_id is None:
            return None
        if not (0 <= unit.assigned_feature_id < len(self.map.semantic_points)):
            return None
        return self.map.semantic_points[unit.assigned_feature_id]

    def _nearest_teammate(self, uid: str):
        unit = self.units[uid]
        others = [
            other for oid, other in self.units.items()
            if oid != uid and other.alive
        ]
        if not others:
            return None
        return min(
            others,
            key=lambda other: float(np.linalg.norm(other.pos - unit.pos)),
        )

    def _role_unit(self, role: SwatRole) -> Optional[SwatUnitState]:
        for unit in self.units.values():
            if unit.role == role and unit.alive:
                return unit
        return None

    def _team_centroid(self) -> np.ndarray:
        alive = [u.pos for u in self.units.values() if u.alive]
        if not alive:
            return np.zeros(2, dtype=np.float32)
        return np.mean(np.stack(alive), axis=0).astype(np.float32)

    def _legitimate_tactical_target(
        self,
        unit: SwatUnitState,
    ) -> Optional[np.ndarray]:
        if unit.local_visual:
            return unit.pos + unit.local_visual_rel
        if unit.local_sound:
            return unit.pos + unit.local_sound_rel
        if unit.last_seen_pos is not None:
            return unit.last_seen_pos.copy()
        if self.blackboard.contact_pos is not None:
            return self.blackboard.contact_pos.copy()
        feature = self._feature_for(unit)
        if feature is not None:
            return np.array(feature.position, dtype=np.float32)
        return None

    # --------------------------- actions ----------------------------
    def _move_unit(
        self,
        unit: SwatUnitState,
        target: Optional[np.ndarray],
        multiplier: float = 1.0,
    ) -> None:
        if target is None or not unit.alive:
            return
        speed = 0.62 * unit.profile.move_multiplier * multiplier
        new_pos = self.map.move_towards(
            unit.pos,
            target,
            speed,
            self.cfg.dt,
        )
        delta = target - new_pos
        if float(np.linalg.norm(delta)) > 1e-6:
            unit.angle = math.atan2(float(delta[1]), float(delta[0]))
        unit.pos = new_pos

    def _formation_error(self, uid: str) -> float:
        unit = self.units[uid]
        shield = self._role_unit(SwatRole.SHIELD)
        if shield is None or unit.role == SwatRole.SHIELD:
            return 0.0
        d = float(np.linalg.norm(unit.pos - shield.pos))
        return min(1.0, abs(d - self.cfg.formation_target_distance) / 5.0)

    def _clear_nearby_feature(self, unit: SwatUnitState) -> None:
        feature = self._feature_for(unit)
        if feature is None:
            return
        p = np.array(feature.position, dtype=np.float32)
        if float(np.linalg.norm(unit.pos - p)) <= self.cfg.feature_reach_distance:
            if feature.feature_id not in self.blackboard.cleared_feature_ids:
                self.blackboard.cleared_feature_ids.add(feature.feature_id)
                self.feature_clear_event = True

    def _fresh_contact(self) -> bool:
        return (
            self.blackboard.contact_pos is not None
            and self.blackboard.contact_age <= 2.5
        )

    def _compliant_contact_known(self, unit: SwatUnitState) -> bool:
        return bool(
            (unit.local_visual and self.player_compliant)
            or self.blackboard.player_compliant_known
        )

    def _cover_available_for_arrest(
        self,
        arresting_unit: SwatUnitState,
        target: np.ndarray,
    ) -> bool:
        for other in self.units.values():
            if other.uid == arresting_unit.uid or not other.alive:
                continue
            distance = float(np.linalg.norm(other.pos - target))
            if distance <= 5.0 and self.map.line_clear(
                other.pos,
                target,
                self.smoke_clouds,
            ):
                return True
        return False

    def _smoke_need(self, unit: SwatUnitState) -> bool:
        if unit.smoke_count <= 0:
            return False
        if unit.local_visual:
            return False

        stale_report = (
            self.blackboard.contact_pos is not None
            and 2.5 < self.blackboard.contact_age <= self.cfg.radio_memory_seconds
        )
        long_corridor = False
        for point in self.map.semantic_points:
            if point.feature != MapFeature.LONG_CORRIDOR:
                continue
            p = np.asarray(point.position, dtype=np.float32)
            if float(np.linalg.norm(p - unit.pos)) <= 2.2:
                long_corridor = True
                break
        return bool(stale_report or long_corridor)

    def _flash_need(self, unit: SwatUnitState) -> bool:
        if unit.flash_count <= 0:
            return False
        if unit.local_visual:
            return True
        return bool(
            self.blackboard.contact_pos is not None
            and self.blackboard.contact_age <= 1.5
        )

    def apply_action(self, uid: str, action: SwatAction) -> float:
        unit = self.units[uid]
        if not unit.alive:
            return -0.02

        unit.last_action = action
        reward = -0.003
        feature = self._feature_for(unit)
        target = self._legitimate_tactical_target(unit)
        role = unit.role
        fresh_contact = self._fresh_contact()
        compliant_known = self._compliant_contact_known(unit)

        # Learned policy still chooses the action. These are hierarchy rewards,
        # teaching that live/recent contact outranks routine room clearing.
        if compliant_known:
            if role == SwatRole.LEADER:
                reward += 0.48 if action == SwatAction.ARREST else -0.18
            elif action in {
                SwatAction.COVER_TEAMMATE,
                SwatAction.HOLD,
                SwatAction.WATCH_ENTRY,
            }:
                reward += 0.10
        elif fresh_contact:
            if role == SwatRole.ASSAULT:
                if action == SwatAction.PURSUE:
                    reward += 0.46
                elif action in {
                    SwatAction.CHECK_CORNER,
                    SwatAction.CLEAR_SECTOR,
                    SwatAction.WATCH_ENTRY,
                    SwatAction.REGROUP,
                }:
                    reward -= 0.18
            elif role == SwatRole.SHIELD and action == SwatAction.ADVANCE:
                reward += 0.18
            elif role == SwatRole.UTILITY:
                if action == SwatAction.USE_FLASH and self._flash_need(unit):
                    reward += 0.30
                elif action == SwatAction.USE_SMOKE and unit.local_visual:
                    reward -= 0.18

        if action == SwatAction.HOLD:
            reward += 0.02

        elif action == SwatAction.ADVANCE:
            self._move_unit(unit, target)
            reward += 0.05
            if role == SwatRole.SHIELD:
                reward += 0.10

        elif action == SwatAction.COVER_TEAMMATE:
            mate = self._nearest_teammate(uid)
            if mate is None:
                reward -= 0.12
            else:
                desired = mate.pos + np.array(
                    [math.cos(mate.angle + math.pi / 2),
                     math.sin(mate.angle + math.pi / 2)],
                    dtype=np.float32,
                ) * 1.3
                self._move_unit(unit, desired, 0.75)
                reward += 0.14

        elif action == SwatAction.FOLLOW_SHIELD:
            shield = self._role_unit(SwatRole.SHIELD)
            if shield is None or shield.uid == uid:
                reward -= 0.16
            else:
                behind = shield.pos - np.array(
                    [math.cos(shield.angle), math.sin(shield.angle)],
                    dtype=np.float32,
                ) * self.cfg.formation_target_distance
                before = self._formation_error(uid)
                self._move_unit(unit, behind)
                after = self._formation_error(uid)
                reward += 0.18 + 0.12 * max(0.0, before - after)

        elif action == SwatAction.CHECK_CORNER:
            corner = self.map.nearest_feature(
                unit.pos,
                allowed={MapFeature.CORNER},
                exclude_ids=self.blackboard.cleared_feature_ids,
            )
            if corner is None:
                reward -= 0.10
            else:
                self._move_unit(
                    unit,
                    np.array(corner.position, dtype=np.float32),
                    0.72,
                )
                unit.assigned_feature_id = corner.feature_id
                reward += 0.18
                self._clear_nearby_feature(unit)

        elif action == SwatAction.CLEAR_SECTOR:
            if feature is None or feature.feature_id in self.blackboard.cleared_feature_ids:
                feature = self.map.nearest_feature(
                    unit.pos,
                    allowed={
                        MapFeature.ROOM_ENTRY,
                        MapFeature.INTERSECTION,
                        MapFeature.CORNER,
                        MapFeature.DEAD_END,
                    },
                    exclude_ids=self.blackboard.cleared_feature_ids,
                )
                if feature is not None:
                    unit.assigned_feature_id = feature.feature_id
            if feature is None:
                reward -= 0.08
            else:
                self._move_unit(
                    unit,
                    np.array(feature.position, dtype=np.float32),
                    0.82,
                )
                reward += 0.16
                self._clear_nearby_feature(unit)

        elif action == SwatAction.WATCH_ENTRY:
            door = self.map.nearest_feature(
                unit.pos,
                allowed={MapFeature.DOOR, MapFeature.ROOM_ENTRY},
            )
            if door is None:
                reward -= 0.10
            else:
                p = np.array(door.position, dtype=np.float32)
                self._move_unit(unit, p, 0.65)
                reward += 0.14

        elif action == SwatAction.REGROUP:
            center = self._team_centroid()
            self._move_unit(unit, center, 0.78)
            reward += 0.10

        elif action == SwatAction.USE_FLASH:
            if unit.flash_count <= 0:
                reward -= 0.22
            elif target is None:
                reward -= 0.18
            elif role not in {SwatRole.UTILITY, SwatRole.LEADER}:
                reward -= 0.08
            elif not self._flash_need(unit):
                reward -= 0.16
            else:
                unit.flash_count -= 1
                effect = (
                    float(np.linalg.norm(self.player_pos - target))
                    <= self.cfg.flash_radius
                    and self.map.line_clear(target, self.player_pos)
                )
                if effect:
                    self.player_flash_timer = max(
                        self.player_flash_timer,
                        self.cfg.flash_effect_seconds,
                    )
                    self.utility_effect_event = True
                    reward += 0.52
                else:
                    reward += 0.08

        elif action == SwatAction.USE_SMOKE:
            if unit.smoke_count <= 0:
                reward -= 0.22
            elif target is None:
                reward -= 0.18
            elif role not in {SwatRole.UTILITY, SwatRole.LEADER, SwatRole.SHIELD}:
                reward -= 0.08
            elif not self._smoke_need(unit):
                reward -= 0.16
            else:
                unit.smoke_count -= 1
                # Smoke goes on the approach line, not magically on hidden truth.
                smoke_target = unit.pos + (target - unit.pos) * 0.58
                self.smoke_clouds.append(
                    SmokeCloud(
                        center=(float(smoke_target[0]), float(smoke_target[1])),
                        radius=self.cfg.smoke_radius,
                        ttl=self.cfg.smoke_ttl,
                    )
                )
                reward += 0.42

        elif action == SwatAction.FALL_BACK:
            access = min(
                self.map.access_points,
                key=lambda p: float(np.linalg.norm(p - unit.pos)),
            )
            self._move_unit(unit, access, 0.86)
            reward += 0.08

        elif action == SwatAction.RELOAD:
            fraction = unit.ammo / max(unit.max_ammo, 1)
            if fraction <= self.cfg.reload_fraction_threshold:
                unit.ammo = unit.max_ammo
                reward += 0.28
            else:
                reward -= 0.12

        elif action == SwatAction.PURSUE:
            if unit.local_visual:
                self._move_unit(
                    unit,
                    unit.pos + unit.local_visual_rel,
                    1.05,
                )
                reward += 0.40
            elif self.blackboard.contact_pos is not None:
                self._move_unit(unit, self.blackboard.contact_pos, 0.92)
                reward += 0.16
            else:
                reward -= 0.20

        elif action == SwatAction.ARREST:
            if role != SwatRole.LEADER:
                reward -= 0.24
            elif not compliant_known or target is None:
                reward -= 0.30
            else:
                distance = float(np.linalg.norm(unit.pos - target))
                if distance > self.cfg.arrest_distance:
                    before = distance
                    self._move_unit(unit, target, 0.82)
                    after = float(np.linalg.norm(unit.pos - target))
                    reward += 0.26 + 0.24 * max(0.0, before - after)
                elif self._cover_available_for_arrest(unit, target):
                    self.player_captured = True
                    self.capture_event = True
                    reward += 1.10
                else:
                    reward += 0.10

        self._clear_nearby_feature(unit)
        return reward

    # -------------------------- world tick --------------------------
    def _player_step(self) -> None:
        if self.player_captured or self.player_neutralized:
            self.player_velocity[:] = 0.0
            return

        if self.player_compliant:
            self.player_velocity[:] = 0.0
            return

        if self.contact_ever_confirmed:
            speed = self.cfg.player_sprint_speed
        else:
            # Before discovery the training intruder behaves stealthily instead
            # of sprinting in a straight line to the exit. This prevents the
            # curriculum from rewarding omniscient pursuit.
            speed = self.cfg.player_walk_speed * 0.72

        if self.player_health < 0.55:
            speed *= self.cfg.player_wounded_speed_multiplier

        if self.player_flash_timer > 0:
            speed *= self.cfg.player_flash_speed_multiplier

        threats = [
            unit.pos for unit in self.units.values()
            if unit.alive and unit.local_visual
        ]
        if threats:
            center = np.mean(np.stack(threats), axis=0)
            direction = _norm(self.player_pos - center)
        elif self.contact_ever_confirmed:
            # Player does NOT read the team blackboard. This fixed vector only
            # represents evasive movement after the intruder knows contact was
            # made.
            direction = _norm(np.array([-0.7, 0.35], dtype=np.float32))
        else:
            # Slow pre-contact movement inside the facility.
            angle = 0.22 * math.sin(self.world_ticks * 0.11)
            direction = _norm(
                np.array(
                    [-0.28 + angle, 0.12 + 0.5 * angle],
                    dtype=np.float32,
                )
            )

        intended = self.player_pos + direction * speed * self.cfg.dt
        if self.map.line_clear(self.player_pos, intended):
            self.player_velocity = (intended - self.player_pos) / self.cfg.dt
            self.player_pos = intended.astype(np.float32)
        else:
            jitter = float(self.rng.uniform(-1.1, 1.1))
            alt = np.array(
                [math.cos(jitter), math.sin(jitter)],
                dtype=np.float32,
            )
            intended = self.player_pos + alt * speed * self.cfg.dt
            if self.map.line_clear(self.player_pos, intended):
                self.player_velocity = (intended - self.player_pos) / self.cfg.dt
                self.player_pos = intended.astype(np.float32)
            else:
                self.player_velocity[:] = 0.0

        x, y = self.map.cell(self.player_pos)
        if x <= 1 and y >= self.cfg.height - 3:
            self.player_escaped = True

    def _player_counterfire(self) -> float:
        """
        Abstract game-only incoming fire model.

        This activates armor/shield/health without turning tactical intent into
        a low-level shooting controller. The player can only target a SWAT unit
        through real line-of-sight; smoke blocks the shot. Flash reduces player
        effectiveness. Shield protection applies only from the unit's front arc.
        """
        if self.player_captured or self.player_compliant:
            return 0.0

        candidates = []
        for unit in self.units.values():
            if not unit.alive:
                continue
            distance = float(np.linalg.norm(unit.pos - self.player_pos))
            if distance > 7.0:
                continue
            if not self.map.line_clear(
                self.player_pos,
                unit.pos,
                self.smoke_clouds,
            ):
                continue
            candidates.append((distance, unit))

        if not candidates:
            return 0.0

        distance, target = min(candidates, key=lambda item: item[0])

        # Deterministic expected-damage style model keeps training stable.
        range_factor = float(np.clip(1.0 - distance / 9.0, 0.15, 0.85))
        flash_factor = 0.28 if self.player_flash_timer > 0.0 else 1.0
        raw_damage = 0.055 * range_factor * flash_factor

        armor_factor = max(1.0, float(target.profile.armor))
        damage = raw_damage / armor_factor

        if target.profile.has_shield:
            to_player = self.player_pos - target.pos
            to_player = _norm(to_player)
            facing = np.array(
                [math.cos(target.angle), math.sin(target.angle)],
                dtype=np.float32,
            )
            # Strong protection only if the shield is actually facing threat.
            if float(np.dot(facing, to_player)) >= math.cos(math.radians(55.0)):
                damage *= 0.22

        target.health = max(0.0, target.health - damage)
        if target.health <= 0.0:
            target.alive = False
            return -0.50

        return -0.015 * damage / 0.055

    def _automatic_contact_resolution(self) -> float:
        reward = 0.0
        for unit in self.units.values():
            if not unit.alive or not unit.local_visual:
                continue
            distance = float(np.linalg.norm(unit.pos - self.player_pos))
            if self.player_compliant:
                continue

            if (
                distance <= self.cfg.automatic_engagement_distance
                and unit.ammo > 0
            ):
                unit.ammo -= 1

                # Deterministic game abstraction: confirmed moving intruder can
                # be wounded/neutralized. The policy still chooses tactical
                # intent, never individual shots.
                closeness = float(
                    np.clip(
                        1.0 - distance / self.cfg.automatic_engagement_distance,
                        0.0,
                        1.0,
                    )
                )
                damage = (
                    self.cfg.engagement_damage_far
                    + (
                        self.cfg.engagement_damage_close
                        - self.cfg.engagement_damage_far
                    ) * closeness
                )
                if self.player_flash_timer > 0:
                    damage *= 1.12
                self.player_health = max(
                    0.0,
                    self.player_health - damage,
                )
                reward += 0.035

                if self.player_health <= self.cfg.player_neutralized_health:
                    self.player_neutralized = True
                    self.player_velocity[:] = 0.0
                    reward += 1.25
                    break

            if distance <= 0.62:
                self.player_captured = True
                self.capture_event = True
                reward += 0.70
                break
        return reward

    def advance_world(self) -> float:
        self.world_ticks += 1
        self.contact_event = False
        self.capture_event = False
        self.feature_clear_event = False
        self.utility_effect_event = False

        self._player_step()

        self.player_flash_timer = max(
            0.0,
            self.player_flash_timer - self.cfg.dt,
        )
        for cloud in self.smoke_clouds:
            cloud.ttl -= self.cfg.dt
        self.smoke_clouds = [c for c in self.smoke_clouds if c.ttl > 0]

        for unit in self.units.values():
            unit.blinded_timer = max(
                0.0,
                unit.blinded_timer - self.cfg.dt,
            )

        self.blackboard.tick(
            self.cfg.dt,
            self.cfg.radio_memory_seconds,
        )
        self._sense_all()
        reward = self._automatic_contact_resolution()
        reward += self._player_counterfire()

        cleared = len(self.blackboard.cleared_feature_ids)
        reward += 0.015 * min(cleared, 8)
        if self.contact_event:
            reward += 0.12
        if self.capture_event:
            reward += 1.0
        if self.player_escaped:
            reward -= 1.2

        self.last_team_reward = reward
        return reward

    # -------------------------- observation -------------------------
    def observation(self, uid: str) -> np.ndarray:
        unit = self.units[uid]
        w = float(self.cfg.width - 1)
        h = float(self.cfg.height - 1)
        role_one_hot = [
            1.0 if unit.role == role else 0.0
            for role in self.roles
        ]
        nearest = self._nearest_teammate(uid)
        if nearest is not None:
            nrel = nearest.pos - unit.pos
            nrel = np.array(
                [nrel[0] / self.cfg.width, nrel[1] / self.cfg.height],
                dtype=np.float32,
            )
            nrole = float(self.roles.index(nearest.role)) / 4.0
        else:
            nrel = np.zeros(2, dtype=np.float32)
            nrole = 0.0

        center_rel = self._team_centroid() - unit.pos
        center_rel = np.array(
            [center_rel[0] / self.cfg.width, center_rel[1] / self.cfg.height],
            dtype=np.float32,
        )

        leader = self._role_unit(SwatRole.LEADER)
        shield = self._role_unit(SwatRole.SHIELD)
        lrel = np.zeros(2, dtype=np.float32)
        srel = np.zeros(2, dtype=np.float32)
        if leader is not None:
            d = leader.pos - unit.pos
            lrel = np.array([d[0]/self.cfg.width, d[1]/self.cfg.height])
        if shield is not None:
            d = shield.pos - unit.pos
            srel = np.array([d[0]/self.cfg.width, d[1]/self.cfg.height])

        local_vis_rel = np.zeros(2, dtype=np.float32)
        if unit.local_visual:
            local_vis_rel = np.array(
                [
                    unit.local_visual_rel[0] / self.cfg.vision_range,
                    unit.local_visual_rel[1] / self.cfg.vision_range,
                ],
                dtype=np.float32,
            )

        sound_rel = np.zeros(2, dtype=np.float32)
        if unit.local_sound:
            sound_rel = np.array(
                [
                    unit.local_sound_rel[0] / self.cfg.hearing_range,
                    unit.local_sound_rel[1] / self.cfg.hearing_range,
                ],
                dtype=np.float32,
            )

        has_memory = unit.last_seen_pos is not None
        mem_rel = np.zeros(2, dtype=np.float32)
        mem_age = 1.0
        if has_memory:
            d = unit.last_seen_pos - unit.pos
            mem_rel = np.array(
                [d[0]/self.cfg.width, d[1]/self.cfg.height],
                dtype=np.float32,
            )
            mem_age = min(1.0, unit.last_seen_age / self.cfg.radio_memory_seconds)

        bb_rel = np.zeros(2, dtype=np.float32)
        if self.blackboard.contact_pos is not None:
            d = self.blackboard.contact_pos - unit.pos
            bb_rel = np.array(
                [d[0]/self.cfg.width, d[1]/self.cfg.height],
                dtype=np.float32,
            )

        feature = self._feature_for(unit)
        feature_type = 0.0
        feature_rel = np.zeros(2, dtype=np.float32)
        feature_cleared = 0.0
        if feature is not None:
            feature_type = float(feature.feature) / float(MapFeature.COVER_POINT)
            p = np.array(feature.position, dtype=np.float32)
            d = p - unit.pos
            feature_rel = np.array(
                [d[0]/self.cfg.width, d[1]/self.cfg.height],
                dtype=np.float32,
            )
            feature_cleared = (
                1.0
                if feature.feature_id in self.blackboard.cleared_feature_ids
                else 0.0
            )

        near_types = set()
        for point in self.map.semantic_points:
            p = np.array(point.position, dtype=np.float32)
            if float(np.linalg.norm(p - unit.pos)) <= 1.6:
                near_types.add(point.feature)

        alive = sum(1 for u in self.units.values() if u.alive)
        contact_count = sum(
            1 for u in self.units.values() if u.local_visual
        )
        utility_target = self._legitimate_tactical_target(unit) is not None

        obs = np.array(
            [
                unit.pos[0] / w * 2 - 1,
                unit.pos[1] / h * 2 - 1,
                math.cos(unit.angle),
                math.sin(unit.angle),
                *role_one_hot,
                min(1.0, unit.profile.armor / 2.5),
                unit.ammo / max(unit.max_ammo, 1),
                unit.smoke_count / max(unit.profile.smoke_count, 1),
                unit.flash_count / max(unit.profile.flash_count, 1),
                unit.health,
                1.0 if unit.local_visual else 0.0,
                unit.local_visual_confidence,
                local_vis_rel[0], local_vis_rel[1],
                np.clip(unit.local_visible_velocity[0] / 1.0, -1, 1),
                np.clip(unit.local_visible_velocity[1] / 1.0, -1, 1),
                1.0 if unit.local_sound else 0.0,
                sound_rel[0], sound_rel[1],
                unit.local_sound_confidence,
                1.0 if has_memory else 0.0,
                mem_rel[0], mem_rel[1],
                mem_age,
                1.0 if self.blackboard.contact_pos is not None else 0.0,
                bb_rel[0], bb_rel[1],
                min(1.0, self.blackboard.contact_age / self.cfg.radio_memory_seconds),
                float(self.blackboard.alert) / float(FacilityAlert.CONFIRMED),
                1.0 if self.blackboard.theft_known else 0.0,
                1.0 if self._compliant_contact_known(unit) else 0.0,
                1.0 if any(
                    float(np.linalg.norm(
                        unit.pos - np.asarray(c.center, dtype=np.float32)
                    )) <= c.radius for c in self.smoke_clouds
                ) else 0.0,
                min(1.0, unit.blinded_timer / 3.0),
                nrel[0], nrel[1], nrole,
                center_rel[0], center_rel[1],
                lrel[0], lrel[1],
                srel[0], srel[1],
                feature_type,
                feature_rel[0], feature_rel[1],
                feature_cleared,
                1.0 if MapFeature.DOOR in near_types else 0.0,
                1.0 if MapFeature.CORNER in near_types else 0.0,
                1.0 if MapFeature.INTERSECTION in near_types else 0.0,
                1.0 if MapFeature.COVER_POINT in near_types else 0.0,
                self._formation_error(uid),
                alive / 5.0,
                contact_count / 5.0,
                1.0 if self.blackboard.contact_age <= 2.0 else 0.0,
                1.0 if unit.profile.has_shield else 0.0,
                1.0 if (
                    unit.ammo / max(unit.max_ammo, 1)
                    <= self.cfg.reload_fraction_threshold
                ) else 0.0,
                1.0 if MapFeature.LONG_CORRIDOR in near_types else 0.0,
                1.0 if utility_target else 0.0,
                float(unit.assigned_sector) / 3.0,
                min(1.0, len(self.blackboard.cleared_feature_ids) / 12.0),
            ],
            dtype=np.float32,
        )
        assert obs.shape == (64,), obs.shape
        return np.clip(obs, -1.0, 1.0)

    def info(self, uid: str) -> dict:
        unit = self.units[uid]
        return {
            "agent": uid,
            "role": unit.role.name,
            "scenario": self.scenario,
            "local_visual": bool(unit.local_visual),
            "local_sound": bool(unit.local_sound),
            "blackboard_contact": self.blackboard.contact_pos is not None,
            "blackboard_contact_age": float(self.blackboard.contact_age),
            "radio_updates": int(self.blackboard.radio_updates),
            "cleared_features": len(self.blackboard.cleared_feature_ids),
            "smoke_clouds": len(self.smoke_clouds),
            "player_captured": bool(self.player_captured),
            "player_neutralized": bool(self.player_neutralized),
            "player_health": float(self.player_health),
            "player_escaped": bool(self.player_escaped),
            "world_ticks": int(self.world_ticks),
            # Deliberately no hidden player position.
        }

    def terminated(self) -> bool:
        return (
            self.player_captured
            or self.player_neutralized
            or self.player_escaped
        )

    def truncated(self) -> bool:
        return self.world_ticks >= self.cfg.max_world_ticks


class SwatSharedPolicyEnv(gym.Env[np.ndarray, int]):
    """
    SB3-compatible shared-policy multi-agent environment.

    One PPO policy is reused for all five physical units. The active agent
    rotates SHIELD -> ASSAULT -> UTILITY -> HEAVY -> LEADER. Each action is
    executed immediately; after all five decisions, the world/player advances.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 5}

    def __init__(
        self,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        scenario: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.core = SwatTeamCore(seed=seed)
        self.forced_scenario = scenario
        self.action_space = spaces.Discrete(len(SwatAction))
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(64,),
            dtype=np.float32,
        )
        self.active_index = 0
        self.active_uid = self.core.agent_ids[0]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scenario = self.forced_scenario
        if options and options.get("scenario") is not None:
            scenario = str(options["scenario"])
        self.core.reset_core(seed=seed, scenario=scenario)
        self.active_index = 0
        self.active_uid = self.core.agent_ids[0]
        return self.core.observation(self.active_uid), self.core.info(self.active_uid)

    def step(self, action: int):
        uid = self.active_uid
        action_enum = SwatAction(int(action))
        reward = self.core.apply_action(uid, action_enum)

        self.active_index += 1
        world_advanced = False
        if self.active_index >= len(self.core.agent_ids):
            self.active_index = 0
            reward += self.core.advance_world()
            world_advanced = True

        self.active_uid = self.core.agent_ids[self.active_index]
        terminated = self.core.terminated()
        truncated = self.core.truncated()
        info = self.core.info(self.active_uid)
        info["acted_agent"] = uid
        info["action_name"] = action_enum.name
        info["world_advanced"] = world_advanced
        return (
            self.core.observation(self.active_uid),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self):
        if self.render_mode != "ansi":
            return None
        rows = []
        unit_cells = {
            self.core.map.cell(unit.pos): unit.role.name[0]
            for unit in self.core.units.values()
            if unit.alive
        }
        for y in range(self.core.cfg.height):
            row = []
            for x in range(self.core.cfg.width):
                ch = "#" if self.core.map.grid[y, x] else "."
                if (x, y) in unit_cells:
                    ch = unit_cells[(x, y)]
                if (
                    any(u.local_visual for u in self.core.units.values())
                    and (x, y) == self.core.map.cell(self.core.player_pos)
                ):
                    ch = "P"
                for cloud in self.core.smoke_clouds:
                    if (
                        float(np.linalg.norm(
                            np.array([x,y], dtype=np.float32)
                            - np.asarray(cloud.center, dtype=np.float32)
                        )) <= cloud.radius
                        and ch == "."
                    ):
                        ch = "~"
                row.append(ch)
            rows.append("".join(row))
        return "\n".join(rows)
