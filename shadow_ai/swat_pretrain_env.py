
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .doctrine import SwatRole
from .facility_map import MapFeature
from .swat import SwatAction
from .swat_env import SwatTeamCore


@dataclass(frozen=True)
class SwatSkill:
    name: str
    role: SwatRole
    expected: tuple[SwatAction, ...]


SWAT_SKILLS = (
    SwatSkill("shield_lead", SwatRole.SHIELD, (SwatAction.ADVANCE,)),
    SwatSkill("assault_follow", SwatRole.ASSAULT, (SwatAction.FOLLOW_SHIELD,)),
    SwatSkill("cover_mate", SwatRole.HEAVY, (SwatAction.COVER_TEAMMATE,)),
    SwatSkill("check_corner", SwatRole.ASSAULT, (SwatAction.CHECK_CORNER,)),
    SwatSkill("clear_sector", SwatRole.LEADER, (SwatAction.CLEAR_SECTOR,)),
    SwatSkill("watch_entry", SwatRole.HEAVY, (SwatAction.WATCH_ENTRY,)),
    SwatSkill("regroup", SwatRole.LEADER, (SwatAction.REGROUP,)),
    SwatSkill("utility_flash", SwatRole.UTILITY, (SwatAction.USE_FLASH,)),
    SwatSkill("utility_smoke", SwatRole.UTILITY, (SwatAction.USE_SMOKE,)),
    SwatSkill("reload", SwatRole.ASSAULT, (SwatAction.RELOAD,)),
    SwatSkill("visual_pursue", SwatRole.ASSAULT, (SwatAction.PURSUE,)),
    SwatSkill("reported_contact_pursue", SwatRole.ASSAULT, (SwatAction.PURSUE,)),
    SwatSkill("compliant_arrest", SwatRole.LEADER, (SwatAction.ARREST,)),
    SwatSkill("reported_compliant_arrest", SwatRole.LEADER, (SwatAction.ARREST,)),
)
SKILL_BY_NAME = {s.name: s for s in SWAT_SKILLS}


class SwatDoctrinePretrainEnv(gym.Env[np.ndarray, int]):
    """
    One-decision role-conditioned SWAT doctrine curriculum.
    Observation/action spaces are identical to SwatSharedPolicyEnv.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        forced_skill: Optional[str] = None,
        robust_context: bool = True,
    ) -> None:
        super().__init__()
        if forced_skill is not None and forced_skill not in SKILL_BY_NAME:
            raise ValueError(forced_skill)
        self.rng = np.random.default_rng(seed)
        self.forced_skill = forced_skill
        self.robust_context = robust_context
        self.core = SwatTeamCore(seed=seed)
        self.action_space = spaces.Discrete(len(SwatAction))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(64,), dtype=np.float32
        )
        self.skill = SWAT_SKILLS[0]
        self.uid = ""

    def _uid_for_role(self, role: SwatRole) -> str:
        for uid, unit in self.core.units.items():
            if unit.role == role:
                return uid
        raise RuntimeError(role)

    def _configure(self) -> None:
        self.core.reset_core(seed=int(self.rng.integers(0, 2**31-1)))
        self.uid = self._uid_for_role(self.skill.role)
        unit = self.core.units[self.uid]

        # Remove accidental local contact first.
        for u in self.core.units.values():
            u.local_visual = False
            u.local_sound = False

        if self.skill.name == "shield_lead":
            self.core.blackboard.contact_pos = np.array([20.0, 8.0], dtype=np.float32)

        elif self.skill.name == "assault_follow":
            shield = self.core._role_unit(SwatRole.SHIELD)
            shield.pos = unit.pos + np.array([-3.0, 0.0], dtype=np.float32)
            shield.angle = 0.0
            self.core.blackboard.contact_pos = np.array([18.0, 8.0], dtype=np.float32)

        elif self.skill.name == "cover_mate":
            mate = self.core._role_unit(SwatRole.SHIELD)
            mate.pos = unit.pos + np.array([-1.5, 0.5], dtype=np.float32)

        elif self.skill.name == "check_corner":
            corner = self.core.map.nearest_feature(
                unit.pos, allowed={MapFeature.CORNER}
            )
            unit.assigned_feature_id = None if corner is None else corner.feature_id

        elif self.skill.name == "clear_sector":
            feature = self.core.map.nearest_feature(
                unit.pos,
                allowed={
                    MapFeature.ROOM_ENTRY,
                    MapFeature.INTERSECTION,
                },
            )
            unit.assigned_feature_id = None if feature is None else feature.feature_id

        elif self.skill.name == "watch_entry":
            door = self.core.map.nearest_feature(
                unit.pos, allowed={MapFeature.DOOR}
            )
            unit.assigned_feature_id = None if door is None else door.feature_id

        elif self.skill.name == "regroup":
            unit.pos = np.array([6.0, 16.0], dtype=np.float32)
            for other in self.core.units.values():
                if other.uid != self.uid:
                    other.pos = np.array([24.0, 3.0], dtype=np.float32)

        elif self.skill.name == "utility_flash":
            # Fresh, confirmed local contact -> flash opportunity.
            self.core.player_pos = unit.pos + np.array([-2.1, 0.0], dtype=np.float32)
            unit.angle = np.pi
            unit.flash_count = max(1, unit.flash_count)
            self.core._sense_unit(unit)
            unit.local_visual = True
            unit.local_visual_confidence = 1.0
            unit.local_visual_rel = self.core.player_pos - unit.pos
            self.core.blackboard.contact_pos = self.core.player_pos.copy()
            self.core.blackboard.contact_age = 0.2
            self.core.blackboard.contact_reported = True

        elif self.skill.name == "utility_smoke":
            # Stale reported contact + corridor crossing -> smoke opportunity.
            corridor = self.core.map.nearest_feature(
                unit.pos,
                allowed={MapFeature.LONG_CORRIDOR},
            )
            if corridor is not None:
                unit.pos = np.array(corridor.position, dtype=np.float32)
                unit.assigned_feature_id = corridor.feature_id
            self.core.blackboard.contact_pos = unit.pos + np.array(
                [-5.0, 0.0],
                dtype=np.float32,
            )
            self.core.blackboard.contact_age = 4.5
            self.core.blackboard.contact_reported = True
            unit.local_visual = False
            unit.smoke_count = max(1, unit.smoke_count)

        elif self.skill.name == "reload":
            unit.ammo = 1

        elif self.skill.name == "visual_pursue":
            self.core.player_pos = unit.pos + np.array([-3.0, 0.0], dtype=np.float32)
            unit.angle = np.pi
            self.core._sense_unit(unit)
            if not unit.local_visual:
                unit.local_visual = True
                unit.local_visual_confidence = 1.0
                unit.local_visual_rel = self.core.player_pos - unit.pos

        elif self.skill.name == "reported_contact_pursue":
            unit.local_visual = False
            self.core.blackboard.contact_pos = unit.pos + np.array(
                [-4.0, 0.5],
                dtype=np.float32,
            )
            self.core.blackboard.contact_age = 0.7
            self.core.blackboard.contact_reported = True
            self.core.blackboard.player_compliant_known = False

        elif self.skill.name == "compliant_arrest":
            self.core.player_compliant = True
            self.core.player_pos = unit.pos + np.array([-0.7, 0.0], dtype=np.float32)
            unit.angle = np.pi
            self.core._sense_unit(unit)
            unit.local_visual = True
            unit.local_visual_confidence = 1.0
            unit.local_visual_rel = self.core.player_pos - unit.pos

        elif self.skill.name == "reported_compliant_arrest":
            self.core.player_compliant = True
            self.core.player_pos = unit.pos + np.array([-3.2, 0.0], dtype=np.float32)
            unit.local_visual = False
            self.core.blackboard.contact_pos = self.core.player_pos.copy()
            self.core.blackboard.contact_age = 0.4
            self.core.blackboard.contact_reported = True
            self.core.blackboard.player_compliant_known = True

        if self.robust_context:
            # Legitimate lower-priority distractors.
            self.core.blackboard.theft_known = bool(self.rng.random() < 0.55)
            self.core.blackboard.contact_reported = bool(self.rng.random() < 0.7)
            if self.core.blackboard.contact_pos is None and self.rng.random() < 0.5:
                self.core.blackboard.contact_pos = np.array(
                    [15.0, 10.0], dtype=np.float32
                )
                self.core.blackboard.contact_age = float(
                    self.rng.uniform(2.0, 8.0)
                )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        requested = (
            None if options is None else options.get("skill")
        )
        name = str(requested) if requested is not None else self.forced_skill
        if name is None:
            self.skill = SWAT_SKILLS[int(self.rng.integers(0, len(SWAT_SKILLS)))]
        else:
            self.skill = SKILL_BY_NAME[name]
        self._configure()
        info = self.core.info(self.uid)
        info["skill"] = self.skill.name
        info["expected_actions"] = [a.name for a in self.skill.expected]
        return self.core.observation(self.uid), info

    def step(self, action: int):
        action_enum = SwatAction(int(action))
        base = self.core.apply_action(self.uid, action_enum)
        correct = action_enum in self.skill.expected
        reward = base + (1.0 if correct else -0.65)
        info = self.core.info(self.uid)
        info["skill"] = self.skill.name
        info["expected_actions"] = [a.name for a in self.skill.expected]
        info["doctrine_correct"] = bool(correct)
        info["action_name"] = action_enum.name
        return self.core.observation(self.uid), float(reward), True, False, info
