from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .doctrine import FacilityAlert, VisualBand
from .env import (
    SensePacket,
    ShadowProtocolEnv,
    TacticalAction,
)


@dataclass(frozen=True)
class DoctrineSkill:
    name: str
    expected_actions: tuple[TacticalAction, ...]


SKILLS: tuple[DoctrineSkill, ...] = (
    DoctrineSkill("routine", (TacticalAction.PATROL,)),
    DoctrineSkill("peripheral_hint", (TacticalAction.FOCUS_VISUAL,)),
    DoctrineSkill("attention", (TacticalAction.FOCUS_VISUAL,)),
    DoctrineSkill(
        "attention_repeat3",
        (TacticalAction.CALL_REINFORCEMENTS,),
    ),
    DoctrineSkill(
        "suspicious_visual",
        (TacticalAction.INVESTIGATE_VISUAL,),
    ),
    DoctrineSkill(
        "suspicious_known",
        (TacticalAction.REQUEST_LOCAL_SUPPORT,),
    ),
    DoctrineSkill(
        "probable_visual",
        (TacticalAction.INVESTIGATE_VISUAL,),
    ),
    DoctrineSkill(
        "probable_known",
        (TacticalAction.RADIO_REPORT,),
    ),
    DoctrineSkill("sound", (TacticalAction.INVESTIGATE_SOUND,)),
    DoctrineSkill(
        "repeat_sound",
        (TacticalAction.REQUEST_LOCAL_SUPPORT,),
    ),
    DoctrineSkill(
        "visual_unreported",
        (TacticalAction.RADIO_REPORT,),
    ),
    DoctrineSkill("visual_reported", (TacticalAction.PURSUE,)),
    DoctrineSkill("arrest", (TacticalAction.ARREST,)),
    DoctrineSkill(
        "last_seen",
        (
            TacticalAction.INTERCEPT,
            TacticalAction.GO_LAST_SEEN,
        ),
    ),
    DoctrineSkill("search", (TacticalAction.SEARCH_LAST_SEEN,)),
    DoctrineSkill(
        "theft_pre_call",
        (TacticalAction.CALL_REINFORCEMENTS,),
    ),
    DoctrineSkill(
        "theft_post_call",
        (TacticalAction.PROTECT_NEAREST_INTEL,),
    ),
)

SKILL_BY_NAME = {skill.name: skill for skill in SKILLS}

# Backward-compatible micro-skill aliases. V6.2 used "visual" for a
# confirmed contact. V6.3 split it into visual_unreported -> RADIO_REPORT
# and visual_reported -> PURSUE. The old "visual" name means the latter.
SKILL_ALIASES = {
    "visual": "visual_reported",
}


def _canonical_skill_name(name: str) -> str:
    return SKILL_ALIASES.get(name, name)


class DoctrinePretrainEnv(ShadowProtocolEnv):
    """
    One-decision curriculum used only for doctrine pretraining.

    It shares exactly the same action and observation spaces as
    ShadowProtocolEnv. Each episode presents one legitimate tactical knowledge
    state and ends after the policy chooses its tactical intention.

    No hidden live player coordinates are inserted into the observation. For
    visual skills, the policy receives only a synthetic SensePacket equivalent
    to what the deterministic perception system would have produced.
    """

    def __init__(
        self,
        *args,
        forced_skill: Optional[str] = None,
        doctrine_correct_bonus: float = 1.00,
        doctrine_wrong_penalty: float = 0.60,
        robust_context: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if forced_skill is not None:
            forced_skill = _canonical_skill_name(str(forced_skill))
            if forced_skill not in SKILL_BY_NAME:
                raise ValueError(
                    f"Unknown doctrine skill: {forced_skill}"
                )
        self.forced_skill = forced_skill
        self.doctrine_correct_bonus = doctrine_correct_bonus
        self.doctrine_wrong_penalty = doctrine_wrong_penalty
        self.robust_context = bool(robust_context)
        self.doctrine_skill = "routine"
        self.expected_actions: tuple[TacticalAction, ...] = (
            TacticalAction.PATROL,
        )

    @property
    def skill_names(self) -> tuple[str, ...]:
        return tuple(skill.name for skill in SKILLS)

    def _choose_skill(self, options: Optional[dict]) -> str:
        if options is not None:
            requested = options.get("skill")
            if requested is not None:
                requested = _canonical_skill_name(str(requested))
                if requested not in SKILL_BY_NAME:
                    raise ValueError(
                        f"Unknown doctrine skill: {requested}"
                    )
                return requested
        if self.forced_skill is not None:
            return self.forced_skill
        return str(self.rng.choice(self.skill_names))

    def _set_clear_perception(self) -> None:
        self.sense = SensePacket()
        self.visual_confidence = 0.0
        self.visual_attention_episode_active = False
        self.visual_anomaly_pos = None
        self.visual_anomaly_age = 999.0
        self.last_visual_rate = 0.0
        self.last_visual_band = VisualBand.NONE
        self.focus_visual_timer = 0.0
        self.flashlight_active = False

    def _set_visual_packet(
        self,
        confidence: float,
        *,
        peripheral: bool = False,
        identified: bool = False,
        distance: float = 5.0,
        motion_salience: float = 0.45,
    ) -> None:
        band = (
            VisualBand.PERIPHERAL
            if peripheral
            else (
                VisualBand.OUTER
                if distance > self.cfg.vision_mid
                else VisualBand.MID
            )
        )
        rel = (float(distance), 0.0)
        self.visual_confidence = float(confidence)
        self.last_visual_band = band
        self.last_visual_rate = 0.35
        self.visual_anomaly_pos = (
            self.guard_pos + np.array(rel, dtype=np.float32)
        )
        self.visual_anomaly_age = 0.0
        self.sense = SensePacket(
            visual=bool(identified),
            visual_candidate=True,
            visual_confidence=float(confidence),
            visual_rel=rel,
            visual_velocity=(0.55, 0.0),
            visual_band=int(band),
            peripheral=bool(peripheral),
            motion_salience=float(motion_salience),
            spot_rate=0.35,
            sound=False,
        )

    def _set_sound_packet(
        self,
        confidence: float = 0.75,
        *,
        repeat_count: int = 0,
    ) -> None:
        self.audio_repeat_count = int(repeat_count)
        self.sense = SensePacket(
            sound=True,
            sound_rel=(-3.5, 0.8),
            sound_confidence=float(confidence),
        )
        self.last_sound_pos = (
            self.guard_pos
            + np.array(self.sense.sound_rel, dtype=np.float32)
        )
        self.time_since_last_sound = 0.0

    def _ensure_buddy_nearby(self) -> None:
        self.buddy_available = True
        self.buddy_supporting = False
        self.buddy_pos = self.guard_pos + np.array(
            [2.0, 0.5],
            dtype=np.float32,
        )
        x, y = self._cell(self.buddy_pos)
        if not self._free(x, y):
            self.buddy_pos = self.guard_pos.copy()

    def _inject_memory_distractor(self) -> None:
        """Add stale legitimate memory without changing active visual priority."""
        self.last_seen_pos = self.guard_pos + np.array(
            [
                float(self.rng.uniform(2.0, 6.0)),
                float(self.rng.uniform(-2.5, 2.5)),
            ],
            dtype=np.float32,
        )
        self.last_seen_velocity = np.array(
            [
                float(self.rng.uniform(-0.8, 0.8)),
                float(self.rng.uniform(-0.8, 0.8)),
            ],
            dtype=np.float32,
        )
        self.last_seen_age = float(self.rng.uniform(0.5, 4.0))

    def _inject_theft_distractor(self) -> None:
        stolen = int(self.rng.integers(0, len(self.known_intel_present)))
        self.known_intel_present[stolen] = 0

    def _inject_robust_context(self, name: str) -> None:
        """
        Randomize nuisance context while preserving the intended tactical phase.

        This bridges the gap between clean micro-states and the real world,
        where visual contact may coexist with stale memory, a known theft,
        buddy state, or an already-called reinforcement team.
        """
        active_visual = name in {
            "peripheral_hint",
            "attention",
            "attention_repeat3",
            "suspicious_visual",
            "suspicious_known",
            "probable_visual",
            "probable_known",
            "visual_unreported",
            "visual_reported",
            "arrest",
        }

        memory_phase = name in {"last_seen", "search"}
        active_sound = name in {"sound", "repeat_sound"}

        # Theft is a lower-priority fact and may coexist with visual/audio/
        # memory states. Do not inject it into pure routine or theft drills.
        if (
            name not in {"routine", "theft_pre_call", "theft_post_call"}
            and self.rng.random() < 0.55
        ):
            self._inject_theft_distractor()

        # Stale remembered contact is a common distractor during a new visual
        # detection. It must never override the live sensory phase.
        if active_visual and self.rng.random() < 0.65:
            self._inject_memory_distractor()

        # Generic operational context that should not decide the current
        # tactical priority by itself.
        if name not in {
            "attention_repeat3",
            "theft_pre_call",
        }:
            self.reinforcements_called = bool(
                self.rng.random() < 0.45
            )

        if name not in {
            "suspicious_known",
            "repeat_sound",
        }:
            self.buddy_available = bool(self.rng.random() < 0.70)
            if self.buddy_available:
                self.buddy_pos = self.guard_pos + np.array(
                    [2.0, 0.5],
                    dtype=np.float32,
                )
                self.buddy_supporting = bool(
                    self.rng.random() < 0.45
                )

        # Preserve deliberate radio semantics for the two visual-confirmed
        # skills and for probable_known.
        if name == "visual_unreported":
            self.radio_reported = False
        elif name == "visual_reported":
            self.radio_reported = True
        elif name == "probable_known":
            self.radio_reported = False
        elif name == "arrest":
            # Arrest outranks radio state; train both combinations.
            self.radio_reported = bool(self.rng.random() < 0.50)
        elif name in {"last_seen", "search"}:
            self.radio_reported = bool(self.rng.random() < 0.65)

        # Repeat-sound drill specifically requires a nearby buddy who has not
        # started supporting yet.
        if name == "repeat_sound":
            self._ensure_buddy_nearby()
            self.buddy_supporting = False

        # Suspicious-known drill also tests explicit support request.
        if name == "suspicious_known":
            self._ensure_buddy_nearby()
            self.buddy_supporting = False

        # attention_repeat3 is explicitly "three strikes -> call". A previous
        # reinforcement call would make that action meaningless.
        if name == "attention_repeat3":
            self.reinforcements_called = False

        # theft contract is explicit.
        if name == "theft_pre_call":
            self.reinforcements_called = False
        elif name == "theft_post_call":
            self.reinforcements_called = True

    def _configure_skill(self, name: str) -> None:
        # Start from a clean epistemic state.
        self._set_clear_perception()
        self.alert = int(FacilityAlert.CALM)
        self.intruder_known = False
        self.time_since_confirmed_intruder = 999.0
        self.personal_suspicion = 0.0
        self.visual_exposure_count = 0
        self.audio_repeat_count = 0
        self.last_seen_pos = None
        self.last_seen_velocity[:] = 0.0
        self.last_seen_age = self.cfg.memory_seconds + 1.0
        self.known_intel_present[:] = 1
        self.reinforcements_called = False
        self.radio_reported = False
        self.buddy_supporting = False
        self.search_anchor = None
        self.search_waypoints = []
        self.search_index = 0
        self.search_waypoints_visited = 0
        self.player_compliant = False

        if name == "routine":
            pass

        elif name == "peripheral_hint":
            self._set_visual_packet(
                float(self.rng.uniform(0.06, 0.23)),
                peripheral=True,
                distance=float(self.rng.uniform(5.0, 8.5)),
                motion_salience=float(self.rng.uniform(0.15, 0.75)),
            )

        elif name == "attention":
            self._set_visual_packet(
                float(self.rng.uniform(0.26, 0.47)),
                distance=float(self.rng.uniform(5.0, 10.0)),
            )
            self.personal_suspicion = float(self.rng.uniform(0.10, 0.38))
            self.visual_exposure_count = int(self.rng.integers(0, 2))

        elif name == "attention_repeat3":
            self.alert = int(FacilityAlert.SUSPICIOUS)
            self._set_visual_packet(
                float(self.rng.uniform(0.26, 0.45)),
                distance=float(self.rng.uniform(5.0, 9.0)),
            )
            self.personal_suspicion = float(self.rng.uniform(0.66, 0.90))
            self.visual_exposure_count = int(self.rng.integers(3, 6))
            self._ensure_buddy_nearby()

        elif name == "suspicious_visual":
            self._set_visual_packet(
                float(self.rng.uniform(0.51, 0.73)),
                distance=float(self.rng.uniform(4.0, 8.0)),
            )
            self.personal_suspicion = float(self.rng.uniform(0.30, 0.58))

        elif name == "suspicious_known":
            self.alert = int(FacilityAlert.SEARCHING)
            self.intruder_known = True
            self.time_since_confirmed_intruder = float(
                self.rng.uniform(1.0, 12.0)
            )
            self._set_visual_packet(
                float(self.rng.uniform(0.51, 0.73)),
                distance=float(self.rng.uniform(4.0, 8.0)),
            )
            self.personal_suspicion = float(self.rng.uniform(0.65, 0.92))
            self._ensure_buddy_nearby()

        elif name == "probable_visual":
            self._set_visual_packet(
                float(self.rng.uniform(0.76, 0.96)),
                distance=float(self.rng.uniform(3.0, 7.0)),
            )
            self.personal_suspicion = float(self.rng.uniform(0.50, 0.75))

        elif name == "probable_known":
            self.alert = int(FacilityAlert.SEARCHING)
            self.intruder_known = True
            self.time_since_confirmed_intruder = float(
                self.rng.uniform(0.5, 10.0)
            )
            self._set_visual_packet(
                float(self.rng.uniform(0.76, 0.96)),
                distance=float(self.rng.uniform(3.0, 7.0)),
            )
            self.personal_suspicion = float(self.rng.uniform(0.75, 1.0))
            self._ensure_buddy_nearby()

        elif name == "sound":
            self._set_sound_packet(
                float(self.rng.uniform(0.55, 0.90)),
                repeat_count=0,
            )

        elif name == "repeat_sound":
            self.alert = int(FacilityAlert.SUSPICIOUS)
            self.personal_suspicion = float(self.rng.uniform(0.55, 0.88))
            self._set_sound_packet(
                float(self.rng.uniform(0.60, 0.95)),
                repeat_count=int(self.rng.integers(2, 5)),
            )
            self._ensure_buddy_nearby()

        elif name == "visual_unreported":
            self.alert = int(FacilityAlert.CONFIRMED)
            self.intruder_known = True
            self.radio_reported = False
            self.time_since_confirmed_intruder = 0.0
            self._set_visual_packet(
                1.0,
                identified=True,
                distance=float(self.rng.uniform(2.0, 6.0)),
                motion_salience=0.85,
            )

        elif name == "visual_reported":
            self.alert = int(FacilityAlert.CONFIRMED)
            self.intruder_known = True
            self.radio_reported = True
            self.time_since_confirmed_intruder = 0.0
            self._set_visual_packet(
                1.0,
                identified=True,
                distance=float(self.rng.uniform(2.0, 6.0)),
                motion_salience=0.85,
            )

        elif name == "arrest":
            self.alert = int(FacilityAlert.CONFIRMED)
            self.intruder_known = True
            self.time_since_confirmed_intruder = 0.0
            self.player_compliant = True
            self._set_visual_packet(
                1.0,
                identified=True,
                distance=0.9,
                motion_salience=0.0,
            )

        elif name == "last_seen":
            self.alert = int(FacilityAlert.SEARCHING)
            self.intruder_known = True
            self.time_since_confirmed_intruder = float(
                self.rng.uniform(1.0, 8.0)
            )
            self.last_seen_pos = self.guard_pos + np.array(
                [float(self.rng.uniform(2.5, 5.5)), 0.0],
                dtype=np.float32,
            )
            self.last_seen_velocity = np.array(
                [float(self.rng.uniform(0.2, 1.0)), 0.0],
                dtype=np.float32,
            )
            self.last_seen_age = float(self.rng.uniform(0.5, 3.0))

        elif name == "search":
            self.alert = int(FacilityAlert.SEARCHING)
            self.intruder_known = True
            self.last_seen_pos = self.guard_pos + np.array(
                [0.8, 0.0],
                dtype=np.float32,
            )
            self.last_seen_velocity = np.array([0.5, 0.0], dtype=np.float32)
            self.last_seen_age = float(self.rng.uniform(1.0, 4.0))
            self._ensure_search_pattern()

        elif name == "theft_pre_call":
            self.alert = int(FacilityAlert.SUSPICIOUS)
            self.personal_suspicion = 0.75
            stolen = int(self.rng.integers(0, 3))
            self.known_intel_present[stolen] = 0
            self.reinforcements_called = False

        elif name == "theft_post_call":
            self.alert = int(FacilityAlert.SUSPICIOUS)
            self.personal_suspicion = 0.82
            stolen = int(self.rng.integers(0, 3))
            self.known_intel_present[stolen] = 0
            self.reinforcements_called = True
            self.reinforcement_timer = float(
                self.rng.uniform(0.2, self.cfg.reinforcement_delay_seconds)
            )

        else:
            raise ValueError(f"Unhandled doctrine skill: {name}")

    def reset(self, *, seed=None, options=None):
        # Base reset initializes all physical, memory and support state.
        obs, info = super().reset(
            seed=seed,
            options={"scenario": "routine"},
        )
        self.steps = 0

        self.doctrine_skill = self._choose_skill(options)
        self.expected_actions = SKILL_BY_NAME[
            self.doctrine_skill
        ].expected_actions
        self.training_scenario = f"doctrine:{self.doctrine_skill}"
        self._configure_skill(self.doctrine_skill)
        if self.robust_context:
            self._inject_robust_context(self.doctrine_skill)

        obs = self._observation()
        info = self._info()
        info["doctrine_skill"] = self.doctrine_skill
        info["expected_actions"] = [
            action.name for action in self.expected_actions
        ]
        info["doctrine_correct"] = None
        return obs, info

    def step(self, action: int):
        """
        Contextual-bandit doctrine step.

        We intentionally do NOT advance hidden world physics here. Pretraining
        teaches only tactical intent selection from legitimate knowledge. Full
        ShadowProtocolEnv fine-tuning teaches consequences and sequences.
        """
        action_enum = TacticalAction(int(action))
        self.steps += 1

        decision = self._decision_context()
        phase = self._tactical_phase(decision)
        previous_guard = self.guard_pos.copy()

        base_reward = self._reward(
            action_enum,
            previous_guard,
            decision,
        )
        correct = action_enum in self.expected_actions

        reward = base_reward
        reward += (
            self.doctrine_correct_bonus
            if correct
            else -self.doctrine_wrong_penalty
        )

        self.last_action = action_enum

        info = self._info()
        info["action_name"] = action_enum.name
        info["terminal_reward"] = 0.0
        info["decision_phase"] = phase
        info["result_phase"] = phase
        info["doctrine_skill"] = self.doctrine_skill
        info["expected_actions"] = [
            expected.name for expected in self.expected_actions
        ]
        info["doctrine_correct"] = bool(correct)
        info["base_reward"] = float(base_reward)

        # Exactly one doctrinal decision per episode.
        terminated = True
        truncated = False
        return self._observation(), float(reward), terminated, truncated, info
