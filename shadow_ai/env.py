from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import heapq
import math
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .doctrine import (
    FacilityAlert,
    NORMAL_GUARD_PROFILE,
    PlayerStance,
    VisualBand,
)


class TacticalAction(IntEnum):
    HOLD = 0
    PATROL = 1
    INVESTIGATE_SOUND = 2
    GO_LAST_SEEN = 3
    INTERCEPT = 4
    SEARCH_LAST_SEEN = 5
    PROTECT_NEAREST_INTEL = 6
    FLANK_LEFT = 7
    FLANK_RIGHT = 8
    PURSUE = 9
    ARREST = 10
    CALL_REINFORCEMENTS = 11
    FOCUS_VISUAL = 12
    INVESTIGATE_VISUAL = 13
    REQUEST_LOCAL_SUPPORT = 14
    RADIO_REPORT = 15



@dataclass
class Config:
    width: int = 32
    height: int = 20
    max_steps: int = 900

    guard_speed: float = 0.34
    buddy_speed: float = 0.46
    player_walk_speed: float = 0.40
    player_sprint_speed: float = 0.62
    player_stealth_speed: float = 0.26

    # V6 layered perception. OUTER is intentionally much longer than V5.
    vision_immediate: float = 1.35
    vision_close: float = 3.0
    vision_mid: float = 6.5
    vision_outer: float = 12.0
    main_fov_degrees: float = 105.0
    peripheral_fov_degrees: float = 175.0

    # Base confidence gained per second before multipliers.
    spot_rate_close: float = 1.45
    spot_rate_mid: float = 0.72
    spot_rate_outer: float = 0.22
    spot_rate_peripheral: float = 0.16
    confidence_decay_per_second: float = 0.20

    focus_day_multiplier: float = 1.75
    focus_night_flashlight_multiplier: float = 2.40
    known_intruder_multiplier: float = 1.65
    repeated_exposure_step: float = 0.38
    repeated_exposure_cap: float = 2.70

    hearing_walk: float = 4.0
    hearing_sprint: float = 7.5
    hearing_stealth: float = 1.4
    sound_repeat_window_seconds: float = 8.0

    memory_seconds: float = 10.0
    known_intruder_recent_seconds: float = 28.0
    dt: float = 0.20

    capture_distance: float = 0.72
    arrest_distance: float = 1.15
    escape_distance: float = 0.75

    search_radius: float = 2.2
    search_reach_distance: float = 0.72
    reinforcement_delay_seconds: float = 2.0
    reinforcement_speed: float = 0.52
    reinforcement_vision_range: float = 6.5
    reinforcement_capture_distance: float = 0.52
    reinforcement_count: int = 2

    local_support_radius: float = 7.0
    local_support_probability: float = 0.72

    # routine, peripheral, attention, repeat_visual, sound, repeat_sound,
    # visual, lost_contact, search, arrest, theft
    curriculum_weights: tuple[
        float, float, float, float, float, float,
        float, float, float, float, float,
    ] = (
        0.12, 0.08, 0.09, 0.09, 0.09, 0.09,
        0.10, 0.09, 0.09, 0.08, 0.08,
    )


@dataclass
class SensePacket:
    # visual=True means 100% identification. visual_candidate can be partial.
    visual: bool = False
    visual_candidate: bool = False
    visual_confidence: float = 0.0
    visual_rel: tuple[float, float] = (0.0, 0.0)
    visual_velocity: tuple[float, float] = (0.0, 0.0)
    visual_band: int = int(VisualBand.NONE)
    peripheral: bool = False
    motion_salience: float = 0.0
    spot_rate: float = 0.0

    sound: bool = False
    sound_rel: tuple[float, float] = (0.0, 0.0)
    sound_confidence: float = 0.0


def _unit(vx: float, vy: float) -> tuple[float, float]:
    mag = math.hypot(vx, vy)
    if mag <= 1e-8:
        return 0.0, 0.0
    return vx / mag, vy / mag


def _clip_vec(vx: float, vy: float, max_mag: float = 1.0) -> tuple[float, float]:
    mag = math.hypot(vx, vy)
    if mag <= max_mag or mag <= 1e-8:
        return vx, vy
    return vx / mag * max_mag, vy / mag * max_mag


class ShadowProtocolEnv(gym.Env[np.ndarray, int]):
    """
    Single-guard tactical RL environment.

    The policy NEVER receives the hidden live player coordinates unless the
    visual sensor succeeds. When sight is lost it receives only:
      * the last visually confirmed position,
      * its age,
      * visually observed velocity,
      * noisy sound cues,
      * guard-local knowledge of intel.

    Low-level movement/pathfinding remains deterministic A*.
    The learned policy chooses tactical intent.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 5}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.cfg = config or Config()
        self._seed = seed

        self.action_space = spaces.Discrete(len(TacticalAction))

        # Observation layout (all float32):
        #  0,1  guard x/y normalized
        #  2,3  facing x/y
        #  4     current alert [0..1]
        #  5     visual contact
        #  6,7  relative visible-player x/y (zero if not visible)
        #  8,9  visually observed player velocity (zero if no visual sample)
        # 10     sound cue present
        # 11,12 noisy relative sound x/y
        # 13     sound confidence
        # 14     has last-seen memory
        # 15,16 relative last-seen x/y
        # 17     memory age normalized
        # 18,19 last observed velocity kept in memory
        # 20..22 local belief: intel present? (1 present / 0 confirmed stolen)
        # 23     reinforcements already called
        # 24     player currently compliant IF visible (else 0)
        # 25     distance to nearest known-present intel normalized
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(43,),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(seed)
        self._build_map()
        self._patrol_points = [
            np.array([3.5, 3.0], dtype=np.float32),
            np.array([10.0, 3.2], dtype=np.float32),
            np.array([17.0, 4.0], dtype=np.float32),
            np.array([26.0, 3.0], dtype=np.float32),
            np.array([27.0, 15.5], dtype=np.float32),
            np.array([18.0, 16.5], dtype=np.float32),
            np.array([9.0, 15.5], dtype=np.float32),
            np.array([3.0, 13.0], dtype=np.float32),
        ]
        self._intel = [
            np.array([4.0, 4.0], dtype=np.float32),
            np.array([15.5, 10.0], dtype=np.float32),
            np.array([27.5, 16.0], dtype=np.float32),
        ]
        self._exit = np.array([30.2, 18.0], dtype=np.float32)

        self.training_scenario = "routine"
        self.episode_intel_order = [0, 1, 2]
        self.episode_goal_mode = "all_intel"
        self.episode_has_intel = np.ones(3, dtype=np.int8)

    # ------------------------------------------------------------------
    # Map + deterministic navigation
    # ------------------------------------------------------------------
    def _build_map(self) -> None:
        w, h = self.cfg.width, self.cfg.height
        self.grid = np.zeros((h, w), dtype=np.uint8)

        # Sparse training arena, analogous to the browser prototype.
        self.grid[2:10, 7] = 1
        self.grid[0:6, 13] = 1
        self.grid[9:17, 13] = 1
        self.grid[3:12, 20] = 1
        self.grid[0:7, 26] = 1
        self.grid[10:20, 26] = 1
        self.grid[12, 4:12] = 1
        self.grid[12, 15:23] = 1

        # Wide openings / doors.
        for y, x in [(6, 7), (7, 7), (6, 13), (7, 13), (8, 20), (9, 20), (7, 26), (8, 26)]:
            if 0 <= y < h and 0 <= x < w:
                self.grid[y, x] = 0

    def _cell(self, pos: np.ndarray) -> tuple[int, int]:
        x = int(np.clip(round(float(pos[0])), 0, self.cfg.width - 1))
        y = int(np.clip(round(float(pos[1])), 0, self.cfg.height - 1))
        return x, y

    def _free(self, x: int, y: int) -> bool:
        return (
            0 <= x < self.cfg.width
            and 0 <= y < self.cfg.height
            and self.grid[y, x] == 0
        )

    def _line_clear(self, a: np.ndarray, b: np.ndarray) -> bool:
        distance = float(np.linalg.norm(b - a))
        samples = max(2, int(distance * 5.0))
        for t in np.linspace(0.0, 1.0, samples, endpoint=True)[1:-1]:
            p = a + (b - a) * t
            x, y = self._cell(p)
            if not self._free(x, y):
                return False
        return True

    def _astar(self, start: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
        sx, sy = self._cell(start)
        gx, gy = self._cell(goal)
        if not self._free(gx, gy):
            gx, gy = self._nearest_free_cell(gx, gy)

        if self._line_clear(start, np.array([gx, gy], dtype=np.float32)):
            return [np.array([gx, gy], dtype=np.float32)]

        start_key = (sx, sy)
        goal_key = (gx, gy)

        frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start_key)]
        came_from: dict[tuple[int, int], Optional[tuple[int, int]]] = {start_key: None}
        cost_so_far: dict[tuple[int, int], float] = {start_key: 0.0}

        neighbors = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414),
        ]

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_key:
                break

            for dx, dy, step_cost in neighbors:
                nx, ny = current[0] + dx, current[1] + dy
                if not self._free(nx, ny):
                    continue

                if dx != 0 and dy != 0:
                    if not self._free(current[0] + dx, current[1]):
                        continue
                    if not self._free(current[0], current[1] + dy):
                        continue

                nxt = (nx, ny)
                new_cost = cost_so_far[current] + step_cost
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + math.hypot(nx - gx, ny - gy)
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current

        if goal_key not in came_from:
            return [goal.copy()]

        rev: list[tuple[int, int]] = []
        cur: Optional[tuple[int, int]] = goal_key
        while cur is not None:
            rev.append(cur)
            cur = came_from[cur]
        rev.reverse()
        return [np.array([x, y], dtype=np.float32) for x, y in rev[1:]]

    def _nearest_free_cell(self, gx: int, gy: int) -> tuple[int, int]:
        if self._free(gx, gy):
            return gx, gy
        for radius in range(1, 6):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    x, y = gx + dx, gy + dy
                    if self._free(x, y):
                        return x, y
        return 1, 1

    def _move_towards(
        self,
        pos: np.ndarray,
        goal: Optional[np.ndarray],
        speed: float,
    ) -> np.ndarray:
        if goal is None:
            return pos

        path = self._astar(pos, goal)
        target = path[0] if path else goal
        direction = target - pos
        d = float(np.linalg.norm(direction))
        if d <= 1e-6:
            return pos

        step = min(speed, d)
        candidate = pos + direction / d * step
        x, y = self._cell(candidate)
        if self._free(x, y):
            return candidate.astype(np.float32)
        return pos

    # ------------------------------------------------------------------
    # Player bot
    # ------------------------------------------------------------------
    def _reset_player_bot(self) -> None:
        self.player_style = self.rng.choice(
            ["stealth", "runner", "deceiver", "mixed"],
            p=[0.30, 0.25, 0.20, 0.25],
        )

        # Randomize which intel exist and in what order they matter.
        count = int(self.rng.integers(0, 4))
        ids = [0, 1, 2]
        self.rng.shuffle(ids)
        self.episode_intel_order = ids[:count]
        self.episode_has_intel = np.zeros(3, dtype=np.int8)
        for idx in self.episode_intel_order:
            self.episode_has_intel[idx] = 1

        self.player_goal_idx = 0
        self.player_has_intel = np.zeros(3, dtype=np.int8)
        self.player_diversion_cooldown = 0.0
        self.player_smoke_cooldown = 0.0
        self.player_current_speed = 0.0
        self.player_compliant = False
        self.player_stance = (
            PlayerStance.CROUCHED
            if self.player_style == "stealth"
            else PlayerStance.STANDING
        )

        if count == 0:
            self.episode_goal_mode = "wander_then_exit"
        elif count == 1:
            self.episode_goal_mode = "single_intel"
        else:
            self.episode_goal_mode = "all_intel"

    def _player_step(self) -> None:
        # Scenario-specific player behavior is used to make rare tactical
        # situations appear often enough for PPO to learn them.

        scenario = self.training_scenario

        # ARREST curriculum: visible, nearby, frequently compliant.
        if scenario == "arrest":
            distance_to_guard = float(np.linalg.norm(self.player_pos - self.guard_pos))
            if distance_to_guard <= self.cfg.arrest_distance * 2.3:
                self.player_compliant = bool(self.rng.random() < 0.78)
                if self.player_compliant:
                    self.player_velocity[:] = 0.0
                    self.player_current_speed = 0.0
                    return

        # LOST-CONTACT curriculum: first create contact, then route behind cover.
        if scenario in {"peripheral", "attention", "repeat_visual"}:
            # Keep these curricula near their perception geometry instead of
            # immediately running to an intel objective.
            goal = self.player_pos.copy()
            if scenario == "peripheral" and self.player_style == "runner":
                goal = self.player_pos + np.array([0.0, 3.0], dtype=np.float32)
        elif scenario in {"sound", "repeat_sound"}:
            goal = self.player_pos + np.array([0.0, 4.0], dtype=np.float32)
        elif scenario == "lost_contact":
            if self.steps < 4:
                goal = np.array([19.0, 5.0], dtype=np.float32)
            elif self.steps < 34:
                # x=20 wall blocks the direct view at this y.
                goal = np.array([23.5, 5.0], dtype=np.float32)
            else:
                goal = np.array([27.0, 14.5], dtype=np.float32)
        elif scenario == "search":
            # Search curriculum keeps the live player away from the remembered
            # area. The player itself is irrelevant until legitimately sensed.
            goal = np.array([29.0, 18.0], dtype=np.float32)
        else:
            # Normal randomized objective selection.
            if self.player_goal_idx < len(self.episode_intel_order):
                intel_id = self.episode_intel_order[self.player_goal_idx]
                goal = self._intel[intel_id]
            else:
                # In routine curriculum we don't always rush the objective.
                if scenario == "routine" and self.steps < 60:
                    wander_points = [
                        np.array([2.5, 16.5], dtype=np.float32),
                        np.array([6.0, 17.0], dtype=np.float32),
                        np.array([4.0, 13.0], dtype=np.float32),
                    ]
                    goal = wander_points[(self.steps // 20) % len(wander_points)]
                else:
                    goal = self._exit

        distance_to_guard = float(np.linalg.norm(self.player_pos - self.guard_pos))
        sees_danger = self.sense.visual

        if scenario in {"sound", "repeat_sound"}:
            # Force enough noisy motion to generate acoustic examples.
            speed = self.cfg.player_sprint_speed
        elif scenario == "visual":
            speed = self.cfg.player_sprint_speed
        elif self.player_style == "stealth":
            speed = self.cfg.player_stealth_speed
        elif self.player_style == "runner":
            speed = self.cfg.player_sprint_speed
        elif self.player_style == "deceiver":
            speed = self.cfg.player_walk_speed
        else:
            speed = (
                self.cfg.player_sprint_speed
                if sees_danger or distance_to_guard < 4.0
                else self.cfg.player_walk_speed
            )

        # Posture is part of the physical world. It affects the deterministic
        # perception model, but is never directly handed to the policy unless
        # the visual sensor produces enough evidence.
        if speed >= self.cfg.player_sprint_speed * 0.95:
            self.player_stance = PlayerStance.STANDING
        elif self.player_style == "stealth":
            self.player_stance = PlayerStance.CROUCHED
        elif self.player_style == "deceiver" and self.rng.random() < 0.45:
            self.player_stance = PlayerStance.CROUCHED
        else:
            self.player_stance = PlayerStance.STANDING

        # General compliance outside the dedicated arrest scenario is rare.
        # Search/chase curricula deliberately disable accidental surrender so
        # that a lost-contact episode cannot collapse into an arrest episode.
        if scenario != "arrest":
            if scenario in {"visual", "sound", "repeat_sound", "peripheral", "attention", "repeat_visual", "lost_contact", "search"}:
                self.player_compliant = False
            elif (
                self.sense.visual
                and distance_to_guard <= self.cfg.arrest_distance * 1.8
            ):
                self.player_compliant = bool(self.rng.random() < 0.08)
            else:
                self.player_compliant = False

        if self.player_compliant:
            self.player_velocity[:] = 0.0
            self.player_current_speed = 0.0
            return

        old = self.player_pos.copy()
        self.player_pos = self._move_towards(self.player_pos, goal, speed)
        self.player_velocity = (self.player_pos - old) / self.cfg.dt
        self.player_current_speed = float(np.linalg.norm(self.player_velocity))

        # Pick up only episode-enabled intel.
        if self.player_goal_idx < len(self.episode_intel_order):
            intel_id = self.episode_intel_order[self.player_goal_idx]
            current_goal = self._intel[intel_id]
            if np.linalg.norm(self.player_pos - current_goal) < 0.8:
                self.player_has_intel[intel_id] = 1
                self.player_goal_idx += 1

    # ------------------------------------------------------------------
    # Sensors: the ONLY functions allowed to sample hidden player truth.
    # ------------------------------------------------------------------
    def _visual_band_for(self, distance: float, peripheral: bool) -> VisualBand:
        if distance <= self.cfg.vision_immediate:
            return VisualBand.IMMEDIATE
        if peripheral:
            return VisualBand.PERIPHERAL
        if distance <= self.cfg.vision_close:
            return VisualBand.CLOSE
        if distance <= self.cfg.vision_mid:
            return VisualBand.MID
        if distance <= self.cfg.vision_outer:
            return VisualBand.OUTER
        return VisualBand.NONE

    def _spot_rate(
        self,
        distance: float,
        peripheral: bool,
        motion_salience: float,
    ) -> float:
        band = self._visual_band_for(distance, peripheral)
        if band == VisualBand.IMMEDIATE:
            return 99.0
        if band == VisualBand.CLOSE:
            base = self.cfg.spot_rate_close
        elif band == VisualBand.MID:
            base = self.cfg.spot_rate_mid
        elif band == VisualBand.OUTER:
            base = self.cfg.spot_rate_outer
        elif band == VisualBand.PERIPHERAL:
            base = self.cfg.spot_rate_peripheral
        else:
            return 0.0

        # Standing is easier to resolve; crouched slow movement is deliberately
        # difficult, especially in peripheral vision.
        posture = (
            1.00
            if self.player_stance == PlayerStance.STANDING
            else 0.52
        )

        if peripheral:
            movement = 0.32 + 2.25 * motion_salience
        else:
            movement = 0.62 + 1.25 * motion_salience

        alert_mult = 1.0 + 0.16 * int(self.alert)
        if self.intruder_known:
            alert_mult *= self.cfg.known_intruder_multiplier

        repetition = min(
            self.cfg.repeated_exposure_cap,
            1.0 + self.cfg.repeated_exposure_step
            * self.visual_exposure_count,
        )

        focus = 1.0
        if self.focus_visual_timer > 0.0:
            focus = (
                self.cfg.focus_night_flashlight_multiplier
                if self.is_night
                else self.cfg.focus_day_multiplier
            )
            focus *= NORMAL_GUARD_PROFILE.focus_multiplier

        return float(base * posture * movement * alert_mult * repetition * focus)

    def _sense(self) -> SensePacket:
        """The only lead-guard sensor allowed to sample hidden player truth."""
        packet = SensePacket()

        relative = self.player_pos - self.guard_pos
        distance = float(np.linalg.norm(relative))
        facing = np.array(
            [math.cos(self.guard_angle), math.sin(self.guard_angle)],
            dtype=np.float32,
        )

        line_clear = (
            distance <= self.cfg.vision_outer
            and self._line_clear(self.guard_pos, self.player_pos)
        )

        candidate = False
        peripheral = False
        dot = -1.0
        if line_clear:
            if distance <= 1e-6:
                dot = 1.0
            else:
                dot = float(np.dot(facing, relative / distance))

            main_cos = math.cos(
                math.radians(self.cfg.main_fov_degrees / 2.0)
            )
            peripheral_cos = math.cos(
                math.radians(self.cfg.peripheral_fov_degrees / 2.0)
            )
            if dot >= main_cos or distance <= self.cfg.vision_immediate:
                candidate = True
            elif dot >= peripheral_cos:
                candidate = True
                peripheral = True

        # Velocity is in map-units / second. Normalize against sprint speed/sec.
        sprint_per_second = self.cfg.player_sprint_speed / self.cfg.dt
        motion_salience = float(
            np.clip(
                self.player_current_speed / max(sprint_per_second, 1e-6),
                0.0,
                1.0,
            )
        )

        if candidate:
            band = self._visual_band_for(distance, peripheral)
            rate = self._spot_rate(distance, peripheral, motion_salience)

            # Immediate zone = deterministic identification.
            if band == VisualBand.IMMEDIATE:
                self.visual_confidence = 1.0
            else:
                # When an already-known intruder produces a genuine new visual
                # stimulus, guards start verification from a suspicious 50%,
                # but receive NO hidden coordinates without the stimulus itself.
                if self.intruder_known and self.visual_confidence < 0.50:
                    self.visual_confidence = 0.50
                self.visual_confidence = float(
                    np.clip(
                        self.visual_confidence + rate * self.cfg.dt,
                        0.0,
                        1.0,
                    )
                )

            # New attention episode when crossing 25% after having fallen below.
            if (
                self.visual_confidence >= 0.25
                and not self.visual_attention_episode_active
            ):
                self.visual_attention_episode_active = True
                self.visual_exposure_count += 1
                self.personal_suspicion = float(
                    np.clip(self.personal_suspicion + 0.08, 0.0, 1.0)
                )

            # Partial observations are deliberately noisy. Certainty removes noise.
            sigma = 1.25 * (1.0 - self.visual_confidence)
            noisy = (
                self.player_pos
                if self.visual_confidence >= 0.999
                else self.player_pos
                + self.rng.normal(0.0, sigma, size=2).astype(np.float32)
            )
            rel = noisy - self.guard_pos

            packet.visual_candidate = True
            packet.visual = self.visual_confidence >= 0.999
            packet.visual_confidence = self.visual_confidence
            packet.visual_rel = (float(rel[0]), float(rel[1]))
            packet.visual_band = int(band)
            packet.peripheral = peripheral
            packet.motion_salience = motion_salience
            packet.spot_rate = rate

            # Reliable motion vector only after strong visual confidence.
            if self.visual_confidence >= 0.75:
                packet.visual_velocity = (
                    float(self.player_velocity[0]),
                    float(self.player_velocity[1]),
                )

            self.visual_anomaly_pos = noisy.astype(np.float32)
            self.visual_anomaly_age = 0.0
            self.last_visual_rate = rate
            self.last_visual_band = band
        else:
            self.visual_confidence = float(
                max(
                    0.0,
                    self.visual_confidence
                    - self.cfg.confidence_decay_per_second * self.cfg.dt,
                )
            )
            self.visual_anomaly_age += self.cfg.dt
            self.last_visual_rate = 0.0
            self.last_visual_band = VisualBand.NONE

            if self.visual_confidence < 0.10:
                self.visual_attention_episode_active = False

        # Hearing: hidden truth becomes a noisy cue. Repetition is tracked per
        # distinct acoustic episode, not every frame of the same continuous sound.
        hearing_radius = self.cfg.hearing_walk
        if self.player_current_speed <= self.cfg.player_stealth_speed / self.cfg.dt * 1.25:
            hearing_radius = self.cfg.hearing_stealth
        elif self.player_current_speed >= self.cfg.player_sprint_speed / self.cfg.dt * 0.70:
            hearing_radius = self.cfg.hearing_sprint

        sound_now = (
            distance <= hearing_radius
            and self.player_current_speed > 0.05
        )

        if sound_now:
            confidence = float(
                np.clip(
                    1.0 - distance / max(hearing_radius, 1e-6),
                    0.0,
                    1.0,
                )
            )
            noise_sigma = 1.35 * (1.0 - confidence) + 0.15
            noisy = self.player_pos + self.rng.normal(
                0.0,
                noise_sigma,
                size=2,
            ).astype(np.float32)
            cue = noisy - self.guard_pos
            packet.sound = True
            packet.sound_rel = (float(cue[0]), float(cue[1]))
            packet.sound_confidence = confidence

            if not self.sound_episode_active:
                self.sound_episode_active = True
                if self.time_since_last_sound <= self.cfg.sound_repeat_window_seconds:
                    self.audio_repeat_count += 1
                else:
                    self.audio_repeat_count = 1
                self.personal_suspicion = float(
                    np.clip(
                        self.personal_suspicion
                        + 0.04
                        + 0.025 * min(self.audio_repeat_count, 4),
                        0.0,
                        1.0,
                    )
                )
            self.time_since_last_sound = 0.0
            self.last_sound_pos = noisy.astype(np.float32)
        else:
            self.time_since_last_sound += self.cfg.dt
            self.sound_episode_active = False

        return packet

    # ------------------------------------------------------------------
    # Knowledge + observation
    # ------------------------------------------------------------------
    def _update_knowledge(self, inspect_intel: bool = True) -> None:
        # 100% confidence creates confirmed visual knowledge.
        if self.sense.visual:
            visible_pos = (
                self.guard_pos
                + np.array(self.sense.visual_rel, dtype=np.float32)
            )
            old_memory = (
                None if self.last_seen_pos is None else self.last_seen_pos.copy()
            )
            self.last_seen_pos = visible_pos
            self.last_seen_velocity = np.array(
                self.sense.visual_velocity,
                dtype=np.float32,
            )
            self.last_seen_age = 0.0
            self.intruder_known = True
            self.time_since_confirmed_intruder = 0.0
            self.alert = int(FacilityAlert.CONFIRMED)
            self.personal_suspicion = 1.0

            if (
                old_memory is None
                or np.linalg.norm(old_memory - visible_pos) > 0.75
            ):
                self._clear_search_pattern()
        else:
            self.last_seen_age += self.cfg.dt
            if self.intruder_known:
                self.time_since_confirmed_intruder += self.cfg.dt
                # Lost contact means searching, not omniscient pursuit.
                if self.alert >= int(FacilityAlert.CONFIRMED):
                    self.alert = int(FacilityAlert.SEARCHING)

        # Thresholds below 100% change suspicion, not confirmed coordinates.
        if self.sense.visual_candidate:
            if self.sense.visual_confidence >= 0.75:
                self.personal_suspicion = max(self.personal_suspicion, 0.78)
                self.alert = max(self.alert, int(FacilityAlert.SUSPICIOUS))
            elif self.sense.visual_confidence >= 0.50:
                self.personal_suspicion = max(self.personal_suspicion, 0.55)
            elif self.sense.visual_confidence >= 0.25:
                self.personal_suspicion = max(self.personal_suspicion, 0.30)

        # Intel belief changes only after physical inspection.
        for idx, pedestal in enumerate(self._intel):
            if inspect_intel and np.linalg.norm(self.guard_pos - pedestal) <= 1.15:
                if self.player_has_intel[idx]:
                    self.known_intel_present[idx] = 0
                    self.alert = max(
                        self.alert,
                        int(FacilityAlert.SUSPICIOUS),
                    )
                    self.personal_suspicion = max(
                        self.personal_suspicion,
                        0.68,
                    )

        if self.last_seen_age > self.cfg.memory_seconds:
            self.last_seen_pos = None
            self.last_seen_velocity[:] = 0.0
            self._clear_search_pattern()

        # "Intruder known recently" eventually expires if the facility never
        # reconfirms anything. Alert remains elevated if other evidence exists.
        if (
            self.intruder_known
            and self.time_since_confirmed_intruder
            > self.cfg.known_intruder_recent_seconds
        ):
            self.intruder_known = False
            if self.alert >= int(FacilityAlert.SEARCHING):
                self.alert = int(FacilityAlert.SUSPICIOUS)

        # Focus lasts only briefly unless the policy chooses it again.
        self.focus_visual_timer = max(
            0.0,
            self.focus_visual_timer - self.cfg.dt,
        )
        self.flashlight_active = bool(
            self.is_night and self.focus_visual_timer > 0.0
        )

        # Personal suspicion decays very slowly only in genuinely calm time.
        if (
            not self.sense.visual_candidate
            and not self.sense.sound
            and not self.intruder_known
        ):
            self.personal_suspicion = max(
                0.0,
                self.personal_suspicion - 0.012 * self.cfg.dt,
            )
    def _observation(self) -> np.ndarray:
        w = float(self.cfg.width - 1)
        h = float(self.cfg.height - 1)
        gx = float(self.guard_pos[0] / w * 2.0 - 1.0)
        gy = float(self.guard_pos[1] / h * 2.0 - 1.0)
        facing = (math.cos(self.guard_angle), math.sin(self.guard_angle))

        visual_rel = (
            np.array(self.sense.visual_rel, dtype=np.float32)
            if self.sense.visual_candidate
            else np.zeros(2, dtype=np.float32)
        )
        visual_rel /= max(1.0, self.cfg.vision_outer)

        visible_vel = (
            np.array(self.sense.visual_velocity, dtype=np.float32)
            if self.sense.visual_confidence >= 0.75
            else np.zeros(2, dtype=np.float32)
        )
        visible_vel = np.array(
            _clip_vec(
                float(visible_vel[0]),
                float(visible_vel[1]),
                3.0,
            ),
            dtype=np.float32,
        ) / 3.0

        sound_rel = (
            np.array(self.sense.sound_rel, dtype=np.float32)
            if self.sense.sound
            else np.zeros(2, dtype=np.float32)
        )
        sound_rel /= max(1.0, self.cfg.hearing_sprint)

        has_memory = self.last_seen_pos is not None
        if has_memory:
            mem_rel = self.last_seen_pos - self.guard_pos
            mem_rel /= max(1.0, self.cfg.vision_outer)
            mem_age = float(
                np.clip(
                    self.last_seen_age / self.cfg.memory_seconds,
                    0.0,
                    1.0,
                )
            )
            mem_vel = np.array(
                _clip_vec(
                    float(self.last_seen_velocity[0]),
                    float(self.last_seen_velocity[1]),
                    3.0,
                ),
                dtype=np.float32,
            ) / 3.0
        else:
            mem_rel = np.zeros(2, dtype=np.float32)
            mem_age = 1.0
            mem_vel = np.zeros(2, dtype=np.float32)

        known_present = self.known_intel_present.astype(np.float32)
        known_positions = [
            p
            for i, p in enumerate(self._intel)
            if self.known_intel_present[i] == 1
        ]
        if known_positions:
            nearest = min(
                float(np.linalg.norm(self.guard_pos - p))
                for p in known_positions
            )
            nearest_intel = 1.0 - float(
                np.clip(
                    nearest / math.hypot(self.cfg.width, self.cfg.height),
                    0.0,
                    1.0,
                )
            )
        else:
            nearest_intel = 0.0

        buddy_distance = 1.0
        if self.buddy_available and self.buddy_pos is not None:
            buddy_distance = float(
                np.clip(
                    np.linalg.norm(self.buddy_pos - self.guard_pos)
                    / self.cfg.local_support_radius,
                    0.0,
                    1.0,
                )
            )

        band_norm = float(self.sense.visual_band) / float(VisualBand.IMMEDIATE)
        repeats_norm = min(1.0, self.visual_exposure_count / 4.0)
        audio_repeats_norm = min(1.0, self.audio_repeat_count / 4.0)
        support_count = 1.0 if (
            self.buddy_available
            and self.buddy_pos is not None
            and np.linalg.norm(self.buddy_pos - self.guard_pos)
            <= self.cfg.local_support_radius
        ) else 0.0

        obs = np.array(
            [
                # V5-compatible core, updated for confidence semantics.
                gx, gy,                                      # 0..1
                facing[0], facing[1],                        # 2..3
                self.alert / float(FacilityAlert.CONFIRMED), # 4
                1.0 if self.sense.visual else 0.0,           # 5
                visual_rel[0], visual_rel[1],                # 6..7
                visible_vel[0], visible_vel[1],              # 8..9
                1.0 if self.sense.sound else 0.0,            # 10
                sound_rel[0], sound_rel[1],                  # 11..12
                self.sense.sound_confidence,                 # 13
                1.0 if has_memory else 0.0,                  # 14
                mem_rel[0], mem_rel[1],                      # 15..16
                mem_age,                                     # 17
                mem_vel[0], mem_vel[1],                      # 18..19
                known_present[0], known_present[1], known_present[2], #20..22
                1.0 if self.reinforcements_called else 0.0,  # 23
                1.0 if (self.sense.visual and self.player_compliant) else 0.0, #24
                nearest_intel,                               # 25

                # V6 perception / psychology / social context.
                self.sense.visual_confidence,                # 26
                band_norm,                                   # 27
                1.0 if self.sense.peripheral else 0.0,       # 28
                1.0 if self.sense.visual_candidate else 0.0, # 29
                self.sense.motion_salience,                  # 30
                float(np.clip(self.sense.spot_rate / 4.0, 0.0, 1.0)), #31
                self.personal_suspicion,                     # 32
                repeats_norm,                                # 33
                audio_repeats_norm,                          # 34
                1.0 if self.intruder_known else 0.0,         # 35
                1.0 if self.focus_visual_timer > 0.0 else 0.0, #36
                1.0 if self.flashlight_active else 0.0,      # 37
                1.0 if self.is_night else 0.0,               # 38
                support_count,                               # 39
                1.0 if self.buddy_supporting else 0.0,       # 40
                1.0 if self.radio_reported else 0.0,         # 41
                1.0 - buddy_distance,                        # 42
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    # ------------------------------------------------------------------
    # V5: search sectors
    # ------------------------------------------------------------------
    def _clear_search_pattern(self) -> None:
        self.search_anchor = None
        self.search_waypoints = []
        self.search_index = 0
        self.search_waypoints_visited = 0

    def _ensure_search_pattern(self) -> None:
        if self.last_seen_pos is None:
            self._clear_search_pattern()
            return

        if (
            self.search_anchor is not None
            and np.linalg.norm(self.search_anchor - self.last_seen_pos) < 0.55
            and self.search_waypoints
        ):
            return

        self.search_anchor = self.last_seen_pos.copy()
        r = self.cfg.search_radius

        offsets = [
            (r, 0.0),
            (0.7 * r, 0.7 * r),
            (0.0, r),
            (-0.7 * r, 0.7 * r),
            (-r, 0.0),
            (-0.7 * r, -0.7 * r),
            (0.0, -r),
            (0.7 * r, -0.7 * r),
        ]

        waypoints: list[np.ndarray] = []
        for ox, oy in offsets:
            raw = self.search_anchor + np.array(
                [ox, oy],
                dtype=np.float32,
            )
            x, y = self._cell(raw)
            if not self._free(x, y):
                x, y = self._nearest_free_cell(x, y)
            p = np.array([x, y], dtype=np.float32)

            if all(np.linalg.norm(p - q) > 0.55 for q in waypoints):
                waypoints.append(p)

        self.search_waypoints = waypoints
        self.search_index = 0
        self.search_waypoints_visited = 0

    def _current_search_target(self) -> Optional[np.ndarray]:
        self._ensure_search_pattern()
        if not self.search_waypoints:
            return None
        idx = min(self.search_index, len(self.search_waypoints) - 1)
        return self.search_waypoints[idx].copy()

    def _advance_search_pattern(self) -> None:
        if not self.search_waypoints:
            return

        target = self.search_waypoints[
            min(self.search_index, len(self.search_waypoints) - 1)
        ]
        if (
            np.linalg.norm(self.guard_pos - target)
            > self.cfg.search_reach_distance
        ):
            return

        self.search_waypoints_visited += 1
        self.search_index += 1

        if self.search_index >= len(self.search_waypoints):
            # A complete sweep is a real environmental event. If nobody was
            # found, discard stale target memory and fall back to elevated
            # security / theft protection / routine.
            self.search_completed_event = True
            self.last_seen_pos = None
            self.last_seen_velocity[:] = 0.0
            self.last_seen_age = self.cfg.memory_seconds + 1.0
            self.search_index = 0
            self.search_waypoints = []
            self.search_anchor = None

            if np.any(self.known_intel_present == 0):
                self.alert = max(self.alert, 1)
            else:
                self.alert = min(self.alert, 1)

    # ------------------------------------------------------------------
    # V5: physical reinforcement team
    # ------------------------------------------------------------------
    def _spawn_reinforcements(self) -> None:
        access_points = [
            np.array([30.0, 1.0], dtype=np.float32),
            np.array([30.0, 18.0], dtype=np.float32),
            np.array([1.0, 18.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
        ]

        self.reinforcement_positions = []
        self.reinforcement_angles = []

        for raw in access_points[: self.cfg.reinforcement_count]:
            x, y = self._cell(raw)
            if not self._free(x, y):
                x, y = self._nearest_free_cell(x, y)

            pos = np.array([x, y], dtype=np.float32)
            self.reinforcement_positions.append(pos)
            self.reinforcement_angles.append(
                math.atan2(
                    self.guard_pos[1] - pos[1],
                    self.guard_pos[0] - pos[0],
                )
            )

        self.reinforcements_active = True
        self.reinforcements_spawned_event = True

    def _reinforcement_can_see_player(self, idx: int) -> bool:
        if idx >= len(self.reinforcement_positions):
            return False

        pos = self.reinforcement_positions[idx]
        relative = self.player_pos - pos
        distance = float(np.linalg.norm(relative))

        if distance > self.cfg.reinforcement_vision_range:
            return False
        if not self._line_clear(pos, self.player_pos):
            return False

        if distance < 1.2:
            return True

        facing = np.array(
            [
                math.cos(self.reinforcement_angles[idx]),
                math.sin(self.reinforcement_angles[idx]),
            ],
            dtype=np.float32,
        )
        dot = float(np.dot(facing, relative / max(distance, 1e-6)))
        cos_half = math.cos(
            math.radians(self.cfg.main_fov_degrees / 2.0)
        )
        return dot >= cos_half

    def _reinforcement_target(self, idx: int) -> np.ndarray:
        # Radio knowledge is allowed only if it came from real observation.
        if self.last_seen_pos is not None:
            return self.last_seen_pos.copy()

        # After a confirmed theft, split the team across remaining objectives.
        known = [
            p
            for i, p in enumerate(self._intel)
            if self.known_intel_present[i] == 1
        ]
        if known:
            return known[idx % len(known)].copy()

        # Otherwise clear toward the lead guard / facility center.
        return self.guard_pos.copy()

    def _step_reinforcements(self) -> None:
        self.reinforcements_spawned_event = False
        self.reinforcement_capture_event = False
        self.reinforcement_radio_contact = False

        if not self.reinforcements_called:
            return

        if not self.reinforcements_active:
            self.reinforcement_timer += self.cfg.dt
            if (
                self.reinforcement_timer
                >= self.cfg.reinforcement_delay_seconds
            ):
                self._spawn_reinforcements()
            else:
                return

        for idx in range(len(self.reinforcement_positions)):
            pos = self.reinforcement_positions[idx]

            # Local, legitimate perception.
            can_see_before = self._reinforcement_can_see_player(idx)
            if can_see_before:
                self.last_seen_pos = self.player_pos.copy()
                self.last_seen_velocity = self.player_velocity.copy()
                self.last_seen_age = 0.0
                self.alert = max(self.alert, 2)
                self.reinforcement_radio_contact = True

            target = (
                self.player_pos.copy()
                if can_see_before
                else self._reinforcement_target(idx)
            )

            new_pos = self._move_towards(
                pos,
                target,
                self.cfg.reinforcement_speed,
            )
            delta = target - new_pos
            if float(np.linalg.norm(delta)) > 1e-6:
                self.reinforcement_angles[idx] = math.atan2(
                    float(delta[1]),
                    float(delta[0]),
                )
            self.reinforcement_positions[idx] = new_pos

            # Re-check after moving. A helper capture is therefore based on
            # actual local line-of-sight, never hidden omniscience.
            can_see_after = self._reinforcement_can_see_player(idx)
            if can_see_after:
                self.last_seen_pos = self.player_pos.copy()
                self.last_seen_velocity = self.player_velocity.copy()
                self.last_seen_age = 0.0
                self.alert = max(self.alert, 2)
                self.reinforcement_radio_contact = True

                if (
                    not self.player_compliant
                    and np.linalg.norm(
                        self.reinforcement_positions[idx]
                        - self.player_pos
                    )
                    <= self.cfg.reinforcement_capture_distance
                ):
                    self.reinforcement_capture_event = True

    # ------------------------------------------------------------------
    # V6 local social support
    # ------------------------------------------------------------------
    def _best_suspicious_target(self) -> Optional[np.ndarray]:
        if self.sense.visual_candidate:
            return (
                self.guard_pos
                + np.array(self.sense.visual_rel, dtype=np.float32)
            )
        if self.visual_anomaly_pos is not None:
            return self.visual_anomaly_pos.copy()
        if self.sense.sound:
            return (
                self.guard_pos
                + np.array(self.sense.sound_rel, dtype=np.float32)
            )
        if self.last_sound_pos is not None:
            return self.last_sound_pos.copy()
        if self.last_seen_pos is not None:
            return self.last_seen_pos.copy()
        return None

    def _request_local_support(self) -> None:
        if not self.buddy_available or self.buddy_pos is None:
            return
        target = self._best_suspicious_target()
        if target is None:
            return
        self.buddy_supporting = True
        self.buddy_target = target.copy()

    def _step_buddy(self) -> None:
        if not self.buddy_available or self.buddy_pos is None:
            return

        # Implicit social reading: a nearby buddy notices a suspicious colleague
        # focusing/investigating during elevated context and starts supporting.
        if (
            not self.buddy_supporting
            and np.linalg.norm(self.buddy_pos - self.guard_pos)
            <= self.cfg.local_support_radius
            and self.personal_suspicion >= 0.45
            and self.last_action in {
                TacticalAction.FOCUS_VISUAL,
                TacticalAction.INVESTIGATE_VISUAL,
                TacticalAction.INVESTIGATE_SOUND,
                TacticalAction.PURSUE,
            }
        ):
            target = self._best_suspicious_target()
            if target is not None:
                self.buddy_supporting = True
                self.buddy_target = target.copy()

        if not self.buddy_supporting or self.buddy_target is None:
            return

        self.buddy_pos = self._move_towards(
            self.buddy_pos,
            self.buddy_target,
            self.cfg.buddy_speed,
        )

        # Once near the suspicious area, the buddy provides local cover rather
        # than magically discovering the player.
        if np.linalg.norm(self.buddy_pos - self.buddy_target) <= 1.1:
            self.buddy_target = self.guard_pos.copy()

    # ------------------------------------------------------------------
    # Tactical action execution
    # ------------------------------------------------------------------
    def _choose_target(self, action: TacticalAction) -> Optional[np.ndarray]:
        if action == TacticalAction.HOLD:
            return None

        if action == TacticalAction.PATROL:
            target = self._patrol_points[self.patrol_index % len(self._patrol_points)]
            if np.linalg.norm(self.guard_pos - target) <= 0.8:
                self.patrol_index += 1
                target = self._patrol_points[self.patrol_index % len(self._patrol_points)]
            return target

        if action == TacticalAction.INVESTIGATE_SOUND:
            if not self.sense.sound:
                return None
            return self.guard_pos + np.array(self.sense.sound_rel, dtype=np.float32)

        if action == TacticalAction.GO_LAST_SEEN:
            return (
                None
                if self.last_seen_pos is None
                else self.last_seen_pos.copy()
            )

        if action == TacticalAction.SEARCH_LAST_SEEN:
            if self.last_seen_pos is None:
                return None

            # SEARCH is only allowed to instantiate its sector pattern once
            # the guard has actually reached the remembered area. Before that,
            # LAST_SEEN doctrine must use INTERCEPT / GO_LAST_SEEN.
            distance_to_memory = float(
                np.linalg.norm(self.guard_pos - self.last_seen_pos)
            )
            if (
                not self.search_waypoints
                and distance_to_memory > 1.35
            ):
                return None

            return self._current_search_target()

        if action == TacticalAction.PURSUE:
            if self.sense.visual:
                return self.guard_pos + np.array(self.sense.visual_rel, dtype=np.float32)
            return None if self.last_seen_pos is None else self.last_seen_pos.copy()

        if action == TacticalAction.INTERCEPT:
            if self.last_seen_pos is None:
                return None
            confidence = max(0.0, 1.0 - self.last_seen_age / self.cfg.memory_seconds)
            return self.last_seen_pos + self.last_seen_velocity * (0.9 * confidence)

        if action in (TacticalAction.FLANK_LEFT, TacticalAction.FLANK_RIGHT):
            if self.last_seen_pos is None:
                return None
            to_target = self.last_seen_pos - self.guard_pos
            ux, uy = _unit(float(to_target[0]), float(to_target[1]))
            sign = -1.0 if action == TacticalAction.FLANK_LEFT else 1.0
            lateral = np.array([-uy * sign, ux * sign], dtype=np.float32) * 2.2
            return self.last_seen_pos + lateral

        if action == TacticalAction.PROTECT_NEAREST_INTEL:
            candidates = [
                (idx, p) for idx, p in enumerate(self._intel)
                if self.known_intel_present[idx] == 1
            ]
            if not candidates:
                return None
            return min(candidates, key=lambda item: np.linalg.norm(self.guard_pos - item[1]))[1].copy()

        if action == TacticalAction.ARREST:
            if self.sense.visual and self.player_compliant:
                return self.guard_pos + np.array(self.sense.visual_rel, dtype=np.float32)
            return None

        if action == TacticalAction.FOCUS_VISUAL:
            return None

        if action == TacticalAction.INVESTIGATE_VISUAL:
            return self._best_suspicious_target()

        if action == TacticalAction.REQUEST_LOCAL_SUPPORT:
            return None

        if action == TacticalAction.RADIO_REPORT:
            return None

        if action == TacticalAction.CALL_REINFORCEMENTS:
            return None

        return None

    def _apply_action(self, action: TacticalAction) -> None:
        if action == TacticalAction.FOCUS_VISUAL:
            target = self._best_suspicious_target()
            if target is not None:
                delta = target - self.guard_pos
                if float(np.linalg.norm(delta)) > 1e-6:
                    self.guard_angle = math.atan2(
                        float(delta[1]),
                        float(delta[0]),
                    )
            self.focus_visual_timer = max(
                self.focus_visual_timer,
                1.20,
            )
            self.flashlight_active = bool(self.is_night)
            return

        if action == TacticalAction.REQUEST_LOCAL_SUPPORT:
            self._request_local_support()
            return

        if action == TacticalAction.RADIO_REPORT:
            if (
                self.personal_suspicion >= 0.50
                or self.intruder_known
                or self.sense.visual_confidence >= 0.50
            ):
                self.radio_reported = True
                self.alert = max(
                    self.alert,
                    int(FacilityAlert.SUSPICIOUS),
                )
            return

        if action == TacticalAction.CALL_REINFORCEMENTS:
            if self.alert >= 1 and not self.reinforcements_called:
                self.reinforcements_called = True
                self.reinforcement_timer = 0.0
            return

        target = self._choose_target(action)
        speed = self.cfg.guard_speed
        if action == TacticalAction.INVESTIGATE_VISUAL:
            speed *= 0.72
        if action in {
            TacticalAction.PURSUE,
            TacticalAction.INTERCEPT,
            TacticalAction.FLANK_LEFT,
            TacticalAction.FLANK_RIGHT,
        }:
            speed *= 1.15

        self.guard_pos = self._move_towards(self.guard_pos, target, speed)

        if target is not None:
            dx = float(target[0] - self.guard_pos[0])
            dy = float(target[1] - self.guard_pos[1])
            if abs(dx) + abs(dy) > 1e-6:
                self.guard_angle = math.atan2(dy, dx)

        if action == TacticalAction.SEARCH_LAST_SEEN:
            self._advance_search_pattern()

    def _decision_context(self) -> dict:
        sound_target = None
        if self.sense.sound:
            sound_target = (
                self.guard_pos
                + np.array(self.sense.sound_rel, dtype=np.float32)
            ).copy()

        visual_target = None
        if self.sense.visual_candidate:
            visual_target = (
                self.guard_pos
                + np.array(self.sense.visual_rel, dtype=np.float32)
            ).copy()

        memory_target = (
            None if self.last_seen_pos is None else self.last_seen_pos.copy()
        )

        intercept_target = None
        if self.last_seen_pos is not None:
            confidence = max(
                0.0,
                1.0 - self.last_seen_age / self.cfg.memory_seconds,
            )
            intercept_target = (
                self.last_seen_pos
                + self.last_seen_velocity * (0.9 * confidence)
            ).copy()

        known_present = self.known_intel_present.copy()
        known_positions = [
            p.copy()
            for idx, p in enumerate(self._intel)
            if known_present[idx] == 1
        ]
        protect_target = None
        if known_positions:
            protect_target = min(
                known_positions,
                key=lambda p: np.linalg.norm(self.guard_pos - p),
            ).copy()

        nearby_support = bool(
            self.buddy_available
            and self.buddy_pos is not None
            and np.linalg.norm(self.buddy_pos - self.guard_pos)
            <= self.cfg.local_support_radius
        )

        return {
            'visual': bool(self.sense.visual),
            'visual_candidate': bool(self.sense.visual_candidate),
            'visual_confidence': float(self.sense.visual_confidence),
            'visual_target': visual_target,
            'visual_band': int(self.sense.visual_band),
            'peripheral': bool(self.sense.peripheral),
            'motion_salience': float(self.sense.motion_salience),
            'spot_rate': float(self.sense.spot_rate),
            'sound': bool(self.sense.sound),
            'sound_confidence': float(self.sense.sound_confidence),
            'sound_target': sound_target,
            'player_compliant': bool(
                self.sense.visual and self.player_compliant
            ),
            'has_memory': self.last_seen_pos is not None,
            'memory_age': float(self.last_seen_age),
            'memory_target': memory_target,
            'memory_velocity': self.last_seen_velocity.copy(),
            'intercept_target': intercept_target,
            'alert': int(self.alert),
            'intruder_known': bool(self.intruder_known),
            'personal_suspicion': float(self.personal_suspicion),
            'visual_exposure_count': int(self.visual_exposure_count),
            'audio_repeat_count': int(self.audio_repeat_count),
            'nearby_support': nearby_support,
            'buddy_supporting': bool(self.buddy_supporting),
            'radio_reported': bool(self.radio_reported),
            'reinforcements_called': bool(self.reinforcements_called),
            'known_intel_present': known_present,
            'confirmed_theft': bool(np.any(known_present == 0)),
            'protect_target': protect_target,
            'search_active': bool(
                self.search_waypoints
                or self.search_waypoints_visited > 0
            ),
            'search_index': int(self.search_index),
            'search_waypoints_visited': int(
                self.search_waypoints_visited
            ),
            # IMPORTANT: decision_context must never mutate the world.
            # Do not call _current_search_target() here because that helper
            # may create a search pattern. Merely observing/info-rendering
            # must not turn LAST_SEEN into SEARCH.
            'search_target': (
                self.search_waypoints[
                    min(self.search_index, len(self.search_waypoints) - 1)
                ].copy()
                if self.search_waypoints
                else None
            ),
            'reinforcements_active': bool(self.reinforcements_active),
        }

    def _tactical_phase(self, decision: dict) -> str:
        if decision['visual'] and decision['player_compliant']:
            return 'arrest'

        # V6.3: a confirmed sighting has two different tactical moments.
        # First report the confirmed contact once; after that, pursue.
        if decision['visual']:
            if not decision['radio_reported']:
                return 'visual_unreported'
            return 'visual'

        confidence = decision['visual_confidence']
        if decision['visual_candidate']:
            if confidence >= 0.75:
                return 'probable_visual'
            if confidence >= 0.50:
                return 'suspicious_visual'
            if confidence >= 0.25:
                return 'attention'
            return 'peripheral_hint'

        if decision['sound'] and not decision['has_memory']:
            if decision['intruder_known'] or decision['audio_repeat_count'] >= 2:
                return 'sound_tactical'
            return 'sound'

        if decision['has_memory']:
            target = decision['memory_target']
            if target is not None:
                distance = float(np.linalg.norm(self.guard_pos - target))
                if distance <= 1.35 or decision['search_active']:
                    return 'search'
            return 'last_seen'

        if decision['confirmed_theft']:
            return 'theft'

        return 'routine'

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _reward(
        self,
        action: TacticalAction,
        previous_guard_pos: np.ndarray,
        decision: dict,
    ) -> float:
        reward = -0.004
        phase = self._tactical_phase(decision)

        if phase == 'routine':
            reward += 0.15 if action == TacticalAction.PATROL else -0.12
            if action == TacticalAction.HOLD:
                reward += 0.10

        elif phase == 'peripheral_hint':
            if action == TacticalAction.FOCUS_VISUAL:
                reward += 0.24
            elif action == TacticalAction.PATROL:
                reward += 0.01
            else:
                reward -= 0.08

        elif phase == 'attention':
            # Priority matters. Repeated teasing must eventually escalate
            # instead of FOCUS_VISUAL always remaining the numerically best
            # reward.
            if (
                decision['visual_exposure_count'] >= 3
                and not decision['reinforcements_called']
            ):
                if action == TacticalAction.CALL_REINFORCEMENTS:
                    reward += 0.64
                elif action == TacticalAction.FOCUS_VISUAL:
                    reward += 0.16
                else:
                    reward -= 0.24
            elif (
                decision['visual_exposure_count'] >= 2
                and decision['nearby_support']
                and not decision['buddy_supporting']
            ):
                if action == TacticalAction.REQUEST_LOCAL_SUPPORT:
                    reward += 0.58
                elif action == TacticalAction.FOCUS_VISUAL:
                    reward += 0.20
                else:
                    reward -= 0.22
            else:
                if action == TacticalAction.FOCUS_VISUAL:
                    reward += 0.50
                else:
                    reward -= 0.14

        elif phase == 'suspicious_visual':
            if not decision['intruder_known']:
                if action == TacticalAction.INVESTIGATE_VISUAL:
                    reward += 0.52
                elif action == TacticalAction.FOCUS_VISUAL:
                    reward += 0.22
                elif (
                    decision['visual_exposure_count'] >= 2
                    and decision['nearby_support']
                    and action == TacticalAction.REQUEST_LOCAL_SUPPORT
                ):
                    reward += 0.34
                else:
                    reward -= 0.13
            else:
                if action == TacticalAction.REQUEST_LOCAL_SUPPORT:
                    reward += 0.42
                elif action == TacticalAction.FOCUS_VISUAL:
                    reward += 0.30
                elif action == TacticalAction.RADIO_REPORT:
                    reward += 0.28
                else:
                    reward -= 0.12

        elif phase == 'probable_visual':
            if decision['intruder_known']:
                if action == TacticalAction.RADIO_REPORT:
                    reward += 0.48
                elif action == TacticalAction.FOCUS_VISUAL:
                    reward += 0.36
                elif action == TacticalAction.REQUEST_LOCAL_SUPPORT:
                    reward += 0.32
                else:
                    reward -= 0.12
            else:
                if action == TacticalAction.INVESTIGATE_VISUAL:
                    reward += 0.48
                elif action == TacticalAction.FOCUS_VISUAL:
                    reward += 0.38
                elif action == TacticalAction.RADIO_REPORT:
                    reward += 0.18
                else:
                    reward -= 0.12

        elif phase == 'sound':
            if action == TacticalAction.INVESTIGATE_SOUND:
                reward += 0.42 + 0.10 * decision['sound_confidence']
            else:
                reward -= 0.12

        elif phase == 'sound_tactical':
            if decision['nearby_support'] and not decision['buddy_supporting']:
                if action == TacticalAction.REQUEST_LOCAL_SUPPORT:
                    reward += 0.62
                elif action == TacticalAction.INVESTIGATE_SOUND:
                    reward += 0.06
                elif action == TacticalAction.PROTECT_NEAREST_INTEL:
                    reward -= 0.34
                else:
                    reward -= 0.22
            else:
                if action == TacticalAction.INVESTIGATE_SOUND:
                    reward += 0.48
                elif (
                    decision['intruder_known']
                    and not decision['radio_reported']
                    and action == TacticalAction.RADIO_REPORT
                ):
                    reward += 0.18
                elif action == TacticalAction.PROTECT_NEAREST_INTEL:
                    reward -= 0.30
                else:
                    reward -= 0.18

        elif phase == 'visual_unreported':
            # First confirmed sighting: communicate once. Pursuit is still
            # sensible, but reporting should clearly win this decision.
            if action == TacticalAction.RADIO_REPORT:
                reward += 0.66
            elif action == TacticalAction.PURSUE:
                reward += 0.16
            elif action == TacticalAction.REQUEST_LOCAL_SUPPORT:
                reward += 0.08
            else:
                reward -= 0.24

        elif phase == 'visual':
            # Contact is already on the radio: stop talking and act.
            if action == TacticalAction.PURSUE:
                reward += 0.54
            elif action == TacticalAction.INTERCEPT:
                reward += 0.08
            elif action == TacticalAction.RADIO_REPORT:
                reward -= 0.28
            elif action == TacticalAction.PROTECT_NEAREST_INTEL:
                reward -= 0.30
            else:
                reward -= 0.22

        elif phase == 'arrest':
            reward += 0.82 if action == TacticalAction.ARREST else -0.28

        elif phase == 'last_seen':
            if action == TacticalAction.INTERCEPT:
                speed = float(np.linalg.norm(decision['memory_velocity']))
                reward += 0.46 if speed > 0.08 else 0.24
            elif action == TacticalAction.GO_LAST_SEEN:
                reward += 0.42
            elif action in {
                TacticalAction.FLANK_LEFT,
                TacticalAction.FLANK_RIGHT,
            }:
                reward += 0.10
            elif action == TacticalAction.SEARCH_LAST_SEEN:
                # Searching before actually reaching the remembered area is
                # precisely the V6.2 confusion we want to eliminate.
                reward -= 0.34
            elif action == TacticalAction.PURSUE:
                reward -= 0.30
            else:
                reward -= 0.18

        elif phase == 'search':
            if action == TacticalAction.SEARCH_LAST_SEEN:
                reward += 0.58
                if (
                    self.search_waypoints_visited
                    > decision['search_waypoints_visited']
                ):
                    reward += 0.18
            elif action in {
                TacticalAction.FLANK_LEFT,
                TacticalAction.FLANK_RIGHT,
            }:
                reward += 0.12
            elif action in {
                TacticalAction.INTERCEPT,
                TacticalAction.GO_LAST_SEEN,
            }:
                reward -= 0.24
            elif action == TacticalAction.PURSUE:
                reward -= 0.38
            else:
                reward -= 0.18

        elif phase == 'theft':
            if not decision['reinforcements_called']:
                if action == TacticalAction.CALL_REINFORCEMENTS:
                    reward += 0.66
                elif action == TacticalAction.PROTECT_NEAREST_INTEL:
                    reward += 0.08
                else:
                    reward -= 0.12
            else:
                if action == TacticalAction.PROTECT_NEAREST_INTEL:
                    reward += 0.34
                elif action == TacticalAction.CALL_REINFORCEMENTS:
                    reward -= 0.18
                else:
                    reward -= 0.07

        # V6.3 anti-contamination: objective protection belongs to a confirmed
        # theft response, not to an active visual/audio identification problem.
        if (
            phase in {
                'peripheral_hint',
                'attention',
                'suspicious_visual',
                'probable_visual',
                'sound',
                'sound_tactical',
                'visual_unreported',
                'visual',
            }
            and action == TacticalAction.PROTECT_NEAREST_INTEL
        ):
            reward -= 0.20

        # Repeated teasing: after the third 25%+ episode, escalation is rational.
        if (
            decision['visual_exposure_count'] >= 3
            and phase in {'attention', 'suspicious_visual', 'probable_visual'}
            and action == TacticalAction.CALL_REINFORCEMENTS
        ):
            reward += 0.38

        # Progress uses only perceived / remembered targets.
        progress_target = None
        if action == TacticalAction.INVESTIGATE_SOUND:
            progress_target = decision['sound_target']
        elif action == TacticalAction.INVESTIGATE_VISUAL:
            progress_target = decision['visual_target']
        elif action == TacticalAction.PURSUE:
            progress_target = (
                decision['visual_target']
                if decision['visual']
                else decision['memory_target']
            )
        elif action == TacticalAction.GO_LAST_SEEN:
            progress_target = decision['memory_target']
        elif action == TacticalAction.SEARCH_LAST_SEEN:
            progress_target = decision['search_target']
        elif action == TacticalAction.INTERCEPT:
            progress_target = decision['intercept_target']
        elif action in {TacticalAction.FLANK_LEFT, TacticalAction.FLANK_RIGHT}:
            progress_target = decision['memory_target']
        elif action == TacticalAction.PROTECT_NEAREST_INTEL:
            progress_target = decision['protect_target']
        elif action == TacticalAction.ARREST:
            progress_target = decision['visual_target']

        if progress_target is not None:
            old_d = float(np.linalg.norm(previous_guard_pos - progress_target))
            new_d = float(np.linalg.norm(self.guard_pos - progress_target))
            reward += float(
                np.clip((old_d - new_d) * 0.065, -0.06, 0.10)
            )

        if phase == 'arrest' and action == TacticalAction.ARREST:
            reward += 0.08
        if self.search_completed_event:
            reward += 0.35
        if self.reinforcements_spawned_event:
            reward += 0.20

        return float(reward)

    def _terminal_events(
        self,
        action: TacticalAction,
        decision: dict,
    ) -> tuple[float, bool]:
        distance = float(np.linalg.norm(self.guard_pos - self.player_pos))

        valid_arrest = (
            action == TacticalAction.ARREST
            and decision["visual"]
            and decision["player_compliant"]
            and distance <= self.cfg.arrest_distance
        )
        if valid_arrest:
            self.arrest_progress += self.cfg.dt
        else:
            self.arrest_progress = max(0.0, self.arrest_progress - self.cfg.dt * 1.5)

        if self.arrest_progress >= 1.0:
            return 12.0, True

        if self.reinforcement_capture_event:
            return 10.0, True

        # Fleeing targets may still be physically caught by a real pursuit.
        if (
            action == TacticalAction.PURSUE
            and decision["visual"]
            and not decision["player_compliant"]
            and distance <= self.cfg.capture_distance * 0.55
        ):
            return 8.0, True

        if (
            self.player_goal_idx >= len(self.episode_intel_order)
            and np.linalg.norm(self.player_pos - self._exit) <= self.cfg.escape_distance
        ):
            return -12.0, True

        return 0.0, False

    def _sample_training_scenario(self) -> str:
        names = [
            'routine',
            'peripheral',
            'attention',
            'repeat_visual',
            'sound',
            'repeat_sound',
            'visual',
            'lost_contact',
            'search',
            'arrest',
            'theft',
        ]
        weights = np.array(self.cfg.curriculum_weights, dtype=np.float64)
        weights = weights / weights.sum()
        return str(self.rng.choice(names, p=weights))

    def _random_free_position(
        self,
        min_distance_from: Optional[np.ndarray] = None,
        min_distance: float = 0.0,
    ) -> np.ndarray:
        for _ in range(500):
            x = float(self.rng.uniform(1.5, self.cfg.width - 1.5))
            y = float(self.rng.uniform(1.5, self.cfg.height - 1.5))
            p = np.array([x, y], dtype=np.float32)
            cx, cy = self._cell(p)
            if not self._free(cx, cy):
                continue
            if (
                min_distance_from is not None
                and np.linalg.norm(p - min_distance_from) < min_distance
            ):
                continue
            return p
        return np.array([2.0, 17.0], dtype=np.float32)

    def _configure_curriculum_episode(self) -> None:
        scenario = self.training_scenario

        self.guard_pos = self._random_free_position()
        self.player_pos = self._random_free_position(
            min_distance_from=self.guard_pos,
            min_distance=7.0 if scenario == 'routine' else 4.0,
        )

        if scenario == 'routine':
            self.alert = int(FacilityAlert.CALM)
            self.player_style = self.rng.choice(['stealth', 'mixed'])

        elif scenario == 'peripheral':
            self.alert = int(FacilityAlert.CALM)
            self.guard_pos = np.array([12.0, 8.0], dtype=np.float32)
            self.guard_angle = 0.0
            angle = math.radians(float(self.rng.choice([66.0, 72.0, -68.0])))
            radius = 5.8
            self.player_pos = self.guard_pos + np.array(
                [math.cos(angle) * radius, math.sin(angle) * radius],
                dtype=np.float32,
            )
            self.player_style = self.rng.choice(['stealth', 'runner'])
            self.player_stance = (
                PlayerStance.CROUCHED
                if self.player_style == 'stealth'
                else PlayerStance.STANDING
            )

        elif scenario == 'attention':
            self.alert = int(FacilityAlert.CALM)
            self.guard_pos = np.array([10.0, 8.0], dtype=np.float32)
            self.guard_angle = 0.0
            self.player_pos = np.array([19.0, 8.5], dtype=np.float32)
            self.player_style = 'stealth'
            self.player_stance = PlayerStance.CROUCHED
            self.visual_confidence = 0.20

        elif scenario == 'repeat_visual':
            self.alert = int(FacilityAlert.SUSPICIOUS)
            self.guard_pos = np.array([10.0, 8.0], dtype=np.float32)
            self.guard_angle = 0.0
            self.player_pos = np.array([18.5, 8.0], dtype=np.float32)
            self.player_style = 'mixed'
            self.player_stance = PlayerStance.STANDING
            self.visual_exposure_count = 2
            self.personal_suspicion = 0.58
            self.visual_confidence = 0.20

        elif scenario == 'sound':
            self.alert = int(FacilityAlert.CALM)
            self.guard_angle = 0.0
            for offset in [
                np.array([-4.5, 0.0], dtype=np.float32),
                np.array([-3.8, 1.4], dtype=np.float32),
                np.array([-3.8, -1.4], dtype=np.float32),
            ]:
                candidate = self.guard_pos + offset
                if self._free(*self._cell(candidate)):
                    self.player_pos = candidate
                    break
            self.player_style = 'runner'

        elif scenario == 'repeat_sound':
            self.alert = int(FacilityAlert.SUSPICIOUS)
            self.guard_angle = 0.0
            self.player_pos = self.guard_pos + np.array([-4.0, 0.0], dtype=np.float32)
            if not self._free(*self._cell(self.player_pos)):
                self.player_pos = self._random_free_position(
                    min_distance_from=self.guard_pos,
                    min_distance=3.0,
                )
            self.player_style = 'runner'
            self.audio_repeat_count = 2
            self.personal_suspicion = 0.48

        elif scenario == 'visual':
            self.alert = int(FacilityAlert.CALM)
            self.guard_angle = 0.0
            candidate = self.guard_pos + np.array([3.0, 0.0], dtype=np.float32)
            if self._free(*self._cell(candidate)):
                self.player_pos = candidate
            self.player_style = 'runner'
            self.visual_confidence = 0.80

        elif scenario == 'lost_contact':
            self.alert = int(FacilityAlert.CONFIRMED)
            self.intruder_known = True
            self.time_since_confirmed_intruder = 0.0
            self.guard_pos = np.array([16.5, 5.0], dtype=np.float32)
            self.player_pos = np.array([18.2, 5.0], dtype=np.float32)
            self.guard_angle = 0.0
            self.player_style = 'runner'
            self.visual_confidence = 1.0

        elif scenario == 'search':
            self.alert = int(FacilityAlert.SEARCHING)
            self.intruder_known = True
            self.guard_pos = np.array([16.2, 5.4], dtype=np.float32)
            self.last_seen_pos = np.array([18.2, 5.4], dtype=np.float32)
            self.last_seen_velocity = np.array([1.0, 0.0], dtype=np.float32)
            self.last_seen_age = 1.2
            self.player_pos = np.array([26.0, 15.0], dtype=np.float32)
            self.guard_angle = 0.0
            self.player_style = 'stealth'
            self._ensure_search_pattern()

        elif scenario == 'arrest':
            self.alert = int(FacilityAlert.CONFIRMED)
            self.intruder_known = True
            candidate = self.guard_pos + np.array([1.0, 0.0], dtype=np.float32)
            if self._free(*self._cell(candidate)):
                self.player_pos = candidate
            self.guard_angle = 0.0
            self.player_style = 'mixed'
            self.visual_confidence = 1.0

        elif scenario == 'theft':
            self.alert = int(FacilityAlert.SUSPICIOUS)
            stolen_id = int(self.rng.integers(0, 3))
            self.player_has_intel[stolen_id] = 1
            self.episode_has_intel[stolen_id] = 1
            self.known_intel_present[stolen_id] = 0
            self.personal_suspicion = 0.68
            self.guard_pos = self._random_free_position()
            self.player_pos = self._random_free_position(
                min_distance_from=self.guard_pos,
                min_distance=8.0,
            )
            self.guard_angle = float(self.rng.uniform(-math.pi, math.pi))

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.steps = 0
        self.guard_pos = np.array([16.0, 8.0], dtype=np.float32)
        self.guard_angle = float(self.rng.uniform(-math.pi, math.pi))
        self.player_pos = np.array([2.0, 17.0], dtype=np.float32)
        self.player_velocity = np.zeros(2, dtype=np.float32)
        self.guard_start = self.guard_pos.copy()
        self.alert = 0
        self.patrol_index = int(
            self.rng.integers(0, len(self._patrol_points))
        )
        self.last_seen_pos: Optional[np.ndarray] = None
        self.last_seen_velocity = np.zeros(2, dtype=np.float32)
        self.last_seen_age = self.cfg.memory_seconds + 1.0
        self.known_intel_present = np.ones(3, dtype=np.int8)
        self.reinforcements_called = False
        self.arrest_progress = 0.0
        self.last_action = TacticalAction.HOLD

        # V6 perception / psychology state.
        self.visual_confidence = 0.0
        self.visual_attention_episode_active = False
        self.visual_exposure_count = 0
        self.visual_anomaly_pos: Optional[np.ndarray] = None
        self.visual_anomaly_age = 999.0
        self.last_visual_rate = 0.0
        self.last_visual_band = VisualBand.NONE
        self.personal_suspicion = 0.0
        self.intruder_known = False
        self.time_since_confirmed_intruder = 999.0
        self.focus_visual_timer = 0.0
        self.is_night = bool(self.rng.random() < 0.5)
        self.flashlight_active = False
        self.audio_repeat_count = 0
        self.sound_episode_active = False
        self.time_since_last_sound = 999.0
        self.last_sound_pos: Optional[np.ndarray] = None
        self.radio_reported = False

        # One physical local buddy lets the single-agent curriculum learn the
        # social-support doctrine before the full multi-agent phase.
        self.buddy_available = bool(
            self.rng.random() < self.cfg.local_support_probability
        )
        self.buddy_pos: Optional[np.ndarray] = None
        if self.buddy_available:
            self.buddy_pos = self._random_free_position(
                min_distance_from=self.guard_pos,
                min_distance=3.0,
            )
        self.buddy_supporting = False
        self.buddy_target: Optional[np.ndarray] = None

        # V5 search-sector state.
        self.search_anchor: Optional[np.ndarray] = None
        self.search_waypoints: list[np.ndarray] = []
        self.search_index = 0
        self.search_waypoints_visited = 0
        self.search_completed_event = False

        # V5 reinforcement state.
        self.reinforcement_timer = 0.0
        self.reinforcements_active = False
        self.reinforcement_positions: list[np.ndarray] = []
        self.reinforcement_angles: list[float] = []
        self.reinforcements_spawned_event = False
        self.reinforcement_capture_event = False
        self.reinforcement_radio_contact = False

        self._reset_player_bot()

        # Explicit curriculum scenario may be injected for deterministic tests.
        forced = None if options is None else options.get("scenario")
        self.training_scenario = (
            str(forced)
            if forced is not None
            else self._sample_training_scenario()
        )

        self._configure_curriculum_episode()

        if self.buddy_available:
            self.buddy_pos = self._random_free_position(
                min_distance_from=self.guard_pos,
                min_distance=2.5,
            )

        self.sense = self._sense()
        self._update_knowledge()
        return self._observation(), self._info()

    def step(self, action: int):
        action_enum = TacticalAction(int(action))
        self.steps += 1

        self.search_completed_event = False
        self.reinforcements_spawned_event = False
        self.reinforcement_capture_event = False
        self.reinforcement_radio_contact = False

        previous_guard = self.guard_pos.copy()

        # Snapshot exactly what the policy knew when it selected the action.
        decision = self._decision_context()
        decision_phase = self._tactical_phase(decision)

        # Lead guard acts from current knowledge.
        self._apply_action(action_enum)

        # World/player advances.
        self._player_step()

        # Lead guard senses first.
        self.sense = self._sense()
        self._update_knowledge()

        self.last_action = action_enum
        self._step_buddy()

        # Reinforcements now physically move/sense using their own LOS. Any
        # radio update they produce becomes knowledge for future decisions.
        self._step_reinforcements()

        reward = self._reward(
            action_enum,
            previous_guard,
            decision,
        )
        terminal_reward, terminated = self._terminal_events(
            action_enum,
            decision,
        )
        self.last_action = action_enum
        reward += terminal_reward

        truncated = self.steps >= self.cfg.max_steps
        obs = self._observation()
        info = self._info()
        info["action_name"] = action_enum.name
        info["terminal_reward"] = terminal_reward
        info["decision_phase"] = decision_phase
        info["result_phase"] = info["tactical_phase"]

        return obs, float(reward), terminated, truncated, info

    def _info(self) -> dict:
        return {
            "alert": self.alert,
            "visual_contact": self.sense.visual,
            "sound_contact": self.sense.sound,
            "last_seen_age": None if self.last_seen_pos is None else self.last_seen_age,
            "intel_known_present": self.known_intel_present.tolist(),
            "reinforcements_called": self.reinforcements_called,
            "player_style": self.player_style,
            "training_scenario": self.training_scenario,
            "episode_goal_mode": self.episode_goal_mode,
            "arrest_progress": float(self.arrest_progress),
            "tactical_phase": self._tactical_phase(
                self._decision_context()
            ),
            "search_waypoint_index": int(self.search_index),
            "search_waypoints_visited": int(
                self.search_waypoints_visited
            ),
            "search_waypoint_count": int(
                len(self.search_waypoints)
            ),
            "reinforcement_timer": float(
                self.reinforcement_timer
            ),
            "reinforcements_active": bool(
                self.reinforcements_active
            ),
            "reinforcement_count": int(
                len(self.reinforcement_positions)
            ),
            "reinforcement_radio_contact": bool(
                self.reinforcement_radio_contact
            ),
            "visual_confidence": float(self.sense.visual_confidence),
            "visual_candidate": bool(self.sense.visual_candidate),
            "visual_band": int(self.sense.visual_band),
            "visual_peripheral": bool(self.sense.peripheral),
            "spot_rate": float(self.sense.spot_rate),
            "personal_suspicion": float(self.personal_suspicion),
            "visual_exposure_count": int(self.visual_exposure_count),
            "audio_repeat_count": int(self.audio_repeat_count),
            "intruder_known": bool(self.intruder_known),
            "focus_visual": bool(self.focus_visual_timer > 0.0),
            "flashlight_active": bool(self.flashlight_active),
            "is_night": bool(self.is_night),
            "radio_reported": bool(self.radio_reported),
            "buddy_available": bool(self.buddy_available),
            "buddy_supporting": bool(self.buddy_supporting),
            # Deliberately no hidden player position here.
        }

    def render(self):
        if self.render_mode != "ansi":
            return None

        g = self._cell(self.guard_pos)
        p = self._cell(self.player_pos)
        rows = []
        for y in range(self.cfg.height):
            row = []
            for x in range(self.cfg.width):
                ch = "#" if self.grid[y, x] else "."
                if (x, y) == g:
                    ch = "G"
                if (
                    self.buddy_available
                    and self.buddy_pos is not None
                    and (x, y) == self._cell(self.buddy_pos)
                ):
                    ch = "B"
                for reinforcement in self.reinforcement_positions:
                    if (x, y) == self._cell(reinforcement):
                        ch = "R"
                if (x, y) == p and self.sense.visual:
                    ch = "P"
                for idx, intel in enumerate(self._intel):
                    if (x, y) == self._cell(intel) and not self.player_has_intel[idx]:
                        ch = str(idx + 1)
                if (x, y) == self._cell(self._exit):
                    ch = "E"
                row.append(ch)
            rows.append("".join(row))
        return "\n".join(rows)
