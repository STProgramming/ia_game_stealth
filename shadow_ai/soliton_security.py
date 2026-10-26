from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np
from .playable_facility import build_research_grid
from .playable_vitals import PlayableVitals


# One playable guard sensor, shared by the lead, patrols and reinforcements.
GUARD_VISION_RANGE = 11.5
GUARD_FOV = 104.0


def turn_towards(angle: float, target: float, max_turn: float) -> float:
    delta = (target - angle + math.pi) % math.tau - math.pi
    return (angle + max(-max_turn, min(max_turn, delta)) + math.pi) % math.tau - math.pi


def build_soliton_grid(width: int = 48, height: int = 30) -> np.ndarray:
    """Build the larger playable facility used by the browser game.

    The training environments deliberately keep their compact map.  This map
    is runtime-only, so existing checkpoints and regression tests retain the
    geometry they were trained against.
    """

    return build_research_grid(width, height)



def line_clear(grid: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    distance = float(np.linalg.norm(b - a))
    samples = max(3, int(math.ceil(distance * 8.0)) + 1)
    height, width = grid.shape
    for t in np.linspace(0.0, 1.0, samples, endpoint=True):
        point = a + (b - a) * t
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if not (0 <= x < width and 0 <= y < height) or grid[y, x]:
            return False
    return True


def can_see(
    grid: np.ndarray,
    observer: np.ndarray,
    angle: float,
    target: np.ndarray,
    vision_range: float,
    fov_degrees: float,
    immediate_range: float = 0.75,
) -> bool:
    delta = target - observer
    distance = float(np.linalg.norm(delta))
    if distance > vision_range:
        return False
    if distance > immediate_range:
        facing = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        dot = float(np.dot(facing, delta / max(distance, 1e-6)))
        if dot < math.cos(math.radians(fov_degrees / 2.0)):
            return False
    return line_clear(grid, observer, target)


def is_behind_guard(
    player: np.ndarray,
    guard: np.ndarray,
    guard_angle: float,
    max_distance: float = 1.18,
) -> bool:
    delta = player - guard
    distance = float(np.linalg.norm(delta))
    if distance > max_distance or distance <= 1e-6:
        return False
    facing = np.array(
        [math.cos(guard_angle), math.sin(guard_angle)],
        dtype=np.float32,
    )
    return float(np.dot(facing, delta / distance)) <= -0.48


@dataclass
class SecurityGuard:
    guard_id: str
    pos: np.ndarray
    angle: float
    patrol: tuple[tuple[float, float], ...]
    patrol_index: int = 0
    state: str = "PATROL"
    target: Optional[np.ndarray] = None
    last_seen: Optional[np.ndarray] = None
    last_seen_age: float = 999.0
    suspicion: float = 0.0
    visual_confidence: float = 0.0
    search_timer: float = 0.0
    search_step: int = 0
    alive: bool = True
    unconscious: bool = False
    radio_cooldown: float = 0.0
    evidence_id: int = -1
    pause_remaining: float = 0.0
    pause_duration: float = 0.0
    pause_angle: float = 0.0
    exit_pos: Optional[np.ndarray] = None
    flash_count: int = 1
    smoke_count: int = 1
    vitals: PlayableVitals = field(default_factory=PlayableVitals)
    aim_timer: float = 0.0
    shot_cooldown: float = 0.0
    ammo: int = 16
    fire_goal: Optional[np.ndarray] = None
    blinded_timer: float = 0.0
    armor: float = 0.0
    has_shield: bool = False
    policy_action: Optional[int] = None
    seeing_player: bool = False
    weapon: str = "PISTOL"
    grenade_count: int = 0
    disarmed: bool = False

    @property
    def active(self) -> bool:
        return self.alive and not self.unconscious and self.vitals.health > 0

    def patrol_target(self) -> np.ndarray:
        point = self.patrol[self.patrol_index % len(self.patrol)]
        return np.asarray(point, dtype=np.float32)

    def advance_patrol(self) -> None:
        self.patrol_index = (self.patrol_index + 1) % len(self.patrol)


@dataclass
class SecurityCamera:
    camera_id: str
    pos: np.ndarray
    center_angle: float
    sweep: float = math.radians(62.0)
    speed: float = 0.72
    phase: float = 0.0
    angle: float = 0.0
    vision_range: float = 11.5
    fov: float = 58.0
    confidence: float = 0.0
    tracking: bool = False
    online: bool = True

    def step(self, dt: float) -> None:
        self.phase += dt * self.speed
        self.angle = self.center_angle + math.sin(self.phase) * self.sweep


@dataclass
class NoiseEvent:
    event_id: int
    pos: np.ndarray
    radius: float
    ttl: float
    source: str
    pulse: bool = True
    device_id: Optional[int] = None


@dataclass
class ThrownDiversion:
    device_id: int
    start: np.ndarray
    target: np.ndarray
    pos: np.ndarray
    flight: float = 0.0
    landed: bool = False
    age: float = 0.0
    next_pulse: float = 0.0
    pulses_left: int = 3


@dataclass
class BodyEvidence:
    body_id: int
    guard_id: str
    pos: np.ndarray
    angle: float
    lethal: bool
    fall: float = 1.0
    discovered: bool = False


@dataclass
class BloodMark:
    mark_id: int
    pos: np.ndarray
    age: float = 0.0
    discovered_by: set[str] = field(default_factory=set)

