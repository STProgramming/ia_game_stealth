"""Navigation and local corner inspections for the playable maps only."""
from collections import deque
from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from .soliton_security import line_clear, turn_towards


class PlayableNavigation:
    def __init__(self, grid, cell, free, astar, nearest_free):
        self.grid = grid
        self.cell = cell
        self.free = free
        self.astar = astar
        self.nearest_free = nearest_free
        self.paths: dict[str, tuple[np.ndarray, list[np.ndarray]]] = {}

    def goal(self, point: np.ndarray) -> np.ndarray:
        height, width = self.grid.shape
        point = np.clip(np.asarray(point, dtype=np.float32), [0, 0],
                        [width - 1, height - 1]).astype(np.float32)
        x, y = self.cell(point)
        if not self.free(x, y):
            point = np.array(self.nearest_free(x, y), dtype=np.float32)
        return point.copy()

    def move(self, key: str, pos: np.ndarray, goal: Optional[np.ndarray], speed: float) -> np.ndarray:
        if goal is None or speed <= 0:
            return pos.copy()
        goal = self.goal(goal)
        corrected = self.goal(pos)
        if not np.array_equal(corrected, pos):
            pos = corrected
            self.paths.pop(key, None)
        cached = self.paths.get(key)
        if cached is None or self.cell(cached[0]) != self.cell(goal):
            path = self.astar(pos, goal)
            center = np.array(self.cell(pos), dtype=np.float32)
            if path and not line_clear(self.grid, pos, path[0]):
                path.insert(0, center)
            cached = (goal.copy(), path)
            self.paths[key] = cached
        path = cached[1]
        while path and np.linalg.norm(path[0] - pos) < 0.001:
            path.pop(0)
        target = path[0] if path else goal
        delta = target - pos
        distance = float(np.linalg.norm(delta))
        if distance < 1e-6:
            return pos.copy()
        candidate = pos + delta / distance * min(speed, distance)
        if self.free(*self.cell(candidate)) and line_clear(self.grid, pos, candidate):
            return candidate.astype(np.float32)
        self.paths.pop(key, None)
        return pos.copy()

    def path_length(self, pos: np.ndarray, goal: np.ndarray) -> float:
        goal = self.goal(goal)
        path = self.astar(pos, goal)
        if not path:
            return float(np.linalg.norm(goal - pos)) if line_clear(self.grid, pos, goal) else float("inf")
        points = [pos] + list(path)
        if not line_clear(self.grid, pos, path[0]):
            points.insert(1, np.array(self.cell(pos), dtype=np.float32))
        length = 0.0
        for a, b in zip(points, points[1:]):
            if not line_clear(self.grid, a, b):
                return float("inf")
            length += float(np.linalg.norm(b - a))
        return length + float(np.linalg.norm(goal - points[-1]))


def local_search_points(grid: np.ndarray, anchor: np.ndarray, unit_index: int = 0) -> list[np.ndarray]:
    """Reachable wall/corner checks near a report, independent of live truth."""
    height, width = grid.shape
    start = tuple(int(round(float(v))) for v in anchor)
    distance = {start: 0}
    queue = deque([start])
    candidates = []
    while queue:
        x, y = queue.popleft()
        depth = distance[x, y]
        neighbors = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not grid[ny, nx]:
                neighbors.append((nx, ny))
                if depth < 7 and (nx, ny) not in distance:
                    distance[nx, ny] = depth + 1
                    queue.append((nx, ny))
        if 2 <= depth <= 7:
            # Cells alongside walls expose corners and doorway approaches.
            wall = len(neighbors) < 4
            candidates.append((not wall, depth, x, y))
    phase = unit_index * 2.399963
    candidates.sort(key=lambda p: (p[0], (math.atan2(p[3] - start[1],
                                                   p[2] - start[0]) - phase) % math.tau,
                                   p[1]))
    entry = np.asarray(anchor, dtype=np.float32).copy()
    if unit_index:
        nearby = [(x, y) for (x, y), depth in distance.items() if depth == 1]
        if nearby:
            def angular_distance(point):
                angle = math.atan2(point[1] - start[1], point[0] - start[0])
                return abs((angle - phase + math.pi) % math.tau - math.pi)

            x, y = min(nearby, key=angular_distance)
            entry = np.array([x, y], dtype=np.float32)
    points = [entry]
    # Spread inspections between units and avoid checking adjacent tiles.
    for _, _, x, y in candidates:
        point = np.array([x, y], dtype=np.float32)
        if all(np.linalg.norm(point - previous) >= 2.0 for previous in points):
            points.append(point)
        if len(points) >= 6:
            break
    return points


@dataclass
class HumanMotion:
    speed: float = 0.0

    def move(self, navigator: PlayableNavigation, key: str, pos: np.ndarray, angle: float,
             target: Optional[np.ndarray], speed: float, dt: float,
             turn_rate: float = 2.0) -> tuple[np.ndarray, float]:
        if target is None:
            self.speed = 0.0
            return pos.copy(), angle
        # Preview must not consume waypoints before the real movement happens.
        cached = navigator.paths.get(key)
        saved = None if cached is None else (cached[0].copy(), list(cached[1]))
        preview = navigator.move(key, pos, target, speed * dt)
        if saved is not None:
            navigator.paths[key] = saved
        direction = preview - pos
        if np.linalg.norm(direction) < 1e-7:
            self.speed = 0.0
            return pos.copy(), angle
        heading = math.atan2(float(direction[1]), float(direction[0]))
        angle = turn_towards(angle, heading, turn_rate * dt)
        alignment = max(0.0, math.cos(heading - angle))
        distance = float(np.linalg.norm(navigator.goal(target) - pos))
        desired = min(speed, math.sqrt(5.0 * distance)) * alignment
        self.speed += max(-3.5 * dt, min(2.5 * dt, desired - self.speed))
        # Turn in place before walking; pathfinding still owns collisions.
        step = min(self.speed, speed * alignment) * dt
        return navigator.move(key, pos, target, step), angle


class MotionRecovery:
    """Replan an active pursuit after a prolonged stall, without crossing walls."""
    def __init__(self):
        self.stalled = {}

    def step(self, navigator, key, old, moved, goal, speed, dt, active):
        waiting = (active and goal is not None and np.linalg.norm(goal - old) > .45
                   and speed > 0 and np.linalg.norm(moved - old) < 1e-5)
        self.stalled[key] = self.stalled.get(key, 0.) + dt if waiting else 0.
        if self.stalled[key] < .8:
            return moved
        navigator.paths.pop(key, None)
        candidate = navigator.move(key, old, goal, speed * dt * .65)
        if np.linalg.norm(candidate - old) < 1e-5:
            # A fractional position near a doorway can invalidate its first leg.
            center = np.array(navigator.cell(old), dtype=np.float32)
            candidate = navigator.move(key, old, center, speed * dt * .65)
        return candidate


@dataclass
class LocalSearch:
    points: list[np.ndarray]
    index: int = 0
    scan_remaining: Optional[float] = None
    scan_start: float = 0.0
    scan_seconds: float = 2.4

    def step(self, pos: np.ndarray, angle: float, dt: float) -> tuple[Optional[np.ndarray], Optional[float], bool]:
        if self.index >= len(self.points):
            return None, None, True
        target = self.points[self.index]
        if np.linalg.norm(target - pos) > 0.2:
            return target, None, False
        if self.scan_remaining is None:
            self.scan_remaining = self.scan_seconds
            self.scan_start = angle
        self.scan_remaining = max(0.0, self.scan_remaining - dt)
        progress = 1.0 - self.scan_remaining / self.scan_seconds
        look = self.scan_start + 0.8 * math.sin(math.tau * progress)
        if self.scan_remaining <= 1e-6:
            self.index += 1
            self.scan_remaining = None
        return None, look, self.index >= len(self.points)
