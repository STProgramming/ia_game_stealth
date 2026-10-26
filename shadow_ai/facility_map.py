
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import heapq
import math
from typing import Iterable, Optional

import numpy as np


class MapFeature(IntEnum):
    OPEN = 0
    CORNER = 1
    DOOR = 2
    ROOM_ENTRY = 3
    INTERSECTION = 4
    DEAD_END = 5
    LONG_CORRIDOR = 6
    COVER_POINT = 7


@dataclass(frozen=True)
class SemanticPoint:
    feature_id: int
    feature: MapFeature
    position: tuple[float, float]
    sector_id: int


class FacilityMap:
    """
    Shared deterministic map/navigation layer for V7 SWAT.

    The map matches the normal-guard training arena but adds semantic points.
    Semantic features are world geometry only; they contain no hidden player
    information.
    """

    def __init__(self, width: int = 32, height: int = 20) -> None:
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.uint8)
        self._build()
        self.access_points = [
            np.array([30.0, 1.0], dtype=np.float32),
            np.array([30.0, 18.0], dtype=np.float32),
            np.array([1.0, 18.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
        ]
        self.door_cells = {
            (7, 6), (7, 7),
            (13, 6), (13, 7),
            (20, 8), (20, 9),
            (26, 7), (26, 8),
        }
        self.semantic_points = self._build_semantics()

    def _build(self) -> None:
        self.grid[2:10, 7] = 1
        self.grid[0:6, 13] = 1
        self.grid[9:17, 13] = 1
        self.grid[3:12, 20] = 1
        self.grid[0:7, 26] = 1
        self.grid[10:20, 26] = 1
        self.grid[12, 4:12] = 1
        self.grid[12, 15:23] = 1

        for y, x in [
            (6, 7), (7, 7),
            (6, 13), (7, 13),
            (8, 20), (9, 20),
            (7, 26), (8, 26),
        ]:
            if 0 <= y < self.height and 0 <= x < self.width:
                self.grid[y, x] = 0

    def cell(self, pos: np.ndarray) -> tuple[int, int]:
        x = int(np.clip(round(float(pos[0])), 0, self.width - 1))
        y = int(np.clip(round(float(pos[1])), 0, self.height - 1))
        return x, y

    def free(self, x: int, y: int) -> bool:
        return (
            0 <= x < self.width
            and 0 <= y < self.height
            and self.grid[y, x] == 0
        )

    def nearest_free_cell(self, x: int, y: int) -> tuple[int, int]:
        if self.free(x, y):
            return x, y
        for radius in range(1, max(self.width, self.height)):
            for oy in range(-radius, radius + 1):
                for ox in range(-radius, radius + 1):
                    nx, ny = x + ox, y + oy
                    if self.free(nx, ny):
                        return nx, ny
        return 1, 1

    def line_clear(
        self,
        a: np.ndarray,
        b: np.ndarray,
        smoke_clouds: Optional[Iterable] = None,
    ) -> bool:
        distance = float(np.linalg.norm(b - a))
        samples = max(2, int(distance * 6.0))
        smoke = list(smoke_clouds or [])
        for t in np.linspace(0.0, 1.0, samples, endpoint=True)[1:-1]:
            p = a + (b - a) * t
            x, y = self.cell(p)
            if not self.free(x, y):
                return False
            for cloud in smoke:
                if getattr(cloud, "ttl", 0.0) > 0.0:
                    center = np.asarray(cloud.center, dtype=np.float32)
                    if float(np.linalg.norm(p - center)) <= float(cloud.radius):
                        return False
        return True

    def astar(self, start: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
        sx, sy = self.cell(start)
        gx, gy = self.cell(goal)
        if not self.free(gx, gy):
            gx, gy = self.nearest_free_cell(gx, gy)

        gvec = np.array([gx, gy], dtype=np.float32)
        if self.line_clear(start, gvec):
            return [gvec]

        start_key = (sx, sy)
        goal_key = (gx, gy)
        frontier = [(0.0, start_key)]
        came = {start_key: None}
        cost = {start_key: 0.0}
        neighbors = [
            (1, 0, 1.0), (-1, 0, 1.0),
            (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (1, -1, 1.414),
            (-1, 1, 1.414), (-1, -1, 1.414),
        ]

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_key:
                break
            for dx, dy, step_cost in neighbors:
                nx, ny = current[0] + dx, current[1] + dy
                if not self.free(nx, ny):
                    continue
                if dx and dy:
                    if not self.free(current[0] + dx, current[1]):
                        continue
                    if not self.free(current[0], current[1] + dy):
                        continue
                nxt = (nx, ny)
                new_cost = cost[current] + step_cost
                if nxt not in cost or new_cost < cost[nxt]:
                    cost[nxt] = new_cost
                    priority = new_cost + math.hypot(nx - gx, ny - gy)
                    heapq.heappush(frontier, (priority, nxt))
                    came[nxt] = current

        if goal_key not in came:
            return [gvec]

        rev = []
        cur = goal_key
        while cur is not None:
            rev.append(cur)
            cur = came[cur]
        rev.reverse()
        return [
            np.array([x, y], dtype=np.float32)
            for x, y in rev[1:]
        ]

    def move_towards(
        self,
        pos: np.ndarray,
        target: np.ndarray,
        speed: float,
        dt: float,
    ) -> np.ndarray:
        delta = target - pos
        dist = float(np.linalg.norm(delta))
        if dist <= 1e-7:
            return pos.copy()
        step = min(dist, speed * dt)
        intended = pos + delta / dist * step
        if self.line_clear(pos, intended):
            return intended.astype(np.float32)

        path = self.astar(pos, target)
        if not path:
            return pos.copy()
        waypoint = path[0]
        delta = waypoint - pos
        dist = float(np.linalg.norm(delta))
        if dist <= 1e-7:
            return pos.copy()
        step = min(dist, speed * dt)
        return (pos + delta / dist * step).astype(np.float32)

    def _free_cardinal(self, x: int, y: int) -> list[tuple[int, int]]:
        return [
            (dx, dy)
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]
            if self.free(x + dx, y + dy)
        ]

    def _sector_for(self, x: int, y: int) -> int:
        col = 0 if x < self.width / 2 else 1
        row = 0 if y < self.height / 2 else 1
        return row * 2 + col

    def _build_semantics(self) -> list[SemanticPoint]:
        points: list[SemanticPoint] = []
        seen: set[tuple[MapFeature, int, int]] = set()

        def add(feature: MapFeature, x: int, y: int):
            key = (feature, x, y)
            if key in seen or not self.free(x, y):
                return
            seen.add(key)
            points.append(
                SemanticPoint(
                    feature_id=len(points),
                    feature=feature,
                    position=(float(x), float(y)),
                    sector_id=self._sector_for(x, y),
                )
            )

        for x, y in self.door_cells:
            add(MapFeature.DOOR, x, y)
            add(MapFeature.ROOM_ENTRY, x, y)

        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if not self.free(x, y):
                    continue

                card = self._free_cardinal(x, y)
                n = len(card)

                if n >= 3:
                    add(MapFeature.INTERSECTION, x, y)
                if n == 1:
                    add(MapFeature.DEAD_END, x, y)

                dirs = set(card)
                straight = (
                    dirs == {(1,0),(-1,0)}
                    or dirs == {(0,1),(0,-1)}
                )
                if straight:
                    add(MapFeature.LONG_CORRIDOR, x, y)

                blocked_x = (
                    not self.free(x - 1, y)
                    or not self.free(x + 1, y)
                )
                blocked_y = (
                    not self.free(x, y - 1)
                    or not self.free(x, y + 1)
                )
                if blocked_x and blocked_y and n >= 2:
                    add(MapFeature.CORNER, x, y)

                adjacent_wall = any(
                    not self.free(x + dx, y + dy)
                    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]
                )
                if adjacent_wall and n >= 2 and (x + y) % 3 == 0:
                    add(MapFeature.COVER_POINT, x, y)

        return points

    def nearest_feature(
        self,
        pos: np.ndarray,
        allowed: Optional[set[MapFeature]] = None,
        exclude_ids: Optional[set[int]] = None,
    ) -> Optional[SemanticPoint]:
        excluded = exclude_ids or set()
        candidates = [
            p for p in self.semantic_points
            if p.feature_id not in excluded
            and (allowed is None or p.feature in allowed)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda p: float(
                np.linalg.norm(
                    np.asarray(p.position, dtype=np.float32) - pos
                )
            ),
        )
