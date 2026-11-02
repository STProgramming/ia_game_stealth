"""Visible engagement by sentries while the response team pursues contacts."""
import math

import numpy as np

from .playable_combat import WEAPONS, firing_position, lane_blocked
from .soliton_security import line_clear, turn_towards


class RuntimeRegularEngagement:
    def _step_regular_engagement(self, guard):
        if (guard.guard_id.startswith('S') or not guard.seeing_player
                or guard.visual_confidence < 1. or guard.disarmed):
            return False
        navigation = self.navigator
        order = self._room_orders.get(guard.guard_id)
        if order is not None and order['evacuated']:
            self.navigator = self._outside_room_navigation
        try:
            return self._regular_engagement_move(guard)
        finally:
            self.navigator = navigation

    def _regular_engagement_move(self, guard):
        # A live contact supersedes a sentry's surveillance/optional policy order.
        self._cancel_guard_care(guard.guard_id)
        player = self.env.player_pos
        allies = [g.pos for g in self._active_guards() if g is not guard]
        distance = float(np.linalg.norm(player - guard.pos))
        blocked = lane_blocked(guard.pos, player, allies)
        profile = WEAPONS.get(guard.weapon)
        reach = profile.reach if profile else 2.
        goal = guard.pos.copy()
        if blocked:
            cached = getattr(guard, '_engagement_goal', None)
            if (cached is not None and np.linalg.norm(cached['contact'] - player) < 1.5
                    and line_clear(self.env.grid, cached['goal'], player)
                    and not lane_blocked(cached['goal'], player, allies)
                    and all(np.linalg.norm(cached['goal'] - p) >= .7 for p in allies)):
                goal = cached['goal'].copy()
            else:
                alternate = firing_position(self.navigator, guard.pos, player, allies,
                                            int(guard.guard_id[1:]))
                if alternate is not None:
                    goal = alternate
            guard.state = 'REPOSITION'
        elif distance > min(6., reach * .72):
            toward = (player - guard.pos) / max(distance, 1e-6)
            candidate = self.navigator.goal(player - toward * min(4., reach * .55))
            if all(np.linalg.norm(candidate - p) >= .7 for p in allies):
                goal = candidate
            guard.state = 'CHASE'
        elif distance < 2.2 and self.equipment.armed:
            alternate = firing_position(self.navigator, guard.pos, player, allies,
                                        int(guard.guard_id[1:]))
            if alternate is not None:
                goal = alternate
            guard.state = 'REPOSITION'
        else:
            guard.state = 'CHASE'
            if not self.equipment.armed:
                goal = player.copy()
        guard._engagement_goal = dict(contact=player.copy(), goal=goal.copy())
        guard.target = goal.copy()
        guard.fire_goal = goal.copy() if blocked else None
        guard.pos, _ = self._guard_move(guard, goal,
            float(self.env.cfg.guard_speed) / self.game_dt, 4.)
        heading = math.atan2(float(player[1] - guard.pos[1]), float(player[0] - guard.pos[0]))
        guard.angle = turn_towards(guard.angle, heading, 4. * self.game_dt)
        return True
