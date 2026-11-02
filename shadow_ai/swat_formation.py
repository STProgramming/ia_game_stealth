"""Shield-led firefight movement and physical, alternating firing positions."""
import math

import numpy as np

from .playable_combat import lane_blocked
from .soliton_security import can_see, line_clear, turn_towards, GUARD_FOV, GUARD_VISION_RANGE


FORMATION_STATES = {'SHIELD ADVANCE', 'SHIELD HOLD', 'STACK BEHIND SHIELD',
                    'PEEK LEFT', 'PEEK RIGHT', 'RELOAD COVER',
                    'BREAKOUT MOVE', 'BREAKOUT FLASH', 'BREAKOUT FLUSH'}


class RuntimeSwatFormation:
    @staticmethod
    def _formation_uid(actor):
        return getattr(actor, 'guard_id', getattr(actor, 'uid', ''))

    def _formation_members(self):
        return [(uid, actor, vitals) for uid, actor, vitals in self._combat_actors()
                if not hasattr(self, 'env') or uid.startswith('S')]

    def _formation_shield(self, members=None):
        members = self._formation_members() if members is None else members
        return next((actor for _, actor, _ in members if self._formation_has_shield(actor)), None)

    def _formation_has_shield(self, actor):
        return actor.has_shield if hasattr(self, 'env') else actor.profile.has_shield

    def _formation_contact(self, actor):
        """Use a genuine visual contact or a recently communicated fixed point."""
        world, grid = self._combat_world()
        campaign = hasattr(self, 'env')
        visual = (actor.seeing_player and actor.visual_confidence >= 1.) if campaign else actor.local_visual
        shield = self._formation_has_shield(actor) if not campaign or self._formation_uid(actor).startswith('S') else False
        if (visual and not self.spawn_protected and getattr(actor, 'blinded_timer', 0.) <= 0
                and can_see(grid, actor.pos, actor.angle, world.player_pos,
                            GUARD_VISION_RANGE * (.85 if shield else 1.),
                            GUARD_FOV - (20. if shield else 0.))):
            smoke = self._smoke_blocks if campaign else self._swat_smoke_blocks
            if not smoke(actor.pos, world.player_pos):
                return world.player_pos.copy()
        point = actor.last_seen if campaign else actor.last_seen_pos
        confidence = actor.visual_confidence if campaign else actor.local_visual_confidence
        if point is not None and actor.last_seen_age <= .65 and confidence >= .75:
            return point.copy()
        if campaign:
            point, age = self.team_contact, self.team_contact_age
        else:
            point, age = self.core.blackboard.contact_pos, self.core.blackboard.contact_age
        return point.copy() if point is not None and age <= .65 else None

    def _formation_clock(self):
        ticks = self.env.steps if hasattr(self, 'env') else self.core.world_ticks
        return ticks * self.game_dt

    def _formation_set_state(self, uid, actor, state):
        actor.formation = state
        if hasattr(self, 'env'):
            actor.state = state
        else:
            self.actions[uid] = state

    def _formation_move(self, uid, actor, goal, speed, state):
        """Move the physical actor, with walls and living teammates as blockers."""
        vitals = actor.vitals if hasattr(self, 'env') else self._unit_vitals[uid]
        old = actor.pos.copy()
        goal = self.navigator.goal(goal)
        moved = self.navigator.move(uid, old, goal, speed * vitals.speed_multiplier * self.game_dt)
        flash_guard = getattr(self, '_friendly_flash_move', None)
        if flash_guard is not None:
            moved = flash_guard(actor, old, moved)
        for other_uid, other, _ in self._combat_actors():
            if other_uid == uid:
                continue
            old_distance = float(np.linalg.norm(old - other.pos))
            distance = float(np.linalg.norm(moved - other.pos))
            if distance < .48 and distance < old_distance - 1e-5:
                moved = old.copy()
                break
        actor.pos = moved
        self._formation_set_state(uid, actor, state)
        actor.target = goal.copy()
        vitals.step(False, self.game_dt, fatigue_on_pause=False,
                    moving=np.linalg.norm(moved - old) > 1e-6)
        if not hasattr(self, 'env'):
            self._unit_vitals_stepped.add(uid)
        return float(np.linalg.norm(moved - old))

    def _formation_face(self, actor, point):
        actor.angle = turn_towards(actor.angle,
            math.atan2(float(point[1] - actor.pos[1]), float(point[0] - actor.pos[0])), 4. * self.game_dt)

    def _clear_formation(self, actor):
        actor.lean_amount = 0.
        actor.formation = ''
        actor._formation_managed = False
        actor._formation_can_fire = True
        if hasattr(self, 'env') and actor.state in FORMATION_STATES:
            if actor.guard_id.startswith('G') and self._player_identified:
                self._guard_goal_from_radio(actor)
            elif actor.last_seen is not None and actor.last_seen_age <= 12.:
                self._start_guard_search(actor, actor.last_seen)
            else:
                actor.state, actor.target = 'PATROL', actor.patrol_target()

    def _step_reload_cover(self, uid, actor):
        """Normal sentries also seek cover while a finite magazine is replaced."""
        reloading = self._enemy_reloading(uid, actor)
        drawing = self._enemy_drawing(uid, actor)
        if not (reloading or drawing):
            if getattr(actor, '_reload_cover_active', False):
                self._clear_formation(actor)
            actor._reload_cover_active = False
            actor._reload_cover_goal = None
            return False
        actor._reload_cover_active = True
        actor.lean_amount = 0.
        actor._formation_managed, actor._formation_can_fire = True, False
        point = self._formation_contact(actor)
        members = self._formation_members()
        shield = self._formation_shield(members)
        if self._formation_has_shield(actor):
            self._formation_move(uid, actor, actor.pos, 0., 'SHIELD HOLD')
        else:
            goal = getattr(actor, '_reload_cover_goal', None)
            if goal is None:
                if shield is not None and shield is not actor and point is not None:
                    direction = (point - shield.pos) / max(float(np.linalg.norm(point - shield.pos)), 1e-6)
                    followers = [other for other_uid, other, _ in members if other is not shield]
                    rank = next((i for i, other in enumerate(followers) if other is actor), 0)
                    candidate = self.navigator.goal(shield.pos - direction * (1.2 + rank * .65))
                    if np.isfinite(self.navigator.path_length(actor.pos, candidate)):
                        goal = candidate
                if goal is None and point is not None:
                    goal = self._incoming_cover_goal(actor.pos, point)
                actor._reload_cover_goal = actor.pos.copy() if goal is None else goal.copy()
                goal = actor._reload_cover_goal
            self._formation_move(uid, actor, goal, 1.7, 'RELOAD COVER')
        if point is not None:
            self._formation_face(actor, point)
        actor.aim_timer = 0.
        return True

    def _step_swat_formation(self, uid, actor):
        campaign = hasattr(self, 'env')
        if campaign and not uid.startswith('S'):
            return False
        if self.spawn_protected or getattr(actor, 'blinded_timer', 0.) > 0:
            self._clear_formation(actor)
            return False
        jobs = self._medical_jobs if campaign else self._context_medical_jobs
        if uid in jobs or any(job.get('helper') == uid for job in jobs.values()):
            self._clear_formation(actor)
            return False
        if campaign:
            if actor.state in {'WITHDRAW', 'LEAVING', 'EXITED', 'WAIT GUARDS'}:
                self._clear_formation(actor)
                return False
            plan = self._room_entries.get(uid)
            if plan is not None and plan['phase'] in {'STACK', 'BREACH WAIT'}:
                self._clear_formation(actor)
                return False
        elif uid in self._withdrawals or uid in self._departed:
            self._clear_formation(actor)
            return False
        if self._step_reload_cover(uid, actor):
            return True
        if uid in getattr(self, '_squad_breakouts', {}) and self._step_squad_breakout(uid, actor):
            actor.lean_amount = 0.
            actor._formation_managed, actor._formation_can_fire = True, True
            return True
        members = self._formation_members()
        shield = self._formation_shield(members)
        point = self._formation_contact(actor)
        if point is None:
            point = next((point for _, other, _ in members
                          if (point := self._formation_contact(other)) is not None), None)
        if shield is None or len(members) < 2 or point is None or np.linalg.norm(shield.pos - point) > 12.:
            self._clear_formation(actor)
            return False
        if getattr(self, '_step_squad_breakout', None) is not None and self._step_squad_breakout(uid, actor):
            actor.lean_amount = 0.
            actor._formation_managed = True
            actor._formation_can_fire = True
            return True
        actor._formation_managed = True
        actor._formation_can_fire = False
        actor.lean_amount = 0.
        direction = point - shield.pos
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        side = np.array([-direction[1], direction[0]], dtype=np.float32)
        followers = [(key, other) for key, other, _ in members if other is not shield]
        followers.sort(key=lambda pair: pair[0])
        if actor is shield:
            following = all(np.linalg.norm(other.pos - shield.pos) <= 4. for _, other in followers)
            fire_ready = (not self._enemy_reloading(uid, actor) and not self._enemy_drawing(uid, actor)
                          and getattr(actor, 'suppression_timer', 0.) <= 0)
            held = not fire_ready or not following
            distance = float(np.linalg.norm(point - actor.pos))
            goal = actor.pos.copy() if held or distance <= 3.2 else actor.pos + direction * min(1., distance - 3.2)
            state = 'SHIELD HOLD' if np.linalg.norm(goal - actor.pos) < .05 else 'SHIELD ADVANCE'
            self._formation_move(uid, actor, goal, .58, state)
            actor._formation_can_fire = fire_ready
        else:
            rank = next(i for i, (key, _) in enumerate(followers) if key == uid)
            selected = int(self._formation_clock() / 1.8) % len(followers) == rank
            sign = 1. if rank % 2 == 0 else -1.
            rear = shield.pos - direction * (1. + rank * .68)
            goal = self.navigator.goal(rear + side * sign * 1.15) if selected else self.navigator.goal(rear)
            if (not np.isfinite(self.navigator.path_length(actor.pos, goal))
                    or selected and not line_clear(self._combat_world()[1], goal, point)):
                selected = False
                goal = self.navigator.goal(rear)
            state = ('PEEK LEFT' if sign > 0 else 'PEEK RIGHT') if selected else 'STACK BEHIND SHIELD'
            self._formation_move(uid, actor, goal, 1.5, state)
            allies = [other.pos for key, other, _ in self._combat_actors() if key != uid]
            ready = selected and np.linalg.norm(actor.pos - goal) < .28
            ready = ready and line_clear(self._combat_world()[1], actor.pos, point) and not lane_blocked(actor.pos, point, allies)
            actor._formation_can_fire = bool(ready)
            actor.lean_amount = sign if ready else 0.
        self._formation_face(actor, point)
        return True

    def _swat_formation_can_fire(self, uid, actor):
        if self._enemy_reloading(uid, actor) or self._enemy_drawing(uid, actor):
            return False
        if getattr(actor, '_reload_cover_active', False):
            self._clear_formation(actor)
            actor._reload_cover_active = False
        if getattr(actor, 'suppression_timer', 0.) > 0 and self._formation_has_shield(actor):
            return False
        return not getattr(actor, '_formation_managed', False) or getattr(actor, '_formation_can_fire', False)
