"""Shield suppression and a finite flash breakout after a visible retreat."""
import math
import numpy as np

from .playable_firearms import base_weapon
from .soliton_security import line_clear, turn_towards, can_see, GUARD_FOV, GUARD_VISION_RANGE


AUTOMATIC_WEAPONS = {'RIFLE', 'SMG', 'ONE_HAND_SMG'}


class RuntimeSquadPressure:
    def _friendly_flash_move(self, actor, old, candidate):
        """Keep teammates outside the announced blast until the flash goes off."""
        for grenade in self.explosives:
            if grenade.kind != 'FLASH' or grenade.owner == 'PLAYER' or grenade.fuse <= 0:
                continue
            before = float(np.linalg.norm(old - grenade.target))
            after = float(np.linalg.norm(candidate - grenade.target))
            if after < 3.25 and after < before - 1e-6:
                return old.copy()
        return candidate

    def _reset_squad_pressure(self):
        self._squad_clock = 0.
        self._shield_burst_hits = {}
        self._squad_breakouts = {}
        self._breakout_next = 0.

    def _squad_is_shield(self, actor):
        return bool(actor.has_shield if hasattr(self, 'env') else actor.profile.has_shield)

    def _pressure_members(self):
        return [(uid, actor, vitals) for uid, actor, vitals in self._combat_actors()
                if not hasattr(self, 'env') or uid.startswith('S')]

    def _step_squad_pressure(self):
        self._squad_clock += self.game_dt
        alive = set()
        for uid, actor, _ in self._combat_actors():
            alive.add(uid)
            actor.suppression_timer = max(0., getattr(actor, 'suppression_timer', 0.) - self.game_dt)
        for uid in list(self._shield_burst_hits):
            if uid not in alive:
                self._shield_burst_hits.pop(uid, None)
            else:
                self._shield_burst_hits[uid] = [stamp for stamp in self._shield_burst_hits[uid]
                                               if self._squad_clock - stamp <= .65]
        for uid, job in list(self._squad_breakouts.items()):
            if uid not in alive or self._squad_clock >= job['expires']:
                del self._squad_breakouts[uid]

    def _register_squad_pressure(self, origin, endpoint, kind, hit, owner='PLAYER'):
        if owner != 'PLAYER' or base_weapon(kind) not in AUTOMATIC_WEAPONS:
            return
        segment = endpoint - origin
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-7:
            return
        for uid, actor, _ in self._pressure_members():
            if not self._squad_is_shield(actor):
                continue
            to_source = origin - actor.pos
            distance = float(np.linalg.norm(to_source))
            if distance < 1e-7:
                continue
            facing = np.array([math.cos(actor.angle), math.sin(actor.angle)])
            if np.dot(facing, to_source / distance) < .5:
                continue
            along = float(np.dot(actor.pos - origin, segment) / length_sq)
            nearest = origin + segment * np.clip(along, 0., 1.)
            if not -.01 <= along <= 1.08 or np.linalg.norm(actor.pos - nearest) > .68:
                continue
            times = [stamp for stamp in self._shield_burst_hits.get(uid, [])
                     if self._squad_clock - stamp <= .65]
            times.append(self._squad_clock)
            self._shield_burst_hits[uid] = times
            if len(times) >= 3:
                was_suppressed = getattr(actor, 'suppression_timer', 0.) > 0
                actor.suppression_timer = 1.0
                actor.aim_timer = 0.
                if not was_suppressed:
                    self._radio(uid, 'Raffica sullo scudo! Mi riparo, copritemi.')

    def _squad_weapon_ready(self, uid, actor):
        if self._squad_is_shield(actor) and getattr(actor, 'suppression_timer', 0.) > 0:
            actor.aim_timer = 0.
            return False
        return self._squad_clock >= getattr(actor, '_squad_flash_lock', 0.)

    def _visible_escape(self, members):
        """Use velocity and orientation only while a teammate sees the player."""
        if self.spawn_protected:
            return None
        world, grid = self._combat_world()
        velocity = world.player_velocity.copy()
        speed = float(np.linalg.norm(velocity))
        if speed < .6:
            return None
        for _, observer, _ in members:
            if getattr(observer, 'blinded_timer', 0.) > 0:
                continue
            visible = (observer.seeing_player and observer.visual_confidence >= 1.
                       if hasattr(self, 'env') else observer.local_visual)
            shield = self._squad_is_shield(observer)
            if not visible or not can_see(grid, observer.pos, observer.angle, world.player_pos,
                    GUARD_VISION_RANGE * (.85 if shield else 1.), GUARD_FOV - (20. if shield else 0.)):
                continue
            smoke = self._smoke_blocks if hasattr(self, 'env') else self._swat_smoke_blocks
            if smoke(observer.pos, world.player_pos):
                continue
            away = world.player_pos - observer.pos
            distance = float(np.linalg.norm(away))
            if not 3.5 <= distance <= 11.5:
                continue
            direction = away / distance
            facing = np.array([math.cos(self.player_actions.angle), math.sin(self.player_actions.angle)])
            if np.dot(velocity / speed, direction) <= .55 or np.dot(facing, direction) <= .35:
                continue
            target = self.navigator.goal(world.player_pos + velocity * .25)
            # Freeze this observation; the breakout must not track a hidden player.
            return target.copy()
        return None

    def _begin_squad_breakout(self, members):
        if self._squad_breakouts or self._squad_clock < self._breakout_next or len(members) < 2:
            return
        shield = next((actor for _, actor, _ in members if self._squad_is_shield(actor)), None)
        if shield is None:
            return
        target = self._visible_escape(members)
        if target is None:
            return
        _, grid = self._combat_world()
        candidates = [(uid, actor, vitals) for uid, actor, vitals in members
                      if not self._squad_is_shield(actor) and getattr(actor, 'flash_count', 0) > 0
                      and vitals.can_aim and not getattr(actor, 'disarmed', False)
                      and getattr(actor, 'blinded_timer', 0.) <= 0
                      and not self._enemy_reloading(uid, actor) and not self._enemy_drawing(uid, actor)]
        for index, (uid, actor, _) in enumerate(candidates):
            toward = target - shield.pos
            toward /= max(float(np.linalg.norm(toward)), 1e-6)
            side = np.array([-toward[1], toward[0]], dtype=np.float32)
            for sign in ((1., -1.) if index % 2 == 0 else (-1., 1.)):
                goal = self.navigator.goal(shield.pos + side * sign * 1.5 - toward * .15)
                if (np.linalg.norm(goal - shield.pos) < 1. or not line_clear(grid, goal, target)
                        or not np.isfinite(self.navigator.path_length(actor.pos, goal))):
                    continue
                if not self._safe_flash_breakout(actor, goal, target):
                    continue
                self._squad_breakouts[uid] = dict(goal=goal.copy(), target=target.copy(),
                    expires=self._squad_clock + 8., phase='MOVE', throw_time=None)
                self._breakout_next = self._squad_clock + 7.
                self._radio(uid, 'Il bersaglio si ritira! Esco dal riparo, preparo una stordente.')
                return

    def _safe_flash_breakout(self, actor, origin, target):
        _, grid = self._combat_world()
        reach = float(np.linalg.norm(target - origin))
        if not 3.5 <= reach <= 9. or not line_clear(grid, origin, target):
            return False
        return all(np.linalg.norm(other.pos - target) >= 4.2 for _, other, _ in self._combat_actors()
                   if other is not actor)

    def _step_squad_breakout(self, uid, actor):
        members = self._pressure_members()
        self._begin_squad_breakout(members)
        job = self._squad_breakouts.get(uid)
        if job is None:
            return False
        if (self._squad_clock >= job['expires'] or self._enemy_reloading(uid, actor)
                or self._enemy_drawing(uid, actor) or getattr(actor, 'blinded_timer', 0.) > 0):
            self._squad_breakouts.pop(uid, None)
            return False
        if job['phase'] == 'MOVE':
            self._formation_move(uid, actor, job['goal'], 1.45, 'BREAKOUT MOVE')
            if np.linalg.norm(actor.pos - job['goal']) <= .32:
                if getattr(actor, 'flash_count', 0) <= 0 or not self._safe_flash_breakout(actor, actor.pos, job['target']):
                    self._squad_breakouts.pop(uid, None)
                    return False
                actor.flash_count -= 1
                actor._squad_flash_lock = self._squad_clock + .35
                self._throw_grenade(actor.pos, job['target'], uid, kind='FLASH')
                job['phase'], job['throw_time'] = 'WAIT', self._squad_clock
                self._radio(uid, 'Stordente lanciata! Attendete l’effetto, poi avanti.')
            return True
        if job['phase'] == 'WAIT' and self._squad_clock - job['throw_time'] < 1.65:
            actor.formation, actor.lean_amount = 'BREAKOUT FLASH', 0.
            point = job['target']
            actor.angle = turn_towards(actor.angle, math.atan2(float(point[1] - actor.pos[1]),
                float(point[0] - actor.pos[0])), 3. * self.game_dt)
            return True
        job['phase'] = 'FLUSH'
        self._formation_move(uid, actor, job['target'], 1.6, 'BREAKOUT FLUSH')
        if self._squad_clock - job['throw_time'] >= 3.3 or np.linalg.norm(actor.pos - job['target']) < .7:
            self._squad_breakouts.pop(uid, None)
        return True
