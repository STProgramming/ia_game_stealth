"""Playable reactions to incoming fire, based on the shot's recorded origin."""
import math
import numpy as np

from .soliton_security import can_see, line_clear, GUARD_FOV, GUARD_VISION_RANGE


class RuntimeHitReactions:
    def _incoming_hit_context(self, enemy, vitals, origin):
        _, grid = self._combat_world()
        shield = enemy.has_shield if hasattr(self, 'env') else enemy.profile.has_shield
        visible = (getattr(enemy, 'blinded_timer', 0.) <= 0
                   and can_see(grid, enemy.pos, enemy.angle, origin,
                               GUARD_VISION_RANGE * (.85 if shield else 1.),
                               GUARD_FOV - (20. if shield else 0.)))
        if hasattr(self, 'env'):
            visible = visible and not self._smoke_blocks(enemy.pos, origin)
        else:
            from .playable_smoke import smoke_obscures
            visible = visible and not smoke_obscures(enemy.pos, origin, self.core.smoke_clouds)
        surprised = (vitals.hit_stun_remaining > 0
                     or getattr(enemy, 'surprised_remaining', 0.) > 0
                     or getattr(vitals, 'surprised_remaining', 0.) > 0)
        return not visible and not surprised

    def _incoming_cover_goal(self, pos, source):
        """Prefer nearby occluding geometry; retreat away when in open ground."""
        _, grid = self._combat_world()
        away = pos - source
        away /= max(float(np.linalg.norm(away)), 1e-6)
        if np.linalg.norm(away) < .1:
            away = np.array([1., 0.], dtype=np.float32)
        side = np.array([-away[1], away[0]], dtype=np.float32)
        candidates = []
        for distance in (2., 3.5, 5.):
            for lateral in (0., -1., 1., -2., 2.):
                point = self.navigator.goal(pos + away * distance + side * lateral)
                travel = float(np.linalg.norm(point - pos))
                if travel < .5 or np.linalg.norm(point - source) <= np.linalg.norm(pos - source) + .2:
                    continue
                hidden = not line_clear(grid, source, point)
                candidates.append((not hidden, travel, point))
        candidates.sort(key=lambda entry: entry[:2])
        for _, _, point in candidates:
            if self.navigator.path_length(pos, point) <= 10.:
                return point.copy()
        return pos.copy()

    def _react_to_incoming_hit(self, uid, enemy, vitals, origin, eligible):
        if not eligible or vitals.health <= 0:
            return
        source = np.asarray(origin, dtype=np.float32).copy()
        # A wound supplies a fixed incoming-fire estimate, never a moving target.
        if hasattr(self, 'env'):
            if hasattr(self, '_cancel_guard_care'):
                self._cancel_guard_care(uid)
            self._player_security_event('GUNFIRE', source)
            self._start_swat_response(uid)
            enemy.last_seen, enemy.last_seen_age = source.copy(), 0.
            self._searches.pop(uid, None)
        else:
            getattr(self, '_context_medical_jobs', {}).pop(uid, None)
            vitals.cancel_treatment()
            self._swat_security_event('GUNFIRE', source)
            enemy.last_seen_pos, enemy.last_seen_age = source.copy(), 0.
        enemy._incoming_cover = dict(source=source, goal=self._incoming_cover_goal(enemy.pos, source),
                                     remaining=6., holding=1.2)
        enemy.aim_timer = 0.
        if hasattr(enemy, 'fire_goal'):
            enemy.fire_goal = None
        if hasattr(self, 'env'):
            enemy.state, enemy.target = 'TAKE COVER', enemy._incoming_cover['goal'].copy()
            self._dialogue_timer = 7.
        self._radio(uid, 'SHOOT FIRE, SHOOT FIRE')

    def _retaliation_payload(self, enemy, source, origin):
        if getattr(enemy, 'disarmed', False) or getattr(enemy, 'blinded_timer', 0.) > 0:
            return None
        _, grid = self._combat_world()
        delta = source - origin
        length = float(np.linalg.norm(delta))
        distance = min(9., length)
        if distance < 2.:
            return None
        target = origin + delta / length * distance
        if not line_clear(grid, origin, target):
            return None
        # Keep fragmentation away from the thrower and nearby teammates.
        for field, kind in (('grenade_count', 'GRENADE'), ('flash_count', 'FLASH'), ('smoke_count', 'SMOKE')):
            if getattr(enemy, field, 0) <= 0:
                continue
            if kind == 'GRENADE' and (distance < 3.2 or any(np.linalg.norm(actor.pos - target) < 3.2
                    for _, actor, _ in self._combat_actors())):
                continue
            return field, kind, target
        return None

    def _retaliation_grenade(self, uid, enemy, source):
        payload = self._retaliation_payload(enemy, source, enemy.pos)
        if payload is not None:
            field, kind, target = payload
            setattr(enemy, field, getattr(enemy, field) - 1)
            self._throw_grenade(enemy.pos, target, uid, kind=kind)
            return True
        return False

    def _retaliation_throw_goal(self, enemy, source):
        """Peek at a reachable nearby corner instead of throwing into cover."""
        if not any(getattr(enemy, field, 0) for field in ('grenade_count', 'flash_count', 'smoke_count')):
            return None
        x, y = self.navigator.cell(enemy.pos)
        candidates = []
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if not self.navigator.free(x + dx, y + dy):
                    continue
                point = np.array([x + dx, y + dy], dtype=np.float32)
                distance = float(np.linalg.norm(point - enemy.pos))
                if not .4 <= distance <= 3.2:
                    continue
                if self._retaliation_payload(enemy, source, point) is not None:
                    candidates.append((distance, point))
        candidates.sort(key=lambda item: item[0])
        for _, point in candidates:
            if self.navigator.path_length(enemy.pos, point) <= 3.5:
                return point
        return None

    def _step_incoming_cover(self, uid, enemy, vitals):
        response = getattr(enemy, '_incoming_cover', None)
        if response is None:
            return False
        response['remaining'] = max(0., response['remaining'] - self.game_dt)
        reached = np.linalg.norm(enemy.pos - response['goal']) <= .3
        if response.get('peek_attempted') and not response.get('retaliated'):
            # A corner's last fraction of a tile can still occlude the throw.
            reached = self._retaliation_payload(enemy, response['source'], enemy.pos) is not None
        if not reached and response['remaining'] > 0:
            if hasattr(self, 'env'):
                enemy.state, enemy.target = 'TAKE COVER', response['goal'].copy()
                enemy.pos, enemy.angle = self._guard_move(enemy, response['goal'],
                    float(self.env.cfg.guard_speed) / self.game_dt * 2.4, 5.)
            else:
                self.actions[uid] = 'TAKE COVER'
                self._move_swat_unit(enemy, response['goal'])
            return True
        if not response.get('retaliated'):
            if not hasattr(self, 'env') or uid.startswith('S'):
                thrown = self._retaliation_grenade(uid, enemy, response['source'])
                if not thrown and not response.get('peek_attempted'):
                    response['peek_attempted'] = True
                    peek = self._retaliation_throw_goal(enemy, response['source'])
                    if peek is not None:
                        response['goal'], response['remaining'] = peek, 5.
                        return True
            response['retaliated'] = True
        response['holding'] = max(0., response['holding'] - self.game_dt)
        source = response['source']
        enemy.angle = math.atan2(float(source[1] - enemy.pos[1]), float(source[0] - enemy.pos[0]))
        if response['holding'] > 0:
            vitals.step(False, self.game_dt, fatigue_on_pause=False)
            if hasattr(self, '_unit_vitals_stepped'):
                self._unit_vitals_stepped.add(uid)
            return True
        enemy._incoming_cover = None
        if hasattr(self, 'env'):
            self._start_guard_search(enemy, source)
        return False
