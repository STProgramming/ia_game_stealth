"""Medical evidence and timed explosives for both interactive modes."""
from dataclasses import dataclass, field
import numpy as np
from .playable_actions import spend_energy
from .playable_combat import MELEE_PROFILES, WEAPONS, knock_down_player, trace_shot
from .soliton_security import line_clear


class MedicalSurveillance:
    def __init__(self, pickups):
        self.stations = {i['id']: i['pos'].copy() for i in pickups if i.get('fixed') and i['kind'] == 'MEDKIT'}
        self.wounded_by = set()
        self.used = set()
        self.assignments = {}
        self.discovered = set()

    def hit(self, uid):
        self.wounded_by.add(uid)

    def goal(self, uid, pos, visible, path_length):
        if visible:
            self.assignments.pop(uid, None)
            return None
        if uid not in self.wounded_by or not self.stations:
            return None
        if uid not in self.assignments:
            self.assignments[uid] = min(self.stations, key=lambda key: path_length(pos, self.stations[key]))
        return self.stations[self.assignments[uid]].copy()

    def inspect(self, uid, observes):
        key = self.assignments.get(uid)
        if key in self.used and (uid,key) not in self.discovered and observes(self.stations[key]):
            self.discovered.add((uid,key))
            self.wounded_by.discard(uid)
            self.assignments.pop(uid, None)
            return self.stations[key].copy()
        return None


@dataclass
class Explosive:
    id: int
    origin: np.ndarray
    target: np.ndarray
    owner: str
    fuse: float = 1.5
    age: float = 0.
    pos: np.ndarray = field(init=False)

    def __post_init__(self):
        self.pos = self.origin.copy()

    def step(self, dt):
        self.age += dt
        self.pos = self.origin + (self.target - self.origin) * min(1., self.age / .45)
        self.fuse = max(0., self.fuse - dt)
        return self.fuse <= 1e-7

    def snapshot(self):
        return dict(id=self.id, x=float(self.pos[0]), y=float(self.pos[1]),
                    target_x=float(self.target[0]), target_y=float(self.target[1]),
                    fuse=self.fuse, radius=3., owner=self.owner)


def blast_damage(center, pos, clear):
    distance = float(np.linalg.norm(center - pos))
    return .45 * max(0., 1. - distance / 3.) if distance < 3. and clear(center, pos) else 0.


class RuntimeCombatSystems:
    def _reset_combat_systems(self):
        self.medical_watch = MedicalSurveillance(self.pickups)
        self.explosives = []
        self.blasts = []
        self._explosive_id = 0
        self._enemy_fire_timers = {}

    def _combat_actors(self):
        if hasattr(self, 'env'):
            return [(g.guard_id, g, g.vitals) for g in self._active_guards()]
        return [(uid, u, self._unit_vitals[uid]) for uid, u in self.core.units.items()
                if u.alive and uid not in self._departed]

    def _combat_world(self):
        return (self.env, self.env.grid) if hasattr(self, 'env') else (self.core, self.core.map.grid)

    def _prepare_condition_input(self, inp):
        world, grid = self._combat_world()
        seen = [g.pos for _, g, _ in self._combat_actors()
                if np.linalg.norm(g.pos - world.player_pos) <= 12. and line_clear(grid, world.player_pos, g.pos)]
        self.player_actions.condition_input(inp, self.player_vitals, self.game_dt, self.rng, world.player_pos, seen)
        if inp.attack_held and self.player_actions.f_seconds == 0 and self.equipment.knife:
            # A fresh knife press can counter on the same frame as impact.
            if any(self._melee_attacks.get(uid) and self._melee_attacks[uid].windup > 0
                   and np.linalg.norm(g.pos - world.player_pos) <= 2.
                   and line_clear(grid, world.player_pos, g.pos) for uid, g, _ in self._combat_actors()):
                inp.lethal = True

    def _counter_disarm(self, uid, enemy, shield):
        attack = self._melee_attacks.get(uid)
        if attack is None or not attack.counterable(shield):
            return False
        weapon = getattr(enemy, 'weapon', 'PISTOL')
        if weapon:
            self.pickups.append(dict(id=f'disarm-{uid}-{len(self.pickups)}',kind=weapon,
                                     amount=enemy.ammo,pos=enemy.pos.copy(),source='MAP'))
        enemy.weapon, enemy.ammo, enemy.disarmed = None, 0, True
        attack.windup = 0.
        attack.cooldown = 1.6
        self.message = f'{uid}: DISARMATO'
        return True

    def _melee_impact(self, uid, enemy, profile):
        world, grid = self._combat_world()
        world.player_pos = knock_down_player(grid, enemy.pos, world.player_pos,
            self.player_actions, self.player_vitals, profile, self.rng)
        self.player_actions.dropped_item = self.equipment.drop_on_fall(
            self.rng, world.player_pos, self.pickups,
            getattr(world, 'steps', getattr(world, 'world_ticks', 0)), profile.drop_chance)
        if self.player_actions.dropped_item == 'SECURITY_CARD' and hasattr(self, 'campaign_items'):
            self.campaign_items.discard('SECURITY_CARD')
        self.medical_watch.hit(uid)
        world.player_velocity[:] = 0.
        if hasattr(self, 'env'):
            world.player_current_speed = 0.
        else:
            world.player_health = self.player_vitals.health
        self.takedown = None
        self.message = f'{uid}: COLPO DI SCUDO — A TERRA' if profile is MELEE_PROFILES['SHIELD'] else f'{uid}: CORPO A CORPO — A TERRA'

    def _throw_grenade(self, origin, target, owner):
        self._explosive_id += 1
        self.explosives.append(Explosive(self._explosive_id, origin.copy(), target.copy(), owner))

    def _player_extra_items(self, inp):
        if self.player_actions.movement_locked or self.player_vitals.health <= 0:
            return
        if inp.use_snack and self.equipment.snacks > 0 and self.player_vitals.eat_snack():
            self.equipment.snacks -= 1
            self.message = 'SNACK MANGIATO — RECUPERA SALUTE E VIGORE PER CALMARTI'
        if inp.use_grenade and self.equipment.grenades > 0 and spend_energy(self.player_vitals, .2):
            world, grid = self._combat_world()
            target, _ = trace_shot(grid, world.player_pos, np.array([inp.aim_x,inp.aim_y]), [], reach=8.)
            self.equipment.grenades -= 1
            self._throw_grenade(world.player_pos, target, 'PLAYER')
            if hasattr(self, 'env'):
                self._player_security_event('GUNFIRE', world.player_pos)
            else:
                self._swat_security_event('GUNFIRE', world.player_pos)

    def _enemy_grenade(self, uid, enemy, vitals):
        world, grid = self._combat_world()
        distance = float(np.linalg.norm(enemy.pos - world.player_pos))
        if (getattr(enemy, 'grenade_count', 0) <= 0 or getattr(enemy, 'disarmed', False)
                or not vitals.can_aim or not 5. <= distance <= 9.
                or any(np.linalg.norm(g.pos - world.player_pos) < 3.5 for _, g, _ in self._combat_actors())
                or not line_clear(grid, enemy.pos, world.player_pos)):
            return False
        if not spend_energy(vitals, .3):
            return False
        enemy.grenade_count -= 1
        self._throw_grenade(enemy.pos, world.player_pos, uid)
        return True

    def _enemy_shot(self, uid, enemy, vitals):
        world, grid = self._combat_world()
        profile = WEAPONS.get(getattr(enemy, 'weapon', None))
        if profile is None or enemy.ammo <= 0 or not vitals.can_aim:
            return False
        distance = float(np.linalg.norm(world.player_pos - enemy.pos))
        if distance > profile.reach or not spend_energy(vitals, profile.energy):
            return False
        delta = world.player_pos - enemy.pos
        side = np.array([-delta[1], delta[0]]) / max(distance, 1e-6)
        offset = float(self.rng.normal(0., profile.spread * distance))
        target = world.player_pos + side * offset
        endpoint, _ = trace_shot(grid, enemy.pos, target, [], reach=min(profile.reach, distance))
        hit = abs(offset) <= (.65 if enemy.weapon == 'SHOTGUN' else .4) and line_clear(grid, enemy.pos, world.player_pos)
        enemy.ammo -= 1
        if hit:
            damage = profile.damage * (max(.35, 1. - distance / 10.) if enemy.weapon == 'SHOTGUN' else 1.)
            self.player_vitals.take_hit(damage)
            self.medical_watch.hit(uid)
            if not hasattr(self, 'env'):
                world.player_health = self.player_vitals.health
        self.shots.append(dict(**{'from':dict(x=float(enemy.pos[0]),y=float(enemy.pos[1]))},
                               to=dict(x=float(endpoint[0]),y=float(endpoint[1])),ttl=.12))
        return True

    def _step_explosives(self):
        world, grid = self._combat_world()
        clear = lambda a,b: line_clear(grid,a,b)
        self.blasts = [dict(b,ttl=b['ttl']-self.game_dt) for b in self.blasts if b['ttl']>self.game_dt]
        remaining = []
        for grenade in self.explosives:
            if not grenade.step(self.game_dt):
                remaining.append(grenade)
                continue
            self.blasts.append(dict(x=float(grenade.pos[0]),y=float(grenade.pos[1]),ttl=.5,radius=3.))
            damage = blast_damage(grenade.pos, world.player_pos, clear)
            if damage:
                self.player_vitals.take_hit(damage)
                if grenade.owner != 'PLAYER':
                    self.medical_watch.hit(grenade.owner)
                if not hasattr(self, 'env'):
                    world.player_health = self.player_vitals.health
                    if world.player_health <= world.cfg.player_neutralized_health:
                        world.player_neutralized = True
            for uid, enemy, vitals in self._combat_actors():
                damage = blast_damage(grenade.pos, enemy.pos, clear)
                if not damage:
                    continue
                vitals.take_hit(damage)
                if hasattr(self, 'env'):
                    if vitals.health <= 0:
                        self._down_guard(uid, True)
                else:
                    enemy.health = vitals.health
                    if enemy.health <= 0:
                        self._down_swat_player_target(enemy, True)
            if hasattr(self, 'env'):
                self._player_security_event('GUNFIRE', grenade.pos)
            else:
                self._swat_security_event('GUNFIRE', grenade.pos)
        self.explosives = remaining
