"""Playable weapon profiles, melee timing and collision geometry."""
import numpy as np
from dataclasses import dataclass
from .soliton_security import line_clear
from .playable_actions import spend_energy


MELEE_RANGE = 1.35
MELEE_ENERGY_COST = .4
PLAYER_SPEED_SCALE = .94


@dataclass(frozen=True)
class MeleeProfile:
    windup: float
    damage: float
    stamina_loss: float
    injury_chance: float
    fear_chance: float = 0.
    drop_chance: float = .25
    energy: float = MELEE_ENERGY_COST


MELEE_PROFILES = {
    "GUARD": MeleeProfile(.3, .12, .25, .15),
    "SWAT": MeleeProfile(.4, .22, .4, .3, 0., .35, .5),
    "SHIELD": MeleeProfile(.75, .32, .6, .55, .6, .5, .65),
}


@dataclass(frozen=True)
class WeaponProfile:
    damage: float
    cooldown: float
    energy: float
    reach: float
    spread: float


WEAPONS = {
    "PISTOL": WeaponProfile(.12, .85, .08, 12., .035),
    "RIFLE": WeaponProfile(.16, .35, .10, 15., .025),
    "SHOTGUN": WeaponProfile(.30, 1.1, .16, 7., .08),
    "SMG": WeaponProfile(.075, .16, .045, 11., .055),
    "ONE_HAND_SMG": WeaponProfile(.07, .19, .05, 9., .17),
}


def swat_loadout(role, index=0):
    if role == "SHIELD":
        return "PISTOL" if index % 2 == 0 else "ONE_HAND_SMG"
    return "SHOTGUN" if role in {"ASSAULT", "HEAVY"} else "SMG"


def player_shot(grid, origin, aim, actors, weapon, rng):
    profile = WEAPONS[weapon]
    if weapon in {'SHOTGUN', 'SMG', 'ONE_HAND_SMG'}:
        delta = aim-origin
        distance = max(float(np.linalg.norm(delta)), 1e-6)
        side = np.array([-delta[1],delta[0]])/distance
        aim = aim + side * rng.normal(0.,profile.spread*distance)
    return trace_shot(grid,origin,aim,actors,reach=profile.reach)


@dataclass
class EnemyMelee:
    windup: float = 0.
    cooldown: float = 0.

    def step(self, eligible, vitals, dt, profile=None):
        profile = profile or MELEE_PROFILES["GUARD"]
        self.cooldown = max(0., self.cooldown - dt)
        if not eligible or vitals.health <= 0 or vitals.hit_stun_remaining > 0:
            self.windup = 0.
            return False
        if self.cooldown > 0:
            return False
        if self.windup <= 0:
            if not vitals.can_aim or not spend_energy(vitals, profile.energy):
                return False
            self.windup = profile.windup
            return False
        self.windup = max(0., self.windup - dt)
        if self.windup > 1e-7:
            return False
        self.windup = 0.
        self.cooldown = 1.6
        return True

    def counterable(self, shield=False):
        return not shield and self.windup > 0


def knock_down_player(grid, origin, pos, actions, vitals, profile=None, rng=None):
    delta = pos - origin
    direction = delta / np.linalg.norm(delta) if np.linalg.norm(delta) > 1e-6 else np.array([1., 0.])
    goal = pos.copy()
    for distance in np.linspace(.1, 1.2, 12):
        candidate = pos + direction * distance
        if not line_clear(grid, pos, candidate):
            break
        goal = candidate
    if profile is None:
        vitals.health = max(0., vitals.health - .08)
    else:
        vitals.melee_trauma(profile.damage, profile.stamina_loss, profile.injury_chance,
                           profile.fear_chance, rng)
    actions.knock_down(vitals)
    return goal.astype(np.float32)


def lane_blocked(origin, target, allies, radius=0.5):
    delta = target - origin
    length_sq = float(np.dot(delta, delta))
    if length_sq < 1e-6:
        return False
    for pos in allies:
        t = float(np.dot(pos - origin, delta) / length_sq)
        if 0.02 < t < 0.98 and np.linalg.norm(pos - (origin + delta * t)) <= radius:
            return True
    return False


def firing_position(navigator, origin, target, allies, index=0):
    delta = target - origin
    direction = delta / max(float(np.linalg.norm(delta)), 1e-6)
    side = np.array([-direction[1], direction[0]], dtype=np.float32)
    sign = 1 if index % 2 == 0 else -1
    # Alternate preferred sides to spread squad members across the approach.
    for distance in (1.5, 2.5, 3.5):
        for flank in (sign, -sign):
            point = navigator.goal(origin + side * distance * flank - direction * 0.5)
            if (np.linalg.norm(point - origin) >= 0.5
                    and line_clear(navigator.grid, point, target)
                    and not lane_blocked(point, target, allies)
                    and not any(np.linalg.norm(point - p) < 0.7 for p in allies)
                    and navigator.path_length(origin, point) <= 10.0):
                return point
    return None


def trace_shot(grid, origin, aim, actors, reach=12.0):
    delta = aim - origin
    distance = float(np.linalg.norm(delta))
    if distance < .1:
        return origin.copy(), None
    direction = delta / distance
    endpoint = origin.copy()
    for travel in np.linspace(0, reach, 97):
        candidate = origin + direction * travel
        if not line_clear(grid, origin, candidate):
            break
        endpoint = candidate
    hits = []
    for actor in actors:
        travel = float(np.dot(actor.pos - origin, direction))
        if (0 < travel <= np.linalg.norm(endpoint - origin)
                and np.linalg.norm(actor.pos - (origin + direction * travel)) < .45):
            hits.append((travel, actor))
    if hits:
        actor = min(hits, key=lambda entry: entry[0])[1]
        return actor.pos.copy(), actor
    return endpoint, None
