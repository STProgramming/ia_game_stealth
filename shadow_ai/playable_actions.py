"""Player input timing and physical actions, shared by both playable modes."""
import math
import numpy as np
from .soliton_security import line_clear
from .playable_equipment import FIREARMS


class PlayerActions:
    def __init__(self):
        self.angle = 0.0
        self.f_seconds = 0.0
        self.e_seconds = 0.0
        self.melee_cooldown = 0.0
        self.knife_timer = 0.0
        self.knife_angle = 0.0
        self.knife_impact = False
        self.dive_cooldown = 0.0
        self.dive_timer = 0.0
        self.prone = False
        self.knockdown_remaining = 0.0
        self.dropped_item = None
        self.rise_timer = 0.0
        self._rising = False
        self._rise_requested = False
        self.crouched = False
        self.station_id = None
        self.panic_timer = 0.

    def condition_input(self, inp, vitals, dt, rng, pos, enemies=()):
        if vitals.fear:
            inp.left, inp.right = inp.right, inp.left
            inp.up, inp.down = inp.down, inp.up
            self.panic_timer -= dt
            if self.panic_timer <= 0:
                self.panic_timer = 1.5
                roll = rng.random()
                if roll < .02 and enemies:
                    vitals.surrendering = True
                elif roll < .14:
                    if roll < .06:
                        inp.fire = True
                    elif roll < .10:
                        inp.dive = True
                    else:
                        inp.crouch = not inp.crouch
            if vitals.surrendering and enemies:
                target = min(enemies, key=lambda p: np.linalg.norm(p - pos))
                delta = target - pos
                near = np.linalg.norm(delta) <= 1.
                inp.left, inp.right = not near and delta[0] < -.15, not near and delta[0] > .15
                inp.up, inp.down = not near and delta[1] < -.15, not near and delta[1] > .15
                inp.compliant = near
                inp.sprint = inp.dive = inp.fire = inp.lethal = inp.choke = inp.attack_held = False
        inp.crouch &= vitals.can_crouch
        vitals.step_condition(inp.crouch, dt)
        inp.crouch &= vitals.can_crouch
        if vitals.injured:
            inp.sprint = inp.dive = False

    def interaction(self, inp, pos, dt, equipment, vitals, items, clear, busy=False, near_object=False):
        if inp.selected_item:
            equipment.select(inp.selected_item)
        held = inp.interact_held
        if held:
            self.e_seconds += dt
            station = next((i for i in items if i.get("fixed") and
                            np.linalg.norm(i["pos"] - pos) <= 1.2 and clear(pos, i["pos"])), None)
            if self.station_id is not None and (busy or station is None or station["id"] != self.station_id):
                vitals.cancel_treatment()
                self.station_id = None
            if self.e_seconds >= .25 and not busy and station is not None and self.station_id is None:
                if vitals.start_treatment("MEDKIT"):
                    self.station_id = station["id"]
        else:
            if self.station_id is not None:
                vitals.cancel_treatment()
                self.station_id = None
            if 0 < self.e_seconds < .25 or inp.interact:
                inp.interact = True
                nearby = near_object or any(not i.get("fixed") and np.linalg.norm(i["pos"] - pos) <= 1.2
                             and clear(pos, i["pos"]) for i in items)
                if not nearby:
                    selected = equipment.selected
                    inp.use_bandage = selected == "BANDAGE"
                    inp.use_medkit = selected == "MEDKIT"
                    inp.use_flash = selected == "FLASH"
                    inp.use_smoke = selected == "SMOKE"
                    inp.use_grenade = selected == "GRENADE"
                    inp.use_snack = selected == "SNACK"
                    inp.fire = selected in FIREARMS
            self.e_seconds = 0.

    def treatment_step(self, vitals, items, dt):
        finished = vitals.treatment == "MEDKIT" and vitals.treatment_remaining <= dt + 1e-7
        vitals.step_treatment(dt)
        if self.station_id is not None and finished and vitals.health > 0:
            items[:] = [i for i in items if i["id"] != self.station_id]
            return self.station_id
        return None

    def step(self, inp, pos, dt):
        self.crouched = inp.crouch
        direction = inp.direction()
        self.dive_timer = max(0., self.dive_timer - dt)
        self.knockdown_remaining = max(0., self.knockdown_remaining - dt)
        self.rise_timer = max(0., self.rise_timer - dt)
        # Finish the transition before handling new input; repeated requests
        # must never restart the get-up timer on its final frame.
        if self._rising and self.rise_timer == 0:
            self.prone = False
            self._rise_requested = False
        self._rising = self.rise_timer > 0
        if self.prone:
            self._rise_requested |= inp.dive or np.linalg.norm(direction) > .1
            inp.dive = False
            if self._rise_requested and self.dive_timer == 0 and self.knockdown_remaining == 0 and not self._rising:
                self.rise_timer = .65
                self._rising = True
        if not self.movement_locked and np.linalg.norm(direction) > .1:
            self.angle = math.atan2(direction[1], direction[0])
        self.melee_cooldown = max(0., self.melee_cooldown - dt)
        self.knife_timer = max(0., self.knife_timer - dt)
        self.dive_cooldown = max(0., self.dive_cooldown - dt)
        # A tap attacks on release; a hold never also performs a lethal attack.
        if inp.attack_held:
            self.f_seconds += dt
            if self.f_seconds >= .22:
                inp.choke = True
        else:
            if 0 < self.f_seconds < .22:
                inp.lethal = True
            self.f_seconds = 0.

    def dive(self, inp, pos, grid, vitals):
        if not inp.dive or vitals.injured or self.movement_locked or self.dive_cooldown > 0 or not spend_energy(vitals, .28):
            return pos
        direction = inp.direction()
        if np.linalg.norm(direction) < .1:
            direction = np.array([math.cos(self.angle), math.sin(self.angle)])
        goal = pos.copy()
        for travel in np.linspace(.1, 3.2, 33):
            candidate = pos + direction * travel
            if not line_clear(grid, pos, candidate):
                break
            goal = candidate
        self.dive_cooldown = 1.2
        self.dive_timer = .35
        self.prone = True
        self._rise_requested = False
        vitals.cancel_treatment()
        return goal.astype(np.float32)

    def knife_swing(self, pos, grid, vitals, target=None):
        if self.movement_locked or self.melee_cooldown > 0 or not spend_energy(vitals, .22):
            return None
        self.melee_cooldown = .6
        self.knife_timer = .3
        self.knife_angle = self.angle
        self.knife_impact = target is not None
        endpoint = pos.copy()
        if target is not None:
            delta = target - pos
            self.knife_angle = math.atan2(float(delta[1]), float(delta[0]))
            endpoint = target.copy()
        else:
            direction = np.array([math.cos(self.angle), math.sin(self.angle)], dtype=np.float32)
            for travel in np.linspace(.1, 2., 20):
                candidate = pos + direction * travel
                if not line_clear(grid, pos, candidate):
                    self.knife_impact = True
                    break
                endpoint = candidate
        return {"pos": endpoint, "impact": self.knife_impact}

    def knock_down(self, vitals):
        self.dropped_item = None
        self.prone = True
        self.knockdown_remaining = .75
        self.dive_timer = 0.
        self.rise_timer = 0.
        self._rising = False
        self._rise_requested = True
        self.knife_timer = 0.
        self.f_seconds = 0.
        vitals.hit_stun_remaining = max(vitals.hit_stun_remaining, .75 + .65)
        vitals.cancel_treatment()

    @property
    def movement_locked(self):
        return self.prone or self.rise_timer > 0

    def snapshot(self):
        return {"angle": self.angle, "diving": self.dive_timer > 0,
                "knockdown_remaining": self.knockdown_remaining,
                "dropped_item": self.dropped_item,
                "prone": self.prone, "rising": self.rise_timer > 0,
                "rise_remaining": self.rise_timer, "tactical_suit": True,
                "knife_remaining": self.knife_timer, "knife_angle": self.knife_angle,
                "knife_impact": self.knife_impact,
                "dive_cooldown": self.dive_cooldown,
                "station_id": self.station_id}


def spend_energy(vitals, cost):
    cost /= vitals.stamina_capacity
    if vitals.health <= 0 or vitals.exhausted or vitals.stamina + 1e-7 < cost:
        return False
    vitals.stamina = max(0., vitals.stamina - cost)
    vitals.recovery_delay = max(vitals.recovery_delay, 2.)
    if vitals.stamina <= 1e-7:
        vitals.exhausted = True
    return True


def projectile_damage(health, armor, vitals, shield, angle, pos, origin, weapon):
    delta = origin - pos
    facing = np.array([math.cos(angle), math.sin(angle)])
    frontal = np.dot(facing, delta / max(float(np.linalg.norm(delta)), 1e-6)) >= .5
    if shield and frontal:
        vitals.stamina = max(0., vitals.stamina - (.30 if weapon == "RIFLE" else .20) / vitals.stamina_capacity)
        vitals.recovery_delay = 2.
        if vitals.stamina <= 1e-7:
            vitals.exhausted = True
        return health, armor, "SHIELD"
    raw = .34 if weapon == "RIFLE" else .42
    if weapon in {'SHOTGUN', 'SMG', 'ONE_HAND_SMG'}:
        raw = {'SHOTGUN': .6, 'SMG': .18, 'ONE_HAND_SMG': .15}[weapon]
        if weapon == 'SHOTGUN':
            raw *= max(.25, 1. - float(np.linalg.norm(pos-origin))/10.)
    if armor > 0:
        reduction = .32 if weapon == "RIFLE" else .16
        absorbed = min(armor, reduction)
        armor = max(0., armor - absorbed)
        raw = min(raw, raw * (.14 if weapon == "RIFLE" else .07) + raw * (1 - absorbed / reduction))
    return max(0., health - raw), armor, "ARMOR" if armor > 0 else "HIT"
