"""Health and fatigue for interactive play, independent of trained observations."""
from dataclasses import dataclass

SWAT_STAMINA_CAPACITY = 2.5


@dataclass
class PlayableVitals:
    health: float = 1.0
    stamina: float = 1.0
    fatigue: float = 0.0
    recovery_delay: float = 0.0
    exhausted: bool = False
    was_running: bool = False
    sprint_seconds: float = 0.0
    boost_remaining: float = 0.0
    bleeding: float = 0.0
    treatment: str | None = None
    treatment_remaining: float = 0.0
    regen_remaining: float = 0.0
    regen_seconds: float = 0.0
    smoke_damage: float = 0.0
    smoke_exposure: float = 0.0
    hit_stun_remaining: float = 0.0
    stamina_capacity: float = 1.0
    injured: bool = False
    crouch_remaining: float = 60.0
    fear: bool = False
    snack_eaten: bool = False
    calm_seconds: float = 0.0
    surrendering: bool = False

    def melee_trauma(self, damage, stamina_loss, injury_chance, fear_chance, rng):
        self.health = max(0., self.health - damage)
        self.stamina = max(0., self.stamina - stamina_loss)
        self.recovery_delay = max(self.recovery_delay, 3.)
        self.exhausted |= self.stamina <= 1e-7
        self.cancel_treatment()
        if rng.random() < injury_chance:
            self.injured = True
        if rng.random() < fear_chance:
            self.fear = True
            self.snack_eaten = False
            self.calm_seconds = 0.
            self.boost_remaining = max(self.boost_remaining, 6.)

    def step_condition(self, crouching, dt):
        if self.injured and crouching:
            self.crouch_remaining = max(0., self.crouch_remaining - dt)
        if self.fear:
            if self.health >= .999 and self.stamina >= .999 and self.snack_eaten:
                self.calm_seconds += dt
                if self.calm_seconds >= 3.:
                    self.fear = self.surrendering = False
                    self.calm_seconds = 0.
            else:
                self.calm_seconds = 0.

    def eat_snack(self):
        if self.health <= 0:
            return False
        self.snack_eaten = True
        self.stamina = min(1., self.stamina + .2)
        if self.stamina >= .35:
            self.exhausted = False
        return True

    @property
    def can_crouch(self):
        return not self.injured or self.crouch_remaining > 1e-7

    def step_smoke(self, density: float, dt: float) -> None:
        self.smoke_exposure = density
        if density <= 0 or self.health <= 0:
            return
        self.stamina = max(0., self.stamina - .35 * density * dt / self.stamina_capacity)
        self.recovery_delay = max(self.recovery_delay, 2.)
        if self.stamina <= 1e-7:
            self.exhausted = True
            damage = min(max(0., .10 - self.smoke_damage), .025 * density * dt, self.health)
            self.smoke_damage += damage
            self.health -= damage

    @property
    def can_aim(self) -> bool:
        return (self.health > 0 and self.stamina > 1e-7 and not self.exhausted
                and self.hit_stun_remaining <= 0)

    def take_hit(self, damage: float) -> None:
        if damage <= 0:
            return
        self.health = max(0.0, self.health - damage)
        self.bleeding = min(1.0, self.bleeding + damage * 2.5)
        self.cancel_treatment()

    def cancel_treatment(self) -> None:
        self.treatment = None
        self.treatment_remaining = 0.0

    def start_treatment(self, kind: str) -> bool:
        if self.health <= 0 or self.treatment is not None:
            return False
        if kind not in {"BANDAGE", "MEDKIT"}:
            return False
        if kind == "BANDAGE" and self.bleeding <= 0:
            return False
        if kind == "MEDKIT" and self.health >= 1 and self.bleeding <= 0 and not self.injured:
            return False
        self.treatment = kind
        self.treatment_remaining = 1.0 if kind == "BANDAGE" else 4.0
        return True

    def step_treatment(self, dt: float) -> None:
        if self.health <= 0:
            self.cancel_treatment()
            return
        # Existing recovery continues independently of the current application.
        if self.regen_seconds > 0:
            elapsed = min(dt, self.regen_seconds)
            healing = self.regen_remaining * elapsed / self.regen_seconds
            self.health = min(1.0, self.health + healing)
            self.regen_remaining = max(0.0, self.regen_remaining - healing)
            self.regen_seconds = max(0.0, self.regen_seconds - elapsed)
        if self.treatment is None:
            return
        elapsed = min(dt, self.treatment_remaining)
        kind = self.treatment
        if kind == "MEDKIT":
            self.health = min(1.0, self.health + .10 * elapsed / 4.0)
        self.treatment_remaining = max(0.0, self.treatment_remaining - elapsed)
        if self.treatment_remaining <= 1e-7:
            self.treat_wound()
            self.cancel_treatment()
            if kind == "MEDKIT":
                self.injured = False
                self.crouch_remaining = 60.
                self.regen_remaining += .25
                self.regen_seconds = 120.0

    def treat_wound(self) -> None:
        self.bleeding = 0.0

    def step_bleeding(self, dt: float, speed: float) -> float:
        """Return lost blood; actual movement increases loss, even during a boost."""
        if self.bleeding <= 0 or self.health <= 0:
            return 0.0
        effort = 1.0 + min(4.0, max(0.0, speed) / 2.0)
        # Light wounds leave occasional drops; below 55% loss reaches its full
        # rate. Severity remains tied to the wound, so healing cannot reopen it.
        severity = .25 + .75 * min(1.0, max(0.0, (.85 - self.health) / .30))
        loss = min(self.health, self.bleeding * .006 * severity * effort * dt)
        self.health -= loss
        self.stamina = max(0.0, self.stamina - loss * 5.0 / self.stamina_capacity)
        self.fatigue = min(.4, self.fatigue + loss * .7)
        # An untreated wound prevents instant recovery of lost vigour.
        self.recovery_delay = max(self.recovery_delay, .5)
        if self.stamina <= 1e-7:
            self.exhausted = True
        return loss

    @property
    def can_sprint(self) -> bool:
        return self.health > 0 and not self.exhausted and self.stamina > 0 and not self.injured

    @property
    def speed_multiplier(self) -> float:
        wounded = 0.7 if self.health < 0.55 else 1.0
        return (wounded * (1.0 - self.fatigue) * (0.7 if self.exhausted else 1.0)
                * (.45 if self.injured else 1.)
                * (1.15 if self.fear and self.boost_remaining > 0 else 1.))

    def step(self, sprinting: bool, dt: float, drain: float = 0.16,
             fatigue_on_pause: bool = True) -> bool:
        self.hit_stun_remaining = max(0.0, self.hit_stun_remaining - dt)
        running = sprinting and self.can_sprint
        if running:
            self.sprint_seconds += dt
            self.stamina = max(0.0, self.stamina - drain * dt / self.stamina_capacity)
            self.recovery_delay = 2.0
            if self.stamina <= 1e-7:
                self.stamina = 0.0
                self.exhausted = True
        else:
            if self.was_running and self.sprint_seconds >= 0.5 and (fatigue_on_pause or self.exhausted):
                # Each recovery cycle costs speed; sustained rest restores it.
                self.fatigue = min(0.4, self.fatigue + 0.08)
                self.sprint_seconds = 0.0
            previous_delay = self.recovery_delay
            self.recovery_delay = max(0.0, self.recovery_delay - dt)
            recovery_dt = max(0.0, dt - previous_delay)
            if recovery_dt > 0:
                self.stamina = min(1.0, self.stamina + 0.11 * recovery_dt)
                if self.stamina >= 0.35:
                    self.exhausted = False
                if self.stamina >= 0.999:
                    self.fatigue = max(0.0, self.fatigue - 0.008 * recovery_dt)
        self.was_running = running
        self.boost_remaining = max(0.0, self.boost_remaining - dt)
        return running

    @property
    def panting(self) -> bool:
        return self.health > 0 and (self.exhausted or self.stamina <= 1e-7)

    def snapshot(self) -> dict:
        return {"health": self.health, "stamina": self.stamina, "stamina_capacity": self.stamina_capacity, "fatigue": self.fatigue,
                "exhausted": self.exhausted, "panting": self.panting, "recovery_delay": self.recovery_delay,
                "speed_multiplier": self.speed_multiplier, "boost_remaining": self.boost_remaining,
                "bleeding": self.bleeding, "treatment": self.treatment,
                "treatment_remaining": self.treatment_remaining,
                "regen_remaining": self.regen_remaining, "regen_seconds": self.regen_seconds,
                "can_aim": self.can_aim, "smoke_exposure": self.smoke_exposure,
                "smoke_damage": self.smoke_damage, "hit_stun_remaining": self.hit_stun_remaining,
                "injured": self.injured, "crouch_remaining": self.crouch_remaining,
                "can_sprint": self.can_sprint, "can_crouch": self.can_crouch,
                "fear": self.fear, "snack_eaten": self.snack_eaten,
                "calm_seconds": self.calm_seconds, "surrendering": self.surrendering}
