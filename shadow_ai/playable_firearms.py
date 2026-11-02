"""Runtime-only chamber, detachable magazines and fictional sound balancing."""
from dataclasses import dataclass, field


CAPACITIES = {"PISTOL": 15, "RIFLE": 30, "SHOTGUN": 8, "SMG": 30, "ONE_HAND_SMG": 20}
VARIANTS = {f"{base}_{suffix}": base for base in ("PISTOL", "RIFLE", "SMG")
            for suffix in ("SUPPRESSED", "SUBSONIC")}
FIREARM_KINDS = set(CAPACITIES) | set(VARIANTS)
WEAPON_MASSES = {'PISTOL':1.0,'RIFLE':3.7,'SHOTGUN':4.1,'SMG':2.8,'ONE_HAND_SMG':1.7,'KNIFE':.3}


def weapon_mass(kind):
    mass = WEAPON_MASSES.get(base_weapon(kind),0.)
    return mass + (.3 if kind in VARIANTS else 0.)


def weapon_movement_factor(kind):
    return 1. / (1. + .14 * max(0.,weapon_mass(kind)-.3))


def weapon_draw_time(kind):
    return weapon_mass(kind) * .42


def base_weapon(kind):
    return VARIANTS.get(kind, kind)


def acoustic_profile(kind):
    # Game-world units, deliberately distinct from real-world acoustic data.
    if kind.endswith("_SUBSONIC"):
        return dict(source="SUBSONIC SHOT", radius=1.6, recognition=.8, suppressed=True)
    if kind.endswith("_SUPPRESSED"):
        return dict(source="SUPPRESSED SHOT", radius=4.5, recognition=2.2, suppressed=True)
    return dict(source="GUNSHOT", radius=80., recognition=80., suppressed=False)


@dataclass
class Firearm:
    kind: str
    magazine: int = 0
    chamber: bool = False
    spares: list[int] = field(default_factory=list)
    reload_mode: str = ""
    reload_remaining: float = 0.
    reload_duration: float = 0.
    incoming: int = 0
    outgoing: int = 0
    discarded: bool = False

    @property
    def capacity(self):
        return CAPACITIES[base_weapon(self.kind)]

    @property
    def rounds(self):
        return self.magazine + int(self.chamber)

    @classmethod
    def loaded(cls, kind, rounds, spares=()):
        gun = cls(kind, spares=list(spares))
        rounds = max(0, int(rounds))
        gun.chamber = rounds > 0
        gun.magazine = min(gun.capacity, max(0, rounds - int(gun.chamber)))
        gun.add_reserve(max(0, rounds - gun.rounds))
        return gun

    def add_reserve(self, rounds):
        rounds = max(0, int(rounds))
        while rounds:
            amount = min(self.capacity, rounds)
            self.spares.append(amount)
            rounds -= amount

    @classmethod
    def from_snapshot(cls, kind, data, rounds=0):
        if not data:
            return cls.loaded(kind,rounds)
        gun=cls(kind)
        gun.magazine=max(0,min(gun.capacity,int(data.get('magazine',0))))
        gun.chamber=bool(data.get('chamber',0))
        gun.spares=[max(0,int(n)) for n in data.get('spare_magazines',[]) if n>0]
        gun.add_reserve(data.get('incoming_magazine',0))
        gun.add_reserve(data.get('retained_magazine',0))
        return gun

    def fire(self):
        if self.reload_mode or not self.chamber:
            return False
        self.chamber = self.magazine > 0
        if self.magazine:
            self.magazine -= 1
        return True

    def start_reload(self, fast=False):
        if self.reload_mode:
            if fast and self.reload_mode == "TACTICAL":
                self.reload_mode = "FAST"
                elapsed = self.reload_duration - self.reload_remaining
                self.reload_duration = 1.05
                self.reload_remaining = max(.15, self.reload_duration - elapsed)
                self.discarded = True
                return dict(drop=True, rounds=self.outgoing)
            return None
        if not self.spares or self.magazine >= self.capacity:
            return None
        self.incoming = max(self.spares)
        self.spares.remove(self.incoming)
        self.outgoing, self.magazine = self.magazine, 0
        self.reload_mode = "FAST" if fast else "TACTICAL"
        self.reload_duration = self.reload_remaining = 1.05 if fast else 2.8
        self.discarded = fast
        return dict(drop=fast, rounds=self.outgoing)

    def step(self, dt):
        if not self.reload_mode:
            return None
        self.reload_remaining = max(0., self.reload_remaining - dt)
        if self.reload_remaining > 1e-7:
            return None
        empty_saved = not self.discarded and self.outgoing == 0
        if not self.discarded and self.outgoing:
            self.spares.append(self.outgoing)
        self.magazine = self.incoming
        if not self.chamber and self.magazine:
            self.chamber = True
            self.magazine -= 1
        self.reload_mode = ""
        self.reload_remaining = self.reload_duration = 0.
        self.incoming = self.outgoing = 0
        self.discarded = False
        return dict(empty_saved=empty_saved)

    def cancel_reload(self):
        if self.reload_mode:
            self.spares.append(self.incoming)
            self.magazine = 0 if self.discarded else self.outgoing
            self.reload_mode = ""
            self.reload_remaining = self.incoming = self.outgoing = 0
            self.reload_duration = 0.
            self.discarded = False

    def snapshot(self):
        return dict(magazine=self.magazine, chamber=int(self.chamber), capacity=self.capacity,
                    spare_magazines=list(self.spares), reserve=sum(self.spares),
                    reload_mode=self.reload_mode, reload_remaining=self.reload_remaining,
                    reload_duration=self.reload_duration, suppressed=acoustic_profile(self.kind)["suppressed"],
                    incoming_magazine=self.incoming, retained_magazine=self.outgoing if self.reload_mode and not self.discarded else 0,
                    subsonic=self.kind.endswith("_SUBSONIC"))
