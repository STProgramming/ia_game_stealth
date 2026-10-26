from dataclasses import dataclass, field, replace
import numpy as np

FALL_DROP_CHANCE = .25
FIREARMS = {"PISTOL", "RIFLE", "SHOTGUN", "SMG", "ONE_HAND_SMG"}
COUNTED_ITEMS = {"AMMO": "ammo", "FLASH": "flash", "SMOKE": "smoke", "GRENADE": "grenades", "SNACK": "snacks",
                 "BANDAGE": "bandages", "MEDKIT": "medkits", "CASING": "casings", "MAGAZINE": "magazines"}
UNIQUE_ITEMS = {"KNIFE": "knife", "HAZMAT": "hazmat", "SAMPLE_VIAL": "sample_vial", "SECURITY_CARD": "security_card"}


@dataclass
class Equipment:
    weapon: str | None = None
    ammo: int = 0
    knife: bool = False
    flash: int = 0
    smoke: int = 0
    bandages: int = 0
    medkits: int = 0
    casings: int = 0
    magazines: int = 0
    hazmat: bool = False
    sample_vial: bool = False
    security_card: bool = False
    weapons: set = field(default_factory=set)
    selected: str = "KNIFE"
    fire_cooldown: float = 0.0
    grenades: int = 0
    snacks: int = 0

    @property
    def armed(self):
        return self.knife or self.weapon in FIREARMS or bool(self.weapons & FIREARMS)

    def next_level_loadout(self):
        return replace(self, weapon=None if self.weapon == "PISTOL" else self.weapon,
                       ammo=0, weapons=self.weapons - {"PISTOL"}, fire_cooldown=0.,
                       selected="KNIFE" if self.selected in {"PISTOL", "AMMO"} else self.selected)

    def snapshot(self):
        return {"weapon": self.weapon, "ammo": self.ammo, "knife": self.knife,
                "flash": self.flash, "smoke": self.smoke,
                "bandages": self.bandages, "medkits": self.medkits,
                "casings": self.casings, "magazines": self.magazines,
                "weapons": sorted(self.weapons), "selected": self.selected,
                "hazmat": self.hazmat, "sample_vial": self.sample_vial,
                "security_card": self.security_card, "grenades": self.grenades, "snacks": self.snacks}

    def collect(self, kind, amount):
        if kind in FIREARMS:
            self.weapons.add(kind)
            self.selected = self.weapon = kind
            self.ammo += amount
        elif kind in COUNTED_ITEMS:
            field_name = COUNTED_ITEMS[kind]
            setattr(self, field_name, getattr(self, field_name) + amount)
        elif kind in UNIQUE_ITEMS:
            setattr(self, UNIQUE_ITEMS[kind], True)
        else:
            return False
        return True

    def drop_on_fall(self, rng, pos, pickups, drop_id, chance=FALL_DROP_CHANCE):
        kind = self.selected
        if kind in FIREARMS:
            available = kind in self.weapons or self.weapon == kind
        else:
            field_name = COUNTED_ITEMS.get(kind) or UNIQUE_ITEMS.get(kind)
            available = field_name is not None and getattr(self, field_name) > 0
        if not available or rng.random() >= chance:
            return None
        amount = 1
        if kind in FIREARMS:
            self.weapons.discard(kind)
            amount, self.ammo = self.ammo, 0
        elif kind in COUNTED_ITEMS:
            setattr(self, field_name, getattr(self, field_name) - 1)
        else:
            setattr(self, field_name, False)
        self.weapon = None
        self.selected = "KNIFE" if self.knife else "NONE"
        pickups.append({"id": f"player-drop-{drop_id}", "kind": kind, "amount": amount,
                        "pos": pos.copy(), "source": "PLAYER"})
        return kind

    def select(self, kind):
        counts = {"BANDAGE": self.bandages, "MEDKIT": self.medkits,
                  "CASING": self.casings, "MAGAZINE": self.magazines,
                  "FLASH": self.flash, "SMOKE": self.smoke, "KNIFE": self.knife,
                  "AMMO": self.ammo, "HAZMAT": self.hazmat, "SAMPLE_VIAL": self.sample_vial,
                  "SECURITY_CARD": self.security_card}
        counts.update(GRENADE=self.grenades, SNACK=self.snacks)
        if kind in self.weapons or counts.get(kind, 0) > 0:
            self.selected = kind
            self.weapon = kind if kind in FIREARMS else None
            return True
        return False

    def take_diversion(self):
        if self.selected == "CASING" and self.casings > 0:
            self.casings -= 1
            return True
        if self.selected == "MAGAZINE" and self.magazines > 0:
            self.magazines -= 1
            return True
        return False

    def medical_input(self, vitals, inp, busy=False):
        if busy:
            vitals.cancel_treatment()
            return
        if vitals.treatment is not None:
            return
        if inp.use_medkit and self.medkits > 0 and vitals.start_treatment("MEDKIT"):
            self.medkits -= 1
        elif inp.use_bandage and self.bandages > 0 and vitals.start_treatment("BANDAGE"):
            self.bandages -= 1


def map_pickups(navigator, mode="guard"):
    placements = [
        ("MAGAZINE", (13, 24), 1), ("MEDKIT", (32, 20), 1),
        ("SMOKE", (42, 9), 1), ("BANDAGE", (6, 20), 2),
        ("SNACK", (7, 22), 2),
    ] if mode == "guard" else [
        ("MEDKIT", (20, 17), 1), ("SMOKE", (24, 10), 1),
        ("BANDAGE", (18, 17), 2), ("MAGAZINE", (12, 10), 1),
        ("SNACK", (12, 11), 2),
    ]
    return [{"id": f"supply-{i}", "kind": kind,
             "pos": navigator.goal(np.array(point, dtype=np.float32)), "amount": amount,
             "source": "MAP", "fixed": kind == "MEDKIT",
             "zone": ("INFERMERIA" if kind == "MEDKIT" else "PRONTO SOCCORSO" if kind == "BANDAGE"
                      else "ARMERIA" if kind in {"PISTOL", "AMMO", "FLASH", "SMOKE", "CASING"}
                      else "MAGAZZINO")} for i, (kind, point, amount) in enumerate(placements)]
