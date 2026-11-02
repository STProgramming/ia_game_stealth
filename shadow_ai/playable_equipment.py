from dataclasses import dataclass, field, replace
import numpy as np
from copy import deepcopy
from .playable_firearms import Firearm, FIREARM_KINDS, weapon_mass, weapon_draw_time, weapon_movement_factor

FALL_DROP_CHANCE = .25
FIREARMS = FIREARM_KINDS
COUNTED_ITEMS = {"AMMO": "ammo", "FLASH": "flash", "SMOKE": "smoke", "GRENADE": "grenades", "SNACK": "snacks",
                 "BANDAGE": "bandages", "MEDKIT": "medkits", "CASING": "casings", "MAGAZINE": "magazines"}
UNIQUE_ITEMS = {"KNIFE": "knife", "HAZMAT": "hazmat", "SAMPLE_VIAL": "sample_vial", "SECURITY_CARD": "security_card"}
BACKPACK_CAPACITY = 10
ITEM_MASSES = {"FLASH": .35, "SMOKE": .4, "GRENADE": .45, "SNACK": .15,
               "BANDAGE": .1, "MEDKIT": .8, "CASING": .005, "MAGAZINE": .1,
               "HAZMAT": 1.2, "SAMPLE_VIAL": .1, "SECURITY_CARD": .02}


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
    firearms: dict = field(default_factory=dict)
    unique_spares: dict = field(default_factory=dict)
    loose_ammo: int = 0
    draw_remaining: float = 0.
    draw_duration: float = 0.
    _ammo_mirror: int | None = field(default=None,repr=False)
    _ammo_owner: str | None = field(default=None,repr=False)

    def __post_init__(self):
        if self.weapon not in FIREARMS and self.ammo>0:
            self.loose_ammo += self.ammo
            self.ammo=0

    def gun(self):
        if self.weapon not in FIREARMS:
            return None
        gun = self.firearms.get(self.weapon)
        external_edit = self._ammo_owner == self.weapon and self._ammo_mirror is not None and self.ammo != self._ammo_mirror
        if gun is None or external_edit:
            if gun:
                gun.cancel_reload()
            gun = Firearm.loaded(self.weapon, self.ammo, gun.spares if gun else ())
            self.firearms[self.weapon] = gun
        self.sync_ammo()
        return gun

    def sync_ammo(self):
        if self.weapon in self.firearms:
            self.ammo = self.firearms[self.weapon].rounds
        elif self.weapon not in FIREARMS:
            self.ammo=0
        self._ammo_owner,self._ammo_mirror=self.weapon,self.ammo

    @property
    def drawing(self):
        return self.draw_remaining>1e-7

    @property
    def movement_factor(self):
        return weapon_movement_factor(self.weapon or self.selected)

    def item_quantity(self, kind):
        if kind in UNIQUE_ITEMS:
            return 1 + self.unique_spares.get(kind, 0) if getattr(self, UNIQUE_ITEMS[kind]) else 0
        if kind == 'AMMO':
            return self.loose_ammo
        return getattr(self, COUNTED_ITEMS[kind], 0) if kind in COUNTED_ITEMS else 0

    def consume_item(self, kind, amount=1):
        if kind not in UNIQUE_ITEMS or amount <= 0 or self.item_quantity(kind) < amount:
            return False
        remaining = self.item_quantity(kind) - amount
        setattr(self, UNIQUE_ITEMS[kind], remaining > 0)
        if remaining > 1:
            self.unique_spares[kind] = remaining - 1
        else:
            self.unique_spares.pop(kind, None)
        return True

    @property
    def inventory_kinds(self):
        kinds = set(self.weapons)
        if self.weapon in FIREARMS:
            kinds.add(self.weapon)
        kinds.update(kind for kind, name in UNIQUE_ITEMS.items() if getattr(self, name))
        kinds.update(kind for kind, name in COUNTED_ITEMS.items()
                     if kind != 'AMMO' and getattr(self, name) > 0)
        if self.loose_ammo > 0:
            kinds.add('AMMO')
        return kinds

    @property
    def slot_count(self):
        return len(self.inventory_kinds)

    @property
    def total_weight(self):
        self.gun()
        weight = sum(weapon_mass(kind) for kind in self.inventory_kinds if kind in FIREARMS)
        weight += weapon_mass('KNIFE') * self.item_quantity('KNIFE')
        rounds = self.loose_ammo + sum(gun.rounds + sum(gun.spares) + gun.incoming +
            (gun.outgoing if gun.reload_mode and not gun.discarded else 0) for gun in self.firearms.values())
        weight += rounds * .012
        weight += sum(mass * self.item_quantity(kind)
                      for kind, mass in ITEM_MASSES.items())
        return weight

    @property
    def backpack_movement_factor(self):
        carried = max(0., self.total_weight - weapon_mass(self.weapon or self.selected))
        return max(.45, 1. / (1. + .035 * carried))

    def step_draw(self,dt):
        self.draw_remaining=max(0.,self.draw_remaining-dt)

    def start_reload(self,fast=False):
        gun=self.gun()
        if gun is None or self.drawing:
            return None
        if not gun.reload_mode and gun.magazine<gun.capacity and self.loose_ammo>0:
            packed=min(gun.capacity,self.loose_ammo)
            if not gun.spares or packed>max(gun.spares):
                gun.add_reserve(packed)
                self.loose_ammo-=packed
        result=gun.start_reload(fast)
        self.sync_ammo()
        return result

    @property
    def ammunition_stock(self):
        return self.loose_ammo + sum(sum(gun.spares) for gun in self.firearms.values())

    def fire(self):
        gun = self.gun()
        fired = gun is not None and not self.drawing and gun.fire()
        self.sync_ammo()
        return fired

    @property
    def armed(self):
        return self.knife or self.weapon in FIREARMS or bool(self.weapons & FIREARMS)

    def next_level_loadout(self):
        self.gun()
        retained=deepcopy({kind:gun for kind,gun in self.firearms.items() if kind!='PISTOL'})
        for gun in retained.values(): gun.cancel_reload()
        weapon=None if self.weapon=='PISTOL' else self.weapon
        return replace(self, weapon=None if self.weapon == "PISTOL" else self.weapon,
                       ammo=retained[weapon].rounds if weapon in retained else 0, weapons=self.weapons - {"PISTOL"}, fire_cooldown=0.,
                       firearms=retained, unique_spares=dict(self.unique_spares), draw_remaining=0.,draw_duration=0.,_ammo_owner=None,_ammo_mirror=None,
                       selected="KNIFE" if self.selected in {"PISTOL", "AMMO"} else self.selected)

    def snapshot(self):
        gun = self.gun()
        firearm=gun.snapshot() if gun else None
        if firearm:
            firearm['shared_reserve']=self.loose_ammo
            firearm['reserve']+=self.loose_ammo
        return {"weapon": self.weapon, "ammo": self.ammo, "knife": self.knife,
                "flash": self.flash, "smoke": self.smoke,
                "bandages": self.bandages, "medkits": self.medkits,
                "casings": self.casings, "magazines": self.magazines,
                "weapons": sorted(self.weapons), "selected": self.selected,
                "firearm": firearm, "loose_ammo": self.loose_ammo,
                "draw_remaining":self.draw_remaining,"draw_duration":self.draw_duration,
                "weapon_weight":weapon_mass(self.weapon or self.selected),"weapon_speed_multiplier":self.movement_factor,
                "inventory_slots":self.slot_count,"inventory_capacity":BACKPACK_CAPACITY,
                "item_quantities":{kind:self.item_quantity(kind) for kind in UNIQUE_ITEMS},
                "backpack_weight":self.total_weight,"backpack_speed_multiplier":self.backpack_movement_factor,
                "ammo_stock": self.ammunition_stock,
                "weapon_ammo": {kind: self.firearms[kind].rounds if kind in self.firearms else 0 for kind in sorted(self.weapons)},
                "hazmat": self.hazmat, "sample_vial": self.sample_vial,
                "security_card": self.security_card, "grenades": self.grenades, "snacks": self.snacks}

    def collect(self, kind, amount, firearm=None):
        if kind not in FIREARMS and kind not in COUNTED_ITEMS and kind not in UNIQUE_ITEMS:
            return False
        if kind not in self.inventory_kinds and self.slot_count >= BACKPACK_CAPACITY:
            return False
        if kind in FIREARMS:
            self.gun()
            incoming=Firearm.from_snapshot(kind,firearm,amount)
            existing = self.firearms.get(kind)
            if existing is not None:
                existing.add_reserve(incoming.rounds+sum(incoming.spares))
            else:
                existing = incoming
                self.firearms[kind] = existing
            self.weapons.add(kind)
            if self.weapon is None and self.selected in {'NONE','KNIFE'}:
                self.select(kind)
            self.sync_ammo()
        elif kind == 'AMMO':
            self.loose_ammo += max(0,int(amount))
        elif kind in COUNTED_ITEMS:
            field_name = COUNTED_ITEMS[kind]
            setattr(self, field_name, getattr(self, field_name) + amount)
        elif kind in UNIQUE_ITEMS:
            amount = max(0, int(amount))
            if amount == 0:
                return False
            count = self.item_quantity(kind) + amount
            setattr(self, UNIQUE_ITEMS[kind], True)
            if count > 1:
                self.unique_spares[kind] = count - 1
            else:
                self.unique_spares.pop(kind, None)
        else:
            return False
        return True

    def drop_selected(self, pos, pickups, drop_id, whole_stack=True):
        kind = self.selected
        field_name = COUNTED_ITEMS.get(kind) or UNIQUE_ITEMS.get(kind)
        if kind in FIREARMS:
            available = kind in self.weapons or self.weapon == kind
        elif kind == 'AMMO':
            available = self.loose_ammo > 0
        else:
            available = field_name is not None and getattr(self, field_name) > 0
        if not available:
            return None
        amount = 1
        firearm = None
        if kind in FIREARMS:
            self.weapons.discard(kind)
            gun = self.gun() if self.weapon == kind else self.firearms.get(kind)
            if gun:
                gun.cancel_reload()
            amount = gun.rounds if gun else (self.ammo if self.weapon == kind else 0)
            firearm = self.firearms.pop(kind, None)
            if firearm:
                amount += sum(firearm.spares)
        elif kind == 'AMMO':
            amount = self.loose_ammo if whole_stack else 1
            self.loose_ammo -= amount
        elif kind in COUNTED_ITEMS:
            amount = getattr(self, field_name) if whole_stack else 1
            setattr(self, field_name, getattr(self, field_name) - amount)
        else:
            amount = self.item_quantity(kind) if whole_stack else 1
            self.consume_item(kind, amount)
        self.weapon = None
        self.selected = "KNIFE" if self.knife else "NONE"
        self.draw_remaining=self.draw_duration=0.
        self.sync_ammo()
        pickups.append({"id": f"player-drop-{drop_id}", "kind": kind, "amount": amount,
                        "pos": pos.copy(), "source": "PLAYER",
                        **({'firearm': firearm.snapshot()} if kind in FIREARMS and firearm else {})})
        return kind

    def drop_on_fall(self, rng, pos, pickups, drop_id, chance=FALL_DROP_CHANCE):
        if rng.random() >= chance:
            return None
        return self.drop_selected(pos, pickups, drop_id, whole_stack=False)

    def select(self, kind):
        counts = {"BANDAGE": self.bandages, "MEDKIT": self.medkits,
                  "CASING": self.casings, "MAGAZINE": self.magazines,
                  "FLASH": self.flash, "SMOKE": self.smoke, "KNIFE": self.knife,
                  "AMMO": self.loose_ammo, "HAZMAT": self.hazmat, "SAMPLE_VIAL": self.sample_vial,
                  "SECURITY_CARD": self.security_card}
        counts.update(GRENADE=self.grenades, SNACK=self.snacks)
        if kind in self.weapons or counts.get(kind, 0) > 0:
            target_weapon=kind if kind in FIREARMS else None
            if self.selected==kind and self.weapon==target_weapon:
                return True
            gun = self.gun()
            if gun and kind != self.weapon:
                gun.cancel_reload()
                self.sync_ammo()
            self.selected = kind
            self.weapon = target_weapon
            if self.weapon and self.weapon not in self.firearms:
                self.firearms[self.weapon]=Firearm.loaded(self.weapon,0)
            self.draw_remaining=self.draw_duration=weapon_draw_time(kind) if kind in FIREARMS or kind=='KNIFE' else 0.
            self.sync_ammo()
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
    if mode == 'swat':
        placements += [('PISTOL_SUBSONIC',(12,11),16), ('AMMO',(12,12),30)]
    return [{"id": f"supply-{i}", "kind": kind,
             "pos": navigator.goal(np.array(point, dtype=np.float32)), "amount": amount,
             "source": "MAP", "fixed": kind == "MEDKIT",
             "zone": ("INFERMERIA" if kind == "MEDKIT" else "PRONTO SOCCORSO" if kind == "BANDAGE"
                      else "ARMERIA" if kind in {"PISTOL", "AMMO", "FLASH", "SMOKE", "CASING"}
                      else "MAGAZZINO")} for i, (kind, point, amount) in enumerate(placements)]
