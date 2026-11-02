"""Shared playable firearm timing, physical clues and perception, without policy changes."""
import math
import numpy as np
from .playable_firearms import Firearm, acoustic_profile, base_weapon, weapon_draw_time
from .soliton_security import NoiseEvent, can_see, line_clear, GUARD_FOV
from .doctrine import FacilityAlert


class RuntimeFirearms:
    def _reset_firearms(self):
        self.spent_cases = []
        self._pending_firearm_sounds = []
        self._firearm_id = 0
        self._enemy_guns = {}
        self._enemy_arsenals = {}
        self._enemy_primary_weapons = {}
        self._enemy_ammo_mirrors = {}
        self._enemy_lost_weapons = {}
        self._heard_firearm_events = set()
        self._firearm_dialogue = {}
        if not hasattr(self, 'env'):
            self.noise_events = []

    def _firearm_event_id(self):
        self._firearm_id += 1
        return 1000000 + self._firearm_id

    def _firearm_sound(self, pos, radius, source, recognition=0., owner='PLAYER', masked=False):
        event = NoiseEvent(self._firearm_event_id(), pos.copy(), radius, .65, source)
        event.recognition = recognition
        event.owner = owner
        event.masked = masked
        self.noise_events.append(event)

    def _discard_magazine(self, pos, rounds, owner='PLAYER'):
        ident = self._firearm_event_id()
        self.pickups.append(dict(id=f'discarded-mag-{ident}', kind='MAGAZINE', amount=1,
            pos=pos.copy(), source='MAP', lost_rounds=rounds, owner=owner))
        self._firearm_sound(pos, 3., 'MAGAZINE DROP', owner=owner)

    def _record_ballistics(self, origin, endpoint, kind, angle, hit, owner='PLAYER'):
        world, grid = self._combat_world()
        if hasattr(self, '_register_squad_pressure'):
            self._register_squad_pressure(origin, endpoint, kind, hit, owner)
        profile = acoustic_profile(kind)
        self._firearm_sound(origin, profile['radius'], profile['source'], profile['recognition'], owner)
        case_pos = self.navigator.goal(origin + np.array([math.sin(angle), -math.cos(angle)], dtype=np.float32) * .42)
        if not line_clear(grid, origin, case_pos):
            case_pos = origin.copy()
        case = dict(id=self._firearm_event_id(), pos=case_pos, owner=owner, landed=False, remaining=.18,
                    discovered=False, observers=set())
        self.spent_cases.append(case)
        masked = not profile['suppressed']
        self._pending_firearm_sounds.append(dict(delay=.18, pos=case_pos.copy(), radius=1.8,
            source='CASING DROP', owner=owner, masked=masked))
        # A trace stopping short of weapon reach hit geometry, rather than air.
        from .playable_combat import WEAPONS
        if not hit and np.linalg.norm(endpoint - origin) < WEAPONS[kind].reach - .25:
            self._pending_firearm_sounds.append(dict(delay=.04, pos=endpoint.copy(), radius=3.8,
                source='BULLET IMPACT', owner=owner, masked=masked))
        if owner == 'PLAYER':
            self.spawn_safe, self.spawn_grace = False, 0.
            if not profile['suppressed']:
                if hasattr(self, 'env'):
                    self._player_security_event('GUNFIRE', origin)
                else:
                    self._swat_security_event('GUNFIRE', origin)

    def _player_discovered(self):
        if hasattr(self, 'env'):
            return self.facility_alert in {'CONTACT', 'SEARCH'} or (self.team_contact is not None and self.team_contact_age <= 12.)
        bb = self.core.blackboard
        return self._equipment_alert == 'CONTACT' or (bb.contact_pos is not None and bb.contact_age <= 12.)

    def _step_player_firearm(self, inp):
        self.equipment.step_draw(self.game_dt)
        if hasattr(self, '_step_squad_pressure'):
            self._step_squad_pressure()
        gun = self.equipment.gun()
        if gun is not None:
            interrupted = (self.takedown or self.player_actions.knockdown_remaining > 0
                           or self.player_actions.rise_timer > 0 or self.player_actions.dive_timer > 0
                           or self.player_vitals.health <= 0)
            if interrupted:
                gun.cancel_reload()
                self.equipment.sync_ammo()
            else:
                fast = inp.fast_reload or self._player_discovered()
                if inp.reload or inp.fast_reload or (gun.reload_mode == 'TACTICAL' and fast):
                    result = self.equipment.start_reload(fast)
                    self.equipment.sync_ammo()
                    if result:
                        if result['drop']:
                            self._discard_magazine(self._combat_world()[0].player_pos, result['rounds'])
                        self.message = 'RICARICA RAPIDA' if fast else 'RICARICA TATTICA · CARICATORE CONSERVATO'
                        self._firearm_sound(self._combat_world()[0].player_pos, 1.8 if fast else .65, 'RELOAD')
                    elif not gun.reload_mode and not self.equipment.drawing:
                        self.message = 'NESSUN CARICATORE DI SCORTA' if not gun.spares else 'CARICATORE PIENO'
                finished = gun.step(self.game_dt)
                if finished and finished['empty_saved']:
                    self.equipment.magazines += 1
                self.equipment.sync_ammo()
        if not hasattr(self, 'env'):
            for sound in self.noise_events:
                sound.ttl -= self.game_dt
            self.noise_events = [sound for sound in self.noise_events if sound.ttl > 0]
        for sound in self._pending_firearm_sounds:
            sound['delay'] -= self.game_dt
            if sound['delay'] <= 1e-7:
                self._firearm_sound(sound['pos'], sound['radius'], sound['source'],
                                    owner=sound['owner'], masked=sound['masked'])
        self._pending_firearm_sounds = [s for s in self._pending_firearm_sounds if s['delay'] > 1e-7]
        for case in self.spent_cases:
            case['remaining'] = max(0., case['remaining'] - self.game_dt)
            case['landed'] = case['remaining'] <= 1e-7
        for uid, actor, vitals in self._combat_actors():
            self._step_enemy_firearms(uid, actor)
            if vitals.hit_stun_remaining <= 0 and getattr(actor, 'blinded_timer', 0.) <= 0:
                self._observe_spent_cases(uid, actor)

    def _enemy_gun(self, uid, actor):
        kind = actor.weapon
        arsenal = self._enemy_arsenals.get(uid)
        if arsenal is None:
            arsenal = {}
            self._enemy_arsenals[uid] = arsenal
            self._enemy_primary_weapons[uid] = kind
            # A sidearm is part of the SWAT loadout, with its own finite stock.
            if self._enemy_is_swat(uid, actor) and base_weapon(kind) in {'RIFLE', 'SHOTGUN', 'SMG', 'ONE_HAND_SMG'}:
                arsenal['PISTOL'] = Firearm.loaded('PISTOL', 16, [15])
        gun = arsenal.get(kind)
        if gun is None:
            lost = kind in self._enemy_lost_weapons.get(uid, set())
            gun = Firearm.loaded(kind, 0 if lost else actor.ammo,
                                 [] if lost else [self._gun_capacity(kind)] * 2)
            arsenal[kind] = gun
        mirror = self._enemy_ammo_mirrors.get(uid)
        # The public ammo field remains an adapter for the legacy simulation.
        # Compare against its last mirror, not the live chamber, so a direct
        # firearm mutation cannot conjure back rounds already fired.
        if (mirror is not None and mirror[0] == kind and actor.ammo != mirror[1]
                and kind not in self._enemy_lost_weapons.get(uid, set())):
            gun.cancel_reload()
            gun = Firearm.loaded(kind, actor.ammo, gun.spares)
            arsenal[kind] = gun
        self._enemy_guns[uid] = gun
        actor.ammo = gun.rounds
        self._enemy_ammo_mirrors[uid] = (kind, actor.ammo)
        return gun

    def _sync_enemy_ammo(self, uid, actor, gun=None):
        gun = gun or self._enemy_guns.get(uid)
        if gun is not None:
            actor.ammo = gun.rounds
            self._enemy_ammo_mirrors[uid] = (actor.weapon, actor.ammo)

    def _enemy_is_swat(self, uid, actor):
        actor_uid = getattr(actor, 'guard_id', getattr(actor, 'uid', uid))
        return not hasattr(self, 'env') or str(actor_uid).startswith('S')

    def _enemy_in_firefight(self, actor):
        if hasattr(self, 'env'):
            return bool(actor.seeing_player or actor.visual_confidence >= 1.
                        or self.team_contact is not None and self.team_contact_age <= 5.)
        return bool(actor.local_visual or self.core.blackboard.contact_pos is not None
                    and self.core.blackboard.contact_age <= 5.)

    def _enemy_drawing(self, uid, actor):
        return getattr(actor, 'enemy_draw_remaining', 0.) > 1e-7

    def _enemy_reloading(self, uid, actor):
        return bool(getattr(actor, 'weapon', None) and self._enemy_gun(uid, actor).reload_mode)

    def _enemy_switch_weapon(self, uid, actor, kind):
        if self._enemy_drawing(uid, actor) or actor.weapon == kind or getattr(actor, 'disarmed', False):
            return False
        current = self._enemy_gun(uid, actor)
        target = self._enemy_arsenals[uid].get(kind)
        if target is None or kind in self._enemy_lost_weapons.get(uid, set()):
            return False
        current.cancel_reload()
        actor.weapon = kind
        actor.ammo = target.rounds
        self._enemy_guns[uid] = target
        self._enemy_ammo_mirrors[uid] = (kind, actor.ammo)
        actor.enemy_draw_duration = actor.enemy_draw_remaining = weapon_draw_time(kind)
        actor.aim_timer = 0.
        if hasattr(actor, 'max_ammo'):
            actor.max_ammo = target.capacity + 1
        return True

    def _step_enemy_firearms(self, uid, actor):
        if not getattr(actor, 'weapon', None) or getattr(actor, 'disarmed', False):
            return
        actor.enemy_draw_remaining = max(0., getattr(actor, 'enemy_draw_remaining', 0.) - self.game_dt)
        gun = self._enemy_gun(uid, actor)
        if self._enemy_drawing(uid, actor):
            return
        if gun.reload_mode:
            gun.step(self.game_dt)
            self._sync_enemy_ammo(uid, actor, gun)
            return
        if not gun.chamber:
            self._enemy_reload(uid, actor)
            return
        if not self._enemy_is_swat(uid, actor) or self._enemy_in_firefight(actor):
            return
        # Once sheltered, service both weapons rather than remaining forever
        # on the handgun. Each reload still requires an actual spare magazine.
        primary = self._enemy_primary_weapons[uid]
        arsenal = self._enemy_arsenals[uid]
        primary_gun = arsenal.get(primary)
        sidearm = arsenal.get('PISTOL') if primary != 'PISTOL' else None
        if actor.weapon != primary and primary_gun is not None:
            if primary_gun.chamber or primary_gun.spares:
                self._enemy_switch_weapon(uid, actor, primary)
        elif sidearm is not None and not sidearm.chamber and sidearm.spares:
            self._enemy_switch_weapon(uid, actor, 'PISTOL')

    @staticmethod
    def _gun_capacity(kind):
        from .playable_firearms import CAPACITIES
        return CAPACITIES[base_weapon(kind)]

    def _enemy_reload(self, uid, actor):
        if not getattr(actor, 'weapon', None) or getattr(actor, 'disarmed', False) or self._enemy_drawing(uid, actor):
            return False
        gun = self._enemy_gun(uid, actor)
        swat = self._enemy_is_swat(uid, actor)
        if swat and self._enemy_in_firefight(actor) and not gun.chamber:
            loaded = [(kind, other) for kind, other in self._enemy_arsenals[uid].items()
                      if kind != actor.weapon and other.chamber
                      and kind not in self._enemy_lost_weapons.get(uid, set())]
            if loaded:
                # Drawing the pistol (.42 s) is quicker than discarding and
                # replacing a magazine, even during the SWAT's fast reload.
                loaded.sort(key=lambda item: weapon_draw_time(item[0]))
                if self._enemy_switch_weapon(uid, actor, loaded[0][0]):
                    return True
        if not gun.spares and not gun.reload_mode and swat:
            alternate = next((kind for kind, other in self._enemy_arsenals[uid].items()
                              if kind != actor.weapon and (other.chamber or other.spares)
                              and kind not in self._enemy_lost_weapons.get(uid, set())), None)
            if alternate is not None:
                return self._enemy_switch_weapon(uid, actor, alternate)
        result = gun.start_reload(swat)
        self._sync_enemy_ammo(uid, actor, gun)
        if result and result['drop']:
            self._discard_magazine(actor.pos, result['rounds'], uid)
        return bool(gun.reload_mode)

    def _enemy_weapon_state(self, uid, actor):
        if getattr(actor, 'weapon', None):
            self._enemy_gun(uid, actor)
        arsenal = self._enemy_arsenals.get(uid, {})
        primary = self._enemy_primary_weapons.get(uid, getattr(actor, 'weapon', None))
        return dict(draw_remaining=getattr(actor, 'enemy_draw_remaining', 0.),
                    draw_duration=getattr(actor, 'enemy_draw_duration', 0.),
                    primary_weapon=primary,
                    secondary_weapon='PISTOL' if primary != 'PISTOL' and 'PISTOL' in arsenal else None,
                    weapon_ammo={kind: gun.rounds for kind, gun in arsenal.items()},
                    weapon_reserves={kind: sum(gun.spares) for kind, gun in arsenal.items()})

    def _heard_firearm(self, uid, pos):
        sounds = [s for s in getattr(self, 'noise_events', []) if getattr(s, 'owner', None) == 'PLAYER'
                  and not getattr(s, 'masked', False) and np.linalg.norm(s.pos - pos) <= s.radius]
        if not sounds or self.spawn_protected:
            return None
        sound = min(sounds, key=lambda s: (s.source not in {'GUNSHOT','SUPPRESSED SHOT','SUBSONIC SHOT'},
                                           np.linalg.norm(s.pos - pos)))
        recognized = np.linalg.norm(sound.pos - pos) <= getattr(sound, 'recognition', 0.)
        if recognized and sound.source in {'SUPPRESSED SHOT', 'SUBSONIC SHOT'}:
            key = (uid, sound.event_id)
            if key not in self._heard_firearm_events:
                self._heard_firearm_events.add(key)
                self._ballistic_caution(uid, sound.pos, recognized=True)
        return sound.pos.copy(), sound.source

    def _ballistic_caution(self, uid, point, recognized=False):
        text = ('Colpo silenziato! Armi pronte, possibile scontro.' if recognized else
                'Bossolo a terra. Intruso armato nel settore, cautela. Cerchiamolo.')
        if hasattr(self, 'env'):
            self.security_compromised = True
            self.security_cycle.caution()
            self.threat_level = max(self.threat_level, 2 if recognized else 1)
            if self.facility_alert != 'CONTACT':
                self.team_contact_source = 'SUPPRESSED GUNFIRE' if recognized else 'SPENT CASING'
            self.env.alert = max(int(self.env.alert), int(FacilityAlert.SUSPICIOUS))
            self.radio_updates += 1
            observer = next((a for key, a, _ in self._combat_actors() if key == uid), None)
            if observer:
                observer.suspicion = max(observer.suspicion, .7 if recognized else .48)
                observer.weapon_ready = recognized
                if observer.state not in {'CHASE', 'VERIFY', 'CLEAR ROOM', 'STACK', 'BREACH WAIT'}:
                    self._start_guard_search(observer, point)
            self.guard_dialogue[uid] = text
            self._dialogue_timer = 7.
            self._update_facility_alert()
        else:
            if self._equipment_alert != 'CONTACT':
                self._equipment_alert = 'SUSPICIOUS'
                self.security_cycle.caution()
            self.core.blackboard.alert = max(self.core.blackboard.alert, FacilityAlert.SUSPICIOUS)
            self._search_anchor = self.navigator.goal(point)
            self._searched_contact = None
            self.message = text
            self._firearm_dialogue[uid] = text
        # An evidence position is never promoted to the hidden player's location.

    def _observe_spent_cases(self, uid, actor):
        if self.spawn_protected:
            return
        _, grid = self._combat_world()
        for case in self.spent_cases:
            if case['owner'] != 'PLAYER' or not case['landed'] or uid in case['observers']:
                continue
            smoke = self._smoke_blocks if hasattr(self, 'env') else self._swat_smoke_blocks
            if can_see(grid, actor.pos, actor.angle, case['pos'], 5., GUARD_FOV) and not smoke(actor.pos, case['pos']):
                case['observers'].add(uid)
                first = not case['discovered']
                case['discovered'] = True
                if first:
                    self._ballistic_caution(uid, case['pos'])

    def _swat_smoke_blocks(self, a, b):
        from .playable_smoke import smoke_obscures
        return smoke_obscures(a, b, self.core.smoke_clouds)

    def _case_snapshot(self):
        return [dict(id=c['id'], x=float(c['pos'][0]), y=float(c['pos'][1]),
                     landed=c['landed'], discovered=c['discovered']) for c in self.spent_cases]

    def _actor_firearm_snapshot(self, uid, actor):
        if not getattr(actor, 'weapon', None):
            return None
        return self._enemy_gun(uid, actor).snapshot()

    def _enemy_weapon_loot(self, uid, actor):
        if not getattr(actor, 'weapon', None):
            return None
        gun = self._enemy_gun(uid, actor)
        gun.cancel_reload()
        actor.ammo = gun.rounds
        data = gun.snapshot()
        self._enemy_ammo_mirrors[uid] = (actor.weapon, actor.ammo)
        gun.magazine, gun.chamber, gun.spares = 0, False, []
        self._enemy_lost_weapons.setdefault(uid, set()).add(actor.weapon)
        return data

    def _enemy_inventory_loot(self, uid, actor):
        if getattr(actor, 'weapon', None):
            self._enemy_gun(uid, actor)
        loot = []
        arsenal = self._enemy_arsenals.get(uid, {})
        kinds = ([actor.weapon] if getattr(actor, 'weapon', None) in arsenal else [])
        kinds += [kind for kind in arsenal if kind not in kinds]
        for kind in kinds:
            gun = arsenal[kind]
            if kind in self._enemy_lost_weapons.get(uid, set()):
                continue
            gun.cancel_reload()
            loot.append((kind, gun.rounds, gun.snapshot()))
            gun.magazine, gun.chamber, gun.spares = 0, False, []
            self._enemy_lost_weapons.setdefault(uid, set()).add(kind)
        actor.ammo = 0
        actor.enemy_draw_remaining = 0.
        if getattr(actor, 'weapon', None):
            self._enemy_ammo_mirrors[uid] = (actor.weapon, 0)
        return loot
