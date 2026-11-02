"""Playable security countdowns and a persistent, bounded radio transcript."""
from collections import deque
from dataclasses import dataclass
from .doctrine import FacilityAlert


@dataclass
class SecurityCycle:
    phase: str = 'CALM'
    remaining: float = 0.
    duration: float = 0.
    rate: float = 1.

    def contact(self):
        self.phase, self.remaining, self.duration = 'CONTACT', 99., 99.

    def caution(self):
        if self.phase in {'CALM', 'SUSPICIOUS'}:
            self.phase, self.remaining, self.duration = 'SUSPICIOUS', 180., 180.

    def step(self, dt, visible=False):
        previous = self.phase
        if visible and self.phase == 'CONTACT':
            self.remaining, self.rate = 99., 0.
            return previous
        self.rate = 4. if self.phase == 'CONTACT' else 1.
        budget = max(0., dt)
        while self.phase != 'CALM' and budget > 1e-9:
            rate = 4. if self.phase == 'CONTACT' else 1.
            elapsed = min(budget, self.remaining / rate)
            self.remaining = max(0., self.remaining - elapsed * rate)
            budget -= elapsed
            if self.remaining > 1e-7:
                break
            self.phase = {'CONTACT': 'SEARCH', 'SEARCH': 'SUSPICIOUS', 'SUSPICIOUS': 'CALM'}[self.phase]
            self.remaining = self.duration = {'SEARCH': 99., 'SUSPICIOUS': 180., 'CALM': 0.}[self.phase]
        self.rate = 4. if self.phase == 'CONTACT' else 1.
        return previous

    def snapshot(self):
        return dict(phase=self.phase, remaining=self.remaining, duration=self.duration, rate=self.rate)


class RadioLog:
    def __init__(self):
        self.time = 0.
        self.serial = 0
        self.entries = deque(maxlen=128)
        self._recent = {}

    def emit(self, sender, text):
        if not text:
            return
        key = (sender, text)
        if self.time - self._recent.get(key, -999.) < 8.:
            return
        self._recent = {key: time for key, time in self._recent.items() if self.time - time < 8.}
        self._recent[key] = self.time
        self.serial += 1
        self.entries.append(dict(id=self.serial, time=round(self.time, 2), sender=sender, text=text))

    def snapshot(self):
        return list(self.entries)[-3:]


class RadioDialogue(dict):
    def __init__(self, log):
        super().__init__()
        self.log = log

    def __setitem__(self, sender, text):
        super().__setitem__(sender, text)
        self.log.emit(sender, text)


class RuntimeSecurity:
    def _reset_security_runtime(self):
        self.security_cycle = SecurityCycle()
        self.radio_log = RadioLog()
        self.guard_dialogue = RadioDialogue(self.radio_log)
        self._firearm_dialogue = RadioDialogue(self.radio_log)

    def _radio(self, sender, text):
        dialogue = self.guard_dialogue if hasattr(self, 'env') else self._firearm_dialogue
        dialogue[sender] = text

    def _step_security_cycle(self):
        self.radio_log.time += self.game_dt
        if hasattr(self, 'env'):
            visible = any(g.seeing_player for g in self._active_guards()) or any(c.tracking for c in self.cameras)
        else:
            visible = any(u.alive and u.uid not in self._departed and u.local_visual for u in self.core.units.values())
            if visible and not self.spawn_protected:
                if self.security_cycle.phase != 'CONTACT':
                    self._radio('SWAT', 'Intruso avvistato! Contatto visivo, squadra in ingaggio.')
                self.security_cycle.contact()
        previous = self.security_cycle.step(self.game_dt, visible and not self.spawn_protected)
        phase = self.security_cycle.phase
        if previous != phase:
            text = {'SEARCH': 'Contatto perso. Avviate evasione: controllate l’ultimo punto noto.',
                    'SUSPICIOUS': 'Ricerca conclusa. Mantenete la cautela per almeno tre minuti.',
                    'CALM': 'Settore controllato. Riprendete la sorveglianza ordinaria.'}.get(phase)
            if text:
                self._radio('HQ', text)
            if hasattr(self, 'env'):
                if phase == 'SUSPICIOUS':
                    for _, unit in self.swat_units:
                        if unit.active:
                            unit.state, unit.target = 'WITHDRAW', unit.exit_pos
                    self._room_entries.clear()
                elif phase == 'CALM':
                    self.security_compromised = False
                    self.team_contact, self.team_contact_source = None, ''
                    self._player_identified = False
                    self.env.alert, self.env.personal_suspicion = FacilityAlert.CALM, 0.
                    self.env.intruder_known, self.env.last_seen_pos = False, None
                    self.threat_level = 0
                    for guard in self._active_guards():
                        if not guard.seeing_player and not guard.guard_id.startswith('S'):
                            guard.suspicion, guard.last_seen = 0., None
                            guard.state, guard.target = 'PATROL', guard.patrol_target()
            else:
                if phase == 'SUSPICIOUS':
                    self._withdrawals.update(uid for uid, u in self.core.units.items() if u.alive)
                if phase == 'CALM':
                    self._equipment_alert = None
                    self.core.blackboard.contact_pos = None
        if hasattr(self, 'env'):
            self.facility_alert = phase
        else:
            self._equipment_alert = phase if phase != 'CALM' else None
