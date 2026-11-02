"""Finite shared medical supplies and radio-requested care in the campaign."""
import numpy as np

from .soliton_security import line_clear


CARE_STATES = {'SEEK MEDKIT', 'WAIT MEDIC', 'MEDICAL RESPONSE', 'TREAT ALLY', 'TREATMENT'}


class RuntimeGuardMedical:
    def _reset_guard_medical(self):
        self._medical_jobs = {}
        self._medical_retry = {}

    def _medical_guard(self, uid):
        if uid == 'G0':
            return self._lead_patrol if self._primary_active() else None
        return next((g for g in self.aux_guards + [u for _, u in self.swat_units]
                     if g.guard_id == uid and g.active), None)

    @staticmethod
    def _needs_guard_care(guard):
        vitals = guard.vitals
        return (vitals.health > 0 and (vitals.bleeding > 0 or vitals.injured
                or vitals.health < .85 and vitals.regen_remaining <= 1e-7))

    def _cancel_guard_care(self, uid):
        for patient_id, job in list(self._medical_jobs.items()):
            if patient_id != uid and job.get('helper') != uid:
                continue
            patient = self._medical_guard(patient_id)
            helper = self._medical_guard(job.get('helper')) if job.get('helper') else None
            if patient is not None:
                patient.vitals.cancel_treatment()
            for actor in (patient, helper):
                if actor is not None and actor.state in CARE_STATES:
                    actor.state, actor.target = 'PATROL', actor.patrol_target()
            del self._medical_jobs[patient_id]

    def _medical_idle(self, guard):
        guard.aim_timer = 0.
        guard.vitals.step(False, self.game_dt, fatigue_on_pause=False)

    def _step_guard_medical(self, guard, visible):
        uid = guard.guard_id
        self._medical_retry[uid] = max(0., self._medical_retry.get(uid, 0.) - self.game_dt)
        # Combat, smoke escape and physical incapacitation always take priority.
        if visible or guard.blinded_timer > 0:
            self._cancel_guard_care(uid)
            return False

        helper_job = next(((pid, job) for pid, job in self._medical_jobs.items()
                           if job.get('helper') == uid and pid != uid), None)
        if helper_job is not None:
            patient_id, job = helper_job
            patient = self._medical_guard(patient_id)
            if (patient is None or patient.seeing_player
                    or getattr(patient, '_incoming_cover', None) is not None
                    or job['started'] and patient.vitals.treatment != 'MEDKIT'
                    or guard.medkits <= 0 and not job['started']):
                self._cancel_guard_care(patient_id)
                return False
            # Travel to the position supplied in the medical radio request.
            guard.target = job['point'].copy()
            if (np.linalg.norm(guard.pos - job['point']) > 1.1
                    or not line_clear(self.env.grid, guard.pos, job['point'])):
                guard.state = 'MEDICAL RESPONSE'
                guard.pos, guard.angle = self._guard_move(guard, guard.target,
                    float(self.env.cfg.guard_speed) / self.game_dt * 1.5)
            else:
                guard.state = 'TREAT ALLY'
                self._medical_idle(guard)
            return True

        job = self._medical_jobs.get(uid)
        if job is None and self._needs_guard_care(guard) and self._medical_retry[uid] <= 0:
            if guard.medkits > 0:
                job = dict(helper=uid, point=guard.pos.copy(), started=False, remaining=30.)
                self._radio(uid, 'Mi curo con il mio kit. Copritemi alla posizione comunicata.')
            else:
                busy = {j.get('helper') for j in self._medical_jobs.values()}
                medics = [(self.navigator.path_length(g.pos, guard.pos), g)
                          for g in self._active_guards()
                          if g.is_medic and g.medkits > 0 and g.guard_id != uid
                          and g.guard_id not in busy and g.guard_id not in self._medical_jobs
                          and not g.seeing_player and g.blinded_timer <= 0
                          and g.vitals.hit_stun_remaining <= 0
                          and getattr(g, '_incoming_cover', None) is None
                          and g.state not in {'WITHDRAW', 'LEAVING', 'EXITED'}]
                medics = [(distance, g) for distance, g in medics if np.isfinite(distance) and distance <= 30.]
                if medics:
                    _, medic = min(medics, key=lambda item: item[0])
                    job = dict(helper=medic.guard_id, point=guard.pos.copy(), started=False, remaining=30.)
                    self._radio(uid, f'Richiedo assistenza medica alla mia posizione, {medic.guard_id}.')
                    self._radio(medic.guard_id, 'Ricevuto, medico in arrivo. Rimani al riparo.')
                else:
                    stations = [(self.navigator.path_length(guard.pos, item['pos']), item)
                                for item in self.pickups if item.get('fixed') and item['kind'] == 'MEDKIT'
                                and not item.get('reserved_by')
                                and item['id'] != self.player_actions.station_id]
                    stations = [(distance, item) for distance, item in stations if np.isfinite(distance)]
                    if stations:
                        _, station = min(stations, key=lambda item: item[0])
                        job = dict(station=station['id'], point=station['pos'].copy(),
                                   started=False, remaining=60.)
                        self._radio(uid, 'Sono ferito. Raggiungo il medikit per curarmi.')
            if job is not None:
                self._medical_jobs[uid] = job
            else:
                self._medical_retry[uid] = 1.

        if job is None:
            return False
        job['remaining'] -= self.game_dt
        if job['remaining'] <= 0:
            self._cancel_guard_care(uid)
            self._medical_retry[uid] = 2.
            return False

        if job['started']:
            if guard.vitals.treatment != 'MEDKIT':
                self._cancel_guard_care(uid)
                return False
            guard.state = 'TREATMENT'
            self._medical_idle(guard)
            guard.vitals.step_treatment(self.game_dt)
            if guard.vitals.treatment is None:
                self._radio(uid, 'Cura completata. Riprendo il servizio.')
                self._cancel_guard_care(uid)
            return True

        station = None
        helper = None
        if 'station' in job:
            station = next((item for item in self.pickups if item['id'] == job['station']
                            and not item.get('reserved_by')
                            and item['id'] != self.player_actions.station_id), None)
            if station is None:
                self._cancel_guard_care(uid)
                return False
            guard.target = job['point'].copy()
            if (np.linalg.norm(guard.pos - guard.target) > 1.1
                    or not line_clear(self.env.grid, guard.pos, guard.target)):
                guard.state = 'SEEK MEDKIT'
                guard.pos, guard.angle = self._guard_move(guard, guard.target,
                    float(self.env.cfg.guard_speed) / self.game_dt)
                return True
        else:
            helper = self._medical_guard(job['helper'])
            if helper is None or helper.medkits <= 0 or helper.seeing_player:
                self._cancel_guard_care(uid)
                return False
            guard.state, guard.target = 'WAIT MEDIC', guard.pos.copy()
            if (np.linalg.norm(helper.pos - guard.pos) > 1.2
                    or not line_clear(self.env.grid, helper.pos, guard.pos)):
                self._medical_idle(guard)
                return True

        if guard.vitals.start_treatment('MEDKIT'):
            job['started'] = True
            if station is not None:
                self.pickups[:] = [item for item in self.pickups if item['id'] != station['id']]
                self.medical_watch.consume(station['id'])
            else:
                helper.medkits -= 1
            guard.state = 'TREATMENT'
            self._medical_idle(guard)
            return True
        self._cancel_guard_care(uid)
        return False

    def _step_guard_recovery(self):
        for guard in self._active_guards():
            if guard.vitals.treatment is None:
                guard.vitals.step_treatment(self.game_dt)
