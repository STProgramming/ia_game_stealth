"""Caution-only radio investigations, with physical searches and missing reports."""
import math
import numpy as np
from .playable_navigation import LocalSearch, local_search_points
from .soliton_security import turn_towards


class RuntimeNoiseDelegation:
    def _reset_noise_delegation(self):
        self._noise_clock = 0.
        self._noise_serial = 0
        self._noise_tasks = {}
        self._noise_requesters = {}
        self._noise_investigators = {}
        self._noise_cooldowns = {}

    @staticmethod
    def _noise_key(point, source):
        return source, int(round(float(point[0]) / 3.)), int(round(float(point[1]) / 3.))

    def _delegate_noise(self, requester, point, source):
        if self.facility_alert != 'SUSPICIOUS' or requester.guard_id.startswith('S'):
            return False
        if requester.guard_id in self._noise_requesters or requester.guard_id in self._noise_investigators:
            return True
        if requester.guard_id in self._room_orders or requester.visual_confidence >= .25:
            return False
        for task in self._noise_tasks.values():
            if task['status'] in {'INVESTIGATING', 'REPORTING'} and task['source'] == source and np.linalg.norm(task['origin'] - point) < 2.:
                return True  # A radio order already covers this local noise.
        key = self._noise_key(point, source)
        if self._noise_cooldowns.get(key, 0.) > self._noise_clock:
            return False
        distance = np.linalg.norm(requester.pos - point)
        path = self.navigator.path_length(requester.pos, point)
        candidates = []
        for guard in self._active_guards():
            uid = guard.guard_id
            if (uid == requester.guard_id or uid.startswith('S') or uid in self._room_orders
                    or uid in self._noise_requesters or uid in self._noise_investigators
                    or uid in self._launch_responses or guard.visual_confidence >= .25
                    or guard.blinded_timer > 0 or guard.vitals.hit_stun_remaining > 0
                    or (self.takedown and self.takedown['guard_id'] == uid)):
                continue
            idle = guard.state in {'PATROL', 'OBSERVE', 'PROTECT INTEL', 'SECURE EXFIL'}
            same_investigation = guard.state == 'INVESTIGATE' and guard.target is not None and np.linalg.norm(guard.target - point) < 2.
            if not idle and not same_investigation:
                continue
            route = self.navigator.path_length(guard.pos, point)
            if np.linalg.norm(guard.pos - point) + .75 < distance and np.isfinite(route) and route + .5 < path:
                candidates.append((route, uid, guard))
        if not candidates:
            return False
        travel, _, investigator = min(candidates, key=lambda entry: (entry[0], entry[1]))
        origin = self._navigable_goal(point)
        points = [origin.copy()] + local_search_points(self.env.grid, origin, int(investigator.guard_id[1:]))[1:3]
        search = LocalSearch(points, scan_seconds=.85)
        route = sum(self.navigator.path_length(a, b) for a, b in zip(points, points[1:]))
        speed = max(.5, float(self.env.cfg.guard_speed) / self.game_dt)
        timeout = max(8., (travel + route) / speed + len(points) * search.scan_seconds + 6.)
        self._noise_serial += 1
        task = dict(id=self._noise_serial, requester=requester.guard_id, investigator=investigator.guard_id,
            origin=origin, source=source, key=key, status='INVESTIGATING', search=search,
            deadline=self._noise_clock + timeout, result='', report_wait=0., evidence=investigator.evidence_id,
            requester_state=requester.state, requester_target=None if requester.target is None else requester.target.copy(),
            investigator_state=investigator.state, investigator_target=None if investigator.target is None else investigator.target.copy())
        self._noise_tasks[task['id']] = task
        self._noise_requesters[requester.guard_id] = task['id']
        self._noise_investigators[investigator.guard_id] = task['id']
        self._noise_cooldowns[key] = self._noise_clock + timeout + 5.
        self._support_calls.pop(requester.guard_id, None)
        self._searches.pop(investigator.guard_id, None)
        self._local_diversion_guards.difference_update({requester.guard_id, investigator.guard_id})
        requester.state, requester.target = 'WAIT REPORT', requester.pos.copy()
        investigator.state, investigator.target = 'INVESTIGATE FOR ALLY', origin.copy()
        self.guard_dialogue[requester.guard_id] = f'Radio: {investigator.guard_id}, sei più vicino al rumore. Controlla tu e fammi rapporto.'
        self.guard_dialogue[investigator.guard_id] = f'Ricevuto {requester.guard_id}. Vado a controllare e ti aggiorno.'
        self._dialogue_timer = 8.
        self.radio_updates += 1
        return True

    def _release_noise_task(self, task, guards):
        self._noise_requesters.pop(task['requester'], None)
        self._noise_investigators.pop(task['investigator'], None)
        for prefix in ['requester', 'investigator']:
            actor = guards.get(task[prefix])
            if actor and actor.state in {'WAIT REPORT', 'INVESTIGATE FOR ALLY', 'REPORTING'}:
                saved = task[f'{prefix}_state']
                actor.state = saved if saved in {'PATROL', 'OBSERVE', 'PROTECT INTEL', 'SECURE EXFIL'} else 'PATROL'
                target = task[f'{prefix}_target']
                actor.target = target.copy() if target is not None and actor.state != 'PATROL' else actor.patrol_target()

    def _send_noise_report(self, task, guards):
        task['status'] = 'REPORTED'
        self._release_noise_task(task, guards)
        self.guard_dialogue[task['investigator']] = f"Radio {task['requester']}: {task['result']}"
        self.guard_dialogue[task['requester']] = 'Ricevuto il rapporto. Mantengo la cautela.'
        self._dialogue_timer = 8.
        self.radio_updates += 1

    def _step_noise_radio(self):
        self._noise_clock += self.game_dt
        guards = {g.guard_id: g for g in self._active_guards()}
        for task in self._noise_tasks.values():
            if task['status'] not in {'INVESTIGATING', 'REPORTING'}:
                continue
            requester, investigator = guards.get(task['requester']), guards.get(task['investigator'])
            if (requester is None or self.facility_alert == 'CONTACT'
                    or task['requester'] in self._room_orders or task['investigator'] in self._room_orders):
                task['status'] = 'CANCELLED'
                self._release_noise_task(task, guards)
                continue
            if task['status'] == 'REPORTING':
                ready = (investigator is not None and investigator.vitals.hit_stun_remaining <= 0
                         and not getattr(investigator, 'radio_disabled', False)
                         and not (self.takedown and self.takedown['guard_id'] == investigator.guard_id))
                if ready:
                    task['report_wait'] -= self.game_dt
                    if task['report_wait'] <= 1e-7:
                        self._send_noise_report(task, guards)
                        continue
            if self._noise_clock >= task['deadline']:
                task['status'] = 'OVERDUE'
                self._release_noise_task(task, guards)
                self.security_compromised = True
                self.security_cycle.caution()
                requester.suspicion = max(requester.suspicion, .9)
                self._start_guard_search(requester, task['origin'])
                self.guard_dialogue[requester.guard_id] = f"Radio: {task['investigator']} non risponde. Sono insospettito, vado a verificare di persona."
                self._dialogue_timer = 9.
                self.radio_updates += 1
        # Keep bounded history for snapshots, without forgetting active requests.
        if len(self._noise_tasks) > 32:
            for key in list(self._noise_tasks):
                if len(self._noise_tasks) <= 32:
                    break
                if self._noise_tasks[key]['status'] not in {'INVESTIGATING', 'REPORTING'}:
                    del self._noise_tasks[key]
        self._noise_cooldowns = {key: time for key, time in self._noise_cooldowns.items() if time > self._noise_clock}

    def _step_noise_order(self, guard):
        uid = guard.guard_id
        ident = self._noise_investigators.get(uid, self._noise_requesters.get(uid))
        if ident is None:
            return False
        task = self._noise_tasks[ident]
        guards = {g.guard_id: g for g in self._active_guards()}
        if guard.seeing_player and guard.visual_confidence >= .25:
            if uid == task['investigator'] and guard.visual_confidence >= 1.:
                task['result'] = 'intruso avvistato, posizione comunicata alla squadra.'
                self._send_noise_report(task, guards)
            else:
                task['status'] = 'CANCELLED'
                self._release_noise_task(task, guards)
            return False
        if uid == task['requester']:
            if guard.state in {'INSPECT BODY','TRACK BLOOD'}:
                task['status'] = 'CANCELLED'
                self._release_noise_task(task, guards)
                return False
            guard.state, guard.target = 'WAIT REPORT', guard.pos.copy()
            guard.pos,guard.angle = self._guard_move(guard,None,0.)
            heading = math.atan2(float(task['origin'][1]-guard.pos[1]), float(task['origin'][0]-guard.pos[0]))
            guard.angle = turn_towards(guard.angle, heading, 2. * self.game_dt)
            return True
        if task['status'] == 'REPORTING':
            if guard.state in {'TRACK BLOOD','INSPECT BODY','SEARCH','VERIFY','CHASE'}:
                return False
            guard.state = 'REPORTING'
            guard.vitals.step(False, self.game_dt, fatigue_on_pause=False)
            return True
        if guard.evidence_id != task['evidence'] or guard.state in {'INSPECT BODY', 'TRACK BLOOD'}:
            task['result'] = 'ho trovato tracce sospette. Continuo a verificare il settore.'
            task['status'], task['report_wait'] = 'REPORTING', .65
            return False
        if any(uid in case['observers'] and np.linalg.norm(case['pos']-task['origin']) < 5. for case in self.spent_cases):
            task['result'] = 'bossoli a terra, il settore è compromesso.'
            task['status'], task['report_wait'] = 'REPORTING', .65
            return False
        goal, look, done = task['search'].step(guard.pos, guard.angle, self.game_dt)
        guard.state, guard.target = 'INVESTIGATE FOR ALLY', goal
        guard.pos, guard.angle = self._guard_move(guard, goal, float(self.env.cfg.guard_speed) / self.game_dt * .82, 3.)
        if look is not None:
            guard.angle = turn_towards(guard.angle, look, 3. * self.game_dt)
        if done:
            task['status'], task['report_wait'] = 'REPORTING', .65
            task['result'] = 'punto e dintorni controllati, nessun intruso individuato.'
            guard.state = 'REPORTING'
        return True

    def _noise_delegation_snapshot(self):
        return [dict(requester=t['requester'], investigator=t['investigator'], status=t['status'],
                     origin=dict(x=float(t['origin'][0]), y=float(t['origin'][1])), result=t['result'],
                     reply_remaining=max(0., t['deadline']-self._noise_clock)) for t in self._noise_tasks.values()]
