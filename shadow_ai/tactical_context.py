"""Optional learned tactical choices over the playable game's physical actions.

Legacy movement policies retain their original observation/action interfaces.
These small role-specific policies see only an actor's state, visual memories,
facility inventory and medical requests already communicated to its team.
"""
from enum import IntEnum
import json
import math
import os
from pathlib import Path

import numpy as np

from .soliton_security import line_clear, turn_towards


CONTEXT_OBSERVATION_SIZE = 32
MANIFEST_PATH = Path(__file__).resolve().parents[1] / '.runtime' / 'training' / 'active.json'


class ContextAction(IntEnum):
    CONTINUE = 0
    SEEK_CARE = 1
    COVER_ALLY = 2
    TAKE_COVER = 3
    SECURE_INTEL = 4
    SEARCH_REPORT = 5


def context_role(runtime, uid, actor):
    return 'swat' if not hasattr(runtime, 'env') or uid.startswith('S') else 'guard'


def _vitals(runtime, uid, actor):
    return actor.vitals if hasattr(runtime, 'env') else runtime._unit_vitals[uid]


def _contact(runtime, actor):
    if hasattr(runtime, 'env'):
        point, age = actor.last_seen, actor.last_seen_age
        if point is None and runtime.team_contact is not None:
            point, age = runtime.team_contact, runtime.team_contact_age
    else:
        point, age = actor.last_seen_pos, actor.last_seen_age
        bb = runtime.core.blackboard
        if point is None and bb.contact_pos is not None:
            point, age = bb.contact_pos, bb.contact_age
    return (None, 999.) if point is None or age > 20. else (point, age)


def _care_jobs(runtime):
    return getattr(runtime, '_medical_jobs', {}) if hasattr(runtime, 'env') else getattr(runtime, '_context_medical_jobs', {})


def _ally_care_point(runtime, uid):
    jobs = _care_jobs(runtime)
    points = [job['point'] for patient, job in jobs.items() if patient != uid and job.get('helper') != uid]
    return points[0].copy() if points else None


def _known_station_points(runtime):
    # Locations belong to the facility's known layout. A spent station ceases to
    # be a usable resource; neither this inventory nor jobs reveal player data.
    return [item['pos'] for item in runtime.pickups if item.get('fixed') and item['kind'] == 'MEDKIT'
            and item.get('reserved_by') in (None, 'PLAYER')]


def observe_context(runtime, uid, actor):
    """Version-one schema, shared without translation by training and live play."""
    vitals = _vitals(runtime, uid, actor)
    campaign = hasattr(runtime, 'env')
    visible = bool(actor.seeing_player if campaign else actor.local_visual)
    confidence = actor.visual_confidence if campaign else actor.local_visual_confidence
    point, age = _contact(runtime, actor)
    contact_rel = np.zeros(2) if point is None else (point - actor.pos) / 11.5
    jobs = _care_jobs(runtime)
    own_care = uid in jobs or any(job.get('helper') == uid for job in jobs.values())
    ally = _ally_care_point(runtime, uid)
    ally_rel = np.zeros(2) if ally is None else (ally - actor.pos) / 20.
    station_points = _known_station_points(runtime)
    station = min(station_points, key=lambda p: np.linalg.norm(p - actor.pos)) if station_points else None
    station_rel = np.zeros(2) if station is None else (station - actor.pos) / 40.
    medics = (any(g.active and g.is_medic and g.medkits > 0 and g.guard_id != uid
                  for g in runtime.aux_guards) if campaign else False)
    intel_points, missing = [], 0
    if campaign:
        intel_points = [p for i, p in enumerate(runtime.env._intel)
                        if runtime.env.episode_has_intel[i] and runtime.env.known_intel_present[i]]
        missing = sum(bool(runtime.env.episode_has_intel[i] and not runtime.env.known_intel_present[i])
                      for i in range(len(runtime.env._intel)))
    intel = min(intel_points, key=lambda p: np.linalg.norm(p - actor.pos)) if intel_points else None
    intel_rel = np.zeros(2) if intel is None else (intel - actor.pos) / 40.
    capacity = max(1, runtime._gun_capacity(getattr(actor, 'weapon', None) or 'PISTOL') + 1)
    shield = actor.has_shield if campaign else actor.profile.has_shield
    armor = actor.armor if campaign else runtime._unit_armor.get(uid, 0.)
    alert = runtime.security_cycle.phase != 'CALM'
    obs = np.array([
        vitals.health, vitals.stamina, vitals.bleeding, vitals.injured, vitals.exhausted,
        getattr(actor, 'ammo', 0) / capacity, armor, getattr(actor, 'medkits', 0) / 2.,
        getattr(actor, 'is_medic', False), context_role(runtime, uid, actor) == 'swat', shield,
        visible, confidence, point is not None, min(1., age / 20.), *contact_rel,
        own_care, ally is not None, *ally_rel, medics, station is not None, *station_rel,
        len(intel_points) / 3., missing / 3., *intel_rel, alert,
        min(1., vitals.regen_remaining / .35),
        (getattr(actor, 'smoke_count', 0) + getattr(actor, 'grenade_count', 0)) / 2.,
    ], dtype=np.float32)
    return np.clip(obs, -1., 1.)


def _move(runtime, uid, actor, point, state):
    if hasattr(runtime, 'env'):
        actor.state, actor.target = state, point.copy()
        actor.pos, actor.angle = runtime._guard_move(actor, point,
            float(runtime.env.cfg.guard_speed) / runtime.game_dt * 1.5, 3.)
    else:
        runtime.actions[uid] = state
        runtime._move_swat_unit(actor, point)


def _standalone_care(runtime, uid, actor):
    vitals = _vitals(runtime, uid, actor)
    if actor.local_visual:
        vitals.cancel_treatment()
        runtime._context_medical_jobs.pop(uid, None)
        return False
    jobs = runtime._context_medical_jobs
    job = jobs.get(uid)
    if job is None:
        if not (vitals.bleeding > 0 or vitals.injured or vitals.health < .85 and vitals.regen_remaining <= 0):
            return False
        supplies = [item for item in runtime.pickups if item.get('fixed') and item['kind'] == 'MEDKIT'
                    and not item.get('reserved_by') and item['id'] != runtime.player_actions.station_id]
        reachable = [(runtime.navigator.path_length(actor.pos, item['pos']), item) for item in supplies]
        reachable = [(distance, item) for distance, item in reachable if np.isfinite(distance)]
        if not reachable:
            return False
        station = min(reachable, key=lambda entry: entry[0])[1]
        job = dict(station=station['id'], point=station['pos'].copy(), started=False)
        jobs[uid] = job
        runtime._radio(uid, 'Sono ferito, raggiungo il medikit. Copritemi.')
    if not job['started']:
        station = next((item for item in runtime.pickups if item['id'] == job['station']
                        and not item.get('reserved_by') and item['id'] != runtime.player_actions.station_id), None)
        if station is None:
            jobs.pop(uid, None)
            return False
        if np.linalg.norm(actor.pos - job['point']) > 1.1 or not line_clear(runtime.core.map.grid, actor.pos, job['point']):
            _move(runtime, uid, actor, job['point'], 'SEEK MEDKIT')
            return True
        if not vitals.start_treatment('MEDKIT'):
            jobs.pop(uid, None)
            return False
        runtime.pickups[:] = [item for item in runtime.pickups if item['id'] != station['id']]
        runtime.medical_watch.consume(station['id'])
        job['started'] = True
    if vitals.treatment != 'MEDKIT':
        jobs.pop(uid, None)
        return False
    runtime.actions[uid] = 'TREATMENT'
    actor.aim_timer = 0.
    vitals.step(False, runtime.game_dt, fatigue_on_pause=False)
    vitals.step_treatment(runtime.game_dt)
    actor.health = vitals.health
    runtime._unit_vitals_stepped.add(uid)
    if vitals.treatment is None:
        jobs.pop(uid, None)
        runtime._radio(uid, 'Cura completata. Riprendo il servizio.')
    return True


def apply_context(runtime, uid, actor, action):
    """Execute a discretionary choice using the same physical helpers as play.

    True means this step was actually executed and replaces legacy movement.
    False means CONTINUE or an unavailable choice; no sample should be credited
    to an unavailable choice. Mandatory sensors and hit reactions run first.
    """
    action = ContextAction(int(action))
    campaign = hasattr(runtime, 'env')
    vitals = _vitals(runtime, uid, actor)
    if vitals.health <= 0 or vitals.hit_stun_remaining > 0 or getattr(actor, 'blinded_timer', 0.) > 0:
        return False
    if action == ContextAction.CONTINUE:
        actor._context_cover = None
        if campaign and actor.state in {'COVER MEDIC', 'TACTICAL COVER', 'SEARCH REPORT'}:
            point, _ = _contact(runtime, actor)
            if uid.startswith('G') and runtime._player_identified:
                runtime._guard_goal_from_radio(actor)
            elif point is not None:
                runtime._start_guard_search(actor, point)
            else:
                actor.state, actor.target = 'PATROL', actor.patrol_target()
        return False
    visible = actor.seeing_player if campaign else actor.local_visual
    if action == ContextAction.SEEK_CARE:
        if campaign:
            return runtime._step_guard_medical(actor, visible)
        if not hasattr(runtime, '_context_medical_jobs'):
            runtime._context_medical_jobs = {}
        return _standalone_care(runtime, uid, actor)
    point, age = _contact(runtime, actor)
    if action == ContextAction.COVER_ALLY:
        ally = _ally_care_point(runtime, uid)
        if ally is None or vitals.health < .55:
            return False
        direction = np.array([math.cos(actor.angle), math.sin(actor.angle)], dtype=np.float32)
        if point is not None:
            direction = (point - ally) / max(float(np.linalg.norm(point - ally)), 1e-6)
        goal = runtime.navigator.goal(ally + direction * 1.8)
        if not np.isfinite(runtime.navigator.path_length(actor.pos, goal)):
            return False
        _move(runtime, uid, actor, goal, 'COVER MEDIC')
        return True
    if action == ContextAction.TAKE_COVER:
        if point is None or not visible and age > 5.:
            return False
        cover = getattr(actor, '_context_cover', None)
        if cover is None or np.linalg.norm(cover['source'] - point) > 1.:
            cover = dict(source=point.copy(), goal=runtime._incoming_cover_goal(actor.pos, point))
            actor._context_cover = cover
        goal = cover['goal']
        if np.linalg.norm(goal - actor.pos) < .25:
            vitals.step(False, runtime.game_dt, fatigue_on_pause=False)
            if campaign:
                actor.state, actor.target = 'TACTICAL COVER', goal.copy()
            else:
                runtime.actions[uid] = 'TACTICAL COVER'
                runtime._unit_vitals_stepped.add(uid)
            actor.aim_timer = 0.
            return True
        _move(runtime, uid, actor, goal, 'TACTICAL COVER')
        actor.aim_timer = 0.
        return True
    if action == ContextAction.SECURE_INTEL:
        if not campaign or visible:
            return False
        # Response teams finish their bounded search before providing support.
        if uid.startswith('S'):
            return False
        assignment = runtime._security_assignment(actor)
        if assignment is None:
            return False
        actor.pause_angle = runtime._security_heading(assignment)
        _move(runtime, uid, actor, assignment['goal'], assignment['state'])
        if np.linalg.norm(actor.pos - assignment['goal']) < .25:
            actor.angle = turn_towards(actor.angle, actor.pause_angle, 2. * runtime.game_dt)
        return True
    if action == ContextAction.SEARCH_REPORT:
        if point is None or visible:
            return False
        if age > 8. or campaign and (uid.startswith('G') and runtime._player_identified
                                    or uid in runtime._searches):
            return False
        _move(runtime, uid, actor, point, 'SEARCH REPORT')
        if np.linalg.norm(actor.pos - point) < .5:
            actor.angle = turn_towards(actor.angle, actor.angle + .5, 2. * runtime.game_dt)
        return True
    return False


class RuntimeContextTactics:
    def _reset_context_tactics(self):
        self._context_models = {}
        self._context_choices = {}
        self._context_medical_jobs = {}
        self.context_decisions = []
        self._legacy_policy_overrides = set()
        if os.getenv('SHADOW_CONTEXT_MODELS', '').lower() == 'off':
            return
        manifest = Path(os.getenv('SHADOW_CONTEXT_MANIFEST', str(MANIFEST_PATH)))
        if not manifest.is_file():
            return
        try:
            data = json.loads(manifest.read_text(encoding='utf-8'))
            if data.get('version') != 1:
                return
            from stable_baselines3 import PPO
            for role in ('guard', 'swat'):
                path = Path(data.get(role, ''))
                if not path.is_absolute():
                    path = manifest.parent / path
                if not path.is_file():
                    continue
                model = PPO.load(path, device='cpu')
                if (getattr(getattr(model, 'observation_space', None), 'shape', None) == (CONTEXT_OBSERVATION_SIZE,)
                        and getattr(getattr(model, 'action_space', None), 'n', None) == len(ContextAction)):
                    self._context_models[role] = model
        except Exception as exc:
            # A partial or incompatible optional manifest leaves the base game
            # playable. Training publishes the complete manifest atomically.
            self._context_models = {}
            self._context_load_error = str(exc)

    def _legacy_guard_choice_executed(self, actor, action):
        """Credit only intentions consistent with the resulting physical order."""
        if action is None:
            return False
        states = {
            0: {'OBSERVE'}, 1: {'PATROL', 'OBSERVE'}, 2: {'INVESTIGATE'},
            3: {'SEARCH', 'RESPOND'}, 4: {'FLANK'}, 5: {'SEARCH'},
            6: {'PROTECT INTEL'}, 7: {'FLANK'}, 8: {'FLANK'},
            9: {'CHASE', 'SEARCH'}, 10: {'CHASE'}, 12: {'VERIFY'}, 13: {'VERIFY'},
        }
        return actor.state in states.get(int(action), set())

    def _step_context_tactics(self, uid, actor):
        if hasattr(self, 'env') and actor.state in {'WITHDRAW', 'LEAVING', 'EXITED', 'SUPPORT INTEL'}:
            return False, False
        model = self._context_models.get(context_role(self, uid, actor))
        if model is None:
            return False, False
        now = self.policy_tick * self.game_dt
        cached = self._context_choices.get(uid)
        refresh = cached is None or now >= cached['next']
        if refresh:
            observation = observe_context(self, uid, actor)
            choice, _ = model.predict(observation, deterministic=True)
            action = ContextAction(int(choice))
            cached = dict(action=action, observation=observation, next=now + .2)
            self._context_choices[uid] = cached
        action = cached['action']
        jobs = _care_jobs(self)
        ongoing = uid in jobs or any(job.get('helper') == uid for job in jobs.values())
        if ongoing:
            # A fresh policy choice cannot abandon a patient mid-application.
            return True, apply_context(self, uid, actor, ContextAction.SEEK_CARE)
        consumed = apply_context(self, uid, actor, action)
        if (not consumed and hasattr(self, 'env')
                and actor.state in {'TACTICAL COVER', 'SEARCH REPORT', 'COVER MEDIC'}):
            # An unavailable/expired choice must release its previous movement
            # state; otherwise a repeated policy output can hold it indefinitely.
            point, _ = _contact(self, actor)
            actor._context_cover = None
            if uid.startswith('G') and self._player_identified:
                self._guard_goal_from_radio(actor)
            elif point is not None:
                self._start_guard_search(actor, point)
            elif uid.startswith('S'):
                self._finish_swat_security_search(actor)
            else:
                actor.state, actor.target = 'PATROL', actor.patrol_target()
        if refresh and (consumed or action == ContextAction.CONTINUE):
            self.context_decisions.append((cached['observation'].copy(), int(action), uid))
        return True, consumed
