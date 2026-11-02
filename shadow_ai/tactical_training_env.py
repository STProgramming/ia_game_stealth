"""Short, randomized tactical rehearsals using the playable game's executors.

The neural policy chooses an intention. Movement, finite medicine, radio
knowledge and camera evidence are handled by the same code used in a match.
The legacy navigation/guard policies are intentionally retained underneath.
"""
from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .interactive_runtime import InteractiveGuardRuntime, InteractiveSwatRuntime
from .playable_threats import MedicalSurveillance
from .soliton_security import BodyEvidence, SecurityCamera, SecurityGuard, line_clear
from .tactical_context import ContextAction, observe_context, apply_context


SCENARIOS = (
    'routine', 'fresh_contact', 'healthy_medic', 'wounded_station',
    'wounded_medic', 'wounded_personal_kit', 'spent_station', 'wounded_no_supplies',
    'cover_medic', 'exposed_wounded', 'empty_weapon_contact', 'exhausted_contact',
    'missing_intel', 'camera_theft', 'camera_body', 'search_report',
    'all_intel_missing', 'no_report',
    'recovering_station', 'recovering_medic', 'recovering_personal_kit',
)
STANDALONE_SCENARIOS = (
    'routine', 'fresh_contact', 'wounded_station', 'spent_station',
    'wounded_no_supplies', 'exposed_wounded', 'empty_weapon_contact',
    'exhausted_contact', 'search_report', 'no_report',
    'recovering_station',
)


def expected_context_action(observation):
    """Observable doctrine target, separate from measured physical progress.

    No scenario label, player route, or hidden player coordinate enters this
    target. It expresses priorities for a compact contextual rehearsal; it is
    not used by the live neural controller.
    """
    o = np.asarray(observation)
    if o[11] > .5:
        if o[0] < .6 or o[5] < .15 or o[4] > .5:
            return int(ContextAction.TAKE_COVER)
        return int(ContextAction.CONTINUE)
    if o[17] > .5:
        return int(ContextAction.SEEK_CARE)
    # The actor knows whether a completed kit is still healing it. That state
    # avoids repeatedly spending finite supplies on the same treated wound.
    wounded = o[2] > .01 or o[3] > .5 or o[0] < .85 and o[30] <= 0
    supplies = o[7] > 0 or o[21] > .5 or o[22] > .5
    if wounded and supplies:
        return int(ContextAction.SEEK_CARE)
    if o[18] > .5 and o[0] >= .6:
        return int(ContextAction.COVER_ALLY)
    if o[29] > .5 and o[25] > 0:
        return int(ContextAction.SECURE_INTEL)
    if o[13] > .5:
        return int(ContextAction.SEARCH_REPORT)
    return int(ContextAction.CONTINUE)


class TacticalTrainingEnv(gym.Env):
    """32 observations / six intentions for campaign guards and SWAT.

    Each decision advances twelve real 30 Hz actor frames. Scenarios randomize
    mission geometry, actor placement, health, resources and remembered radio
    reports. Training and evaluation select different seeds externally.
    """
    metadata = {'render_modes': []}

    def __init__(self, mode='guard', seed=None, scenario=None, training=True,
                 horizon=12, physics_frames=12, model_path=None):
        super().__init__()
        if mode not in {'guard', 'swat'}:
            raise ValueError('mode must be guard or swat')
        if scenario is not None and scenario not in SCENARIOS:
            raise ValueError(f'Unknown tactical scenario: {scenario}')
        self.mode = mode
        self.forced_scenario = scenario
        self.training = bool(training)
        self.horizon = int(horizon)
        self.physics_frames = int(physics_frames)
        self.action_space = spaces.Discrete(len(ContextAction))
        self.observation_space = spaces.Box(-1., 1., shape=(32,), dtype=np.float32)
        self._seed = int(seed or 0)
        path = Path(model_path) if model_path else Path(__file__).resolve().parent.parent / 'models_v7_1_guard/guard_v7_1_final.zip'
        self.runtime = InteractiveGuardRuntime(path, seed=self._seed)
        self._campaign_runtime = self.runtime
        self._standalone_runtime = None
        self.backend = 'campaign'
        self.actor = None
        self.scenario = None
        self.steps = 0
        self._episode_correct = 0
        self._episode_progress = 0.
        self._progress_events = 0
        self._physical_applicable = False

    def _point(self, point):
        return self.runtime.navigator.goal(np.asarray(point, dtype=np.float32))

    def _station(self, point):
        return dict(id='training-shared-kit', kind='MEDKIT', amount=1, fixed=True,
                    source='MAP', pos=self._point(point))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
        options = options or {}
        scenario = options.get('scenario', self.forced_scenario)
        self.scenario = str(scenario or self.np_random.choice(SCENARIOS))
        if self.scenario not in SCENARIOS:
            raise ValueError(f'Unknown tactical scenario: {self.scenario}')
        backend = options.get('backend')
        if backend is None:
            use_standalone = (self.mode == 'swat' and self.training
                              and self.scenario in STANDALONE_SCENARIOS
                              and bool(self.np_random.integers(2)))
            backend = 'standalone' if use_standalone else 'campaign'
        if backend not in {'campaign', 'standalone'}:
            raise ValueError('backend must be campaign or standalone')
        if backend == 'standalone':
            if self.mode != 'swat' or self.scenario not in STANDALONE_SCENARIOS:
                raise ValueError(f'Standalone SWAT does not support {self.mode}/{self.scenario}')
            return self._reset_standalone()
        self.backend, self.runtime = 'campaign', self._campaign_runtime
        r = self.runtime
        # All three real layouts are rehearsed. The theft-camera scenario needs
        # two objectives so that securing the remaining dossier is possible.
        r.level = 2 if self.scenario == 'camera_theft' else int(self.np_random.integers(1, 4))
        r.seed = self._seed + int(self.np_random.integers(0, 100000))
        r.reset('routine')
        r.lead_alive = False
        r.spawn_safe = False
        r.spawn_grace = 0.
        r.cameras = []
        r.bodies = []
        r.blood = []
        r.pickups = []
        r.noise_events = []
        r.env.player_pos = np.array([2., 27.], dtype=np.float32)
        r.env.player_velocity[:] = 0.
        r.env.player_current_speed = 0.
        base = self._point([float(self.np_random.uniform(11.7, 15.3)),
                            float(self.np_random.uniform(11.7, 14.5))])
        uid = 'S0' if self.mode == 'swat' else 'G8'
        actor = SecurityGuard(uid, base.copy(), float(self.np_random.uniform(-math.pi, math.pi)),
                              (tuple(base), tuple(base + [1., 0.])))
        actor.has_shield = self.mode == 'swat' and bool(self.np_random.integers(2))
        actor.armor = float(self.np_random.uniform(.3, 1.)) if self.mode == 'swat' else 0.
        actor.flash_count = int(self.np_random.integers(0, 3))
        actor.smoke_count = int(self.np_random.integers(0, 3))
        actor.grenade_count = int(self.np_random.integers(0, 2)) if self.mode == 'swat' else 0
        actor.vitals.stamina = float(self.np_random.uniform(.65, 1.))
        actor.ammo = int(self.np_random.integers(8, 17))
        r.aux_guards = [actor] if self.mode == 'guard' else []
        r.swat_units = [('ASSAULT', actor)] if self.mode == 'swat' else []
        self.actor = actor

        if self.scenario in {'wounded_station', 'wounded_medic', 'wounded_personal_kit',
                             'spent_station', 'wounded_no_supplies', 'recovering_station',
                             'recovering_medic', 'recovering_personal_kit'}:
            if self.scenario.startswith('recovering_'):
                actor.vitals.health = float(self.np_random.uniform(.35, .8))
                actor.vitals.regen_remaining = float(self.np_random.uniform(.1, .25))
                actor.vitals.regen_seconds = 120.
            else:
                actor.vitals.take_hit(float(self.np_random.uniform(.26, .55)))
            if self.scenario in {'wounded_station', 'spent_station', 'recovering_station'}:
                r.pickups = [self._station(base + [float(self.np_random.uniform(.1, .6)), 0.])]
            if self.scenario in {'wounded_medic', 'recovering_medic'}:
                medic = SecurityGuard('G9', self._point(base + [float(self.np_random.uniform(1.3, 3.)), 0.]),
                                      math.pi, (tuple(base),), is_medic=True, medkits=2)
                r.aux_guards.append(medic)
            if self.scenario in {'wounded_personal_kit', 'recovering_personal_kit'}:
                actor.is_medic = True
                actor.medkits = int(self.np_random.integers(1, 3))
        if self.scenario == 'healthy_medic':
            actor.is_medic, actor.medkits = True, 2
        r.medical_watch = MedicalSurveillance(r.pickups)
        if self.scenario == 'spent_station':
            r.medical_watch.consume(r.pickups[0]['id'])
            r.pickups.clear()

        if self.scenario == 'cover_medic':
            patient_point = self._point(base + [float(self.np_random.uniform(1.8, 3.2)), 0.])
            patient = SecurityGuard('G9', patient_point.copy(), math.pi, (tuple(patient_point),))
            patient.vitals.take_hit(.4)
            medic = SecurityGuard('G10', self._point(patient_point + [.6, 0.]), math.pi,
                                  (tuple(patient_point),), is_medic=True, medkits=2)
            r.aux_guards.extend([patient, medic])
            r._step_guard_medical(patient, False)

        if self.scenario in {'fresh_contact', 'exposed_wounded', 'empty_weapon_contact', 'exhausted_contact'}:
            actor.angle = 0.
            point = self._point(base + [float(self.np_random.uniform(2., 4.)), 0.])
            r.env.player_pos = point.copy()
            actor.seeing_player = True
            actor.visual_confidence = 1.
            actor.last_seen, actor.last_seen_age = point.copy(), 0.
            r.team_contact, r.team_contact_age = point.copy(), 0.
            if self.scenario == 'exposed_wounded':
                actor.vitals.take_hit(float(self.np_random.uniform(.45, .65)))
            if self.scenario == 'empty_weapon_contact':
                actor.ammo = 0
            if self.scenario == 'exhausted_contact':
                actor.vitals.stamina, actor.vitals.exhausted = 0., True

        if self.scenario in {'missing_intel', 'all_intel_missing', 'camera_theft'}:
            indices = r.env.episode_intel_order
            index = indices[0]
            r.env.player_has_intel[index] = 1
            if self.scenario == 'all_intel_missing':
                for index in indices:
                    r.env.player_has_intel[index] = 1
                    r.env.known_intel_present[index] = 0
                report = self._point(base + [float(self.np_random.uniform(2., 5.)), 1.])
                actor.last_seen, actor.last_seen_age = report.copy(), float(self.np_random.uniform(2., 9.))
            elif self.scenario == 'camera_theft':
                target = r.env._intel[index]
                camera_point = self._point(target + [-1.8, 0.])
                angle = float(np.arctan2(*(target - camera_point)[::-1]))
                r.cameras = [SecurityCamera('C-T', camera_point, angle, sweep=0., angle=angle)]
                r._step_cameras()
                for _ in range(37):
                    r._step_intel_radio()
            else:
                r.env.known_intel_present[index] = 0
            r.security_cycle.caution()
            r.security_compromised = True
        if self.scenario == 'camera_body':
            body_point = self._point(base + [1., 0.])
            camera_point = self._point(base + [-2., 0.])
            r.bodies = [BodyEvidence(1, 'G-DEAD', body_point, 0., bool(self.np_random.integers(2)))]
            r.cameras = [SecurityCamera('C-B', camera_point, 0., sweep=0., angle=0.)]
            r._step_cameras()
        if self.scenario == 'search_report':
            point = self._point(base + [float(self.np_random.uniform(2., 5.)),
                                       float(self.np_random.uniform(-2., 2.))])
            actor.last_seen, actor.last_seen_age = point.copy(), float(self.np_random.uniform(1., 12.))
            r.team_contact, r.team_contact_age = point.copy(), actor.last_seen_age

        self.steps = 0
        self._episode_correct = 0
        self._episode_progress = 0.
        self._progress_events = 0
        observation = observe_context(r, uid, actor)
        self._physical_applicable = expected_context_action(observation) != ContextAction.CONTINUE
        return observation, self._info(observation)

    def _reset_standalone(self):
        """Use actual dedicated SWAT units, senses, movement and station care."""
        if self._standalone_runtime is None:
            model_path = Path(__file__).resolve().parent.parent / 'models_v7_1_swat/swat_v7_1_final.zip'
            self._standalone_runtime = InteractiveSwatRuntime(model_path, seed=self._seed)
        r = self._standalone_runtime
        self.backend, self.runtime = 'standalone', r
        r.seed = self._seed + int(self.np_random.integers(0, 100000))
        r.reset('entry_clear')
        r.spawn_safe, r.spawn_grace = False, 0.
        r.core.blackboard.contact_pos = None
        r.core.blackboard.contact_age = 999.
        r.core.smoke_clouds = []
        r.core.player_pos = self._point([2., 2.])
        r.core.player_velocity[:] = 0.
        r.pickups, r.blood = [], []
        r._context_medical_jobs = {}
        uid = str(self.np_random.choice(r.core.agent_ids))
        for other_uid, other in r.core.units.items():
            other.alive = other_uid == uid
        actor = r.core.units[uid]
        base = self._point([float(self.np_random.uniform(10.8, 14.)),
                            float(self.np_random.uniform(10.8, 14.))])
        actor.pos, actor.angle = base.copy(), float(self.np_random.uniform(-math.pi, math.pi))
        actor.guard_id = uid  # Common actor identifier for rehearsal reporting.
        actor.vitals = r._unit_vitals[uid]
        actor.vitals.stamina = float(self.np_random.uniform(.65, 1.))
        actor.ammo = int(self.np_random.integers(8, 17))
        actor.local_visual, actor.local_visual_confidence = False, 0.
        actor.last_seen_pos, actor.last_seen_age = None, 999.
        self.actor = actor
        if self.scenario in {'wounded_station', 'spent_station', 'wounded_no_supplies', 'recovering_station'}:
            if self.scenario == 'recovering_station':
                actor.vitals.health = float(self.np_random.uniform(.35, .8))
                actor.vitals.regen_remaining = float(self.np_random.uniform(.1, .25))
                actor.vitals.regen_seconds = 120.
            else:
                actor.vitals.take_hit(float(self.np_random.uniform(.26, .55)))
            actor.health = actor.vitals.health
            if self.scenario != 'wounded_no_supplies':
                r.pickups = [self._station(base + [float(self.np_random.uniform(.1, .6)), 0.])]
        r.medical_watch = MedicalSurveillance(r.pickups)
        if self.scenario == 'spent_station':
            r.medical_watch.consume(r.pickups[0]['id'])
            r.pickups.clear()
        if self.scenario in {'fresh_contact', 'exposed_wounded', 'empty_weapon_contact', 'exhausted_contact'}:
            actor.angle = 0.
            point = self._point(base + [float(self.np_random.uniform(2., 4.)), 0.])
            r.core.player_pos = point.copy()
            actor.local_visual, actor.local_visual_confidence = True, 1.
            actor.last_seen_pos, actor.last_seen_age = point.copy(), 0.
            r.core.blackboard.contact_pos, r.core.blackboard.contact_age = point.copy(), 0.
            if self.scenario == 'exposed_wounded':
                actor.vitals.take_hit(float(self.np_random.uniform(.45, .65)))
                actor.health = actor.vitals.health
            if self.scenario == 'empty_weapon_contact':
                actor.ammo = 0
            if self.scenario == 'exhausted_contact':
                actor.vitals.stamina, actor.vitals.exhausted = 0., True
        if self.scenario == 'search_report':
            point = self._point(base + [float(self.np_random.uniform(2., 5.)),
                                       float(self.np_random.uniform(-2., 2.))])
            actor.last_seen_pos, actor.last_seen_age = point.copy(), float(self.np_random.uniform(1., 12.))
            r.core.blackboard.contact_pos, r.core.blackboard.contact_age = point.copy(), actor.last_seen_age
        self.steps = self._episode_correct = self._progress_events = 0
        self._episode_progress = 0.
        observation = observe_context(r, uid, actor)
        self._physical_applicable = expected_context_action(observation) != ContextAction.CONTINUE
        return observation, self._info(observation)

    def _info(self, observation, **extra):
        return dict(scenario=self.scenario, mode=self.mode, backend=self.backend,
                    level=getattr(self.runtime, 'level', None),
                    expected_action=expected_context_action(observation),
                    action_accuracy=self._episode_correct / max(1, self.steps),
                    physical_progress=self._episode_progress,
                    physical_applicable=self._physical_applicable,
                    physical_success=self._progress_events > 0,
                    success=self._progress_events > 0,
                    **extra)

    def _progress_measure(self, action, before, old_health, old_treatment, old_jobs):
        actor, r = self.actor, self.runtime
        jobs = r._medical_jobs if self.backend == 'campaign' else r._context_medical_jobs
        distance = float(np.linalg.norm(actor.pos - before))
        if action == ContextAction.CONTINUE:
            return 0.
        if action == ContextAction.SEEK_CARE:
            new_job = actor.guard_id in jobs and actor.guard_id not in old_jobs
            treatment = actor.vitals.treatment == 'MEDKIT'
            healing = max(0., actor.vitals.health - old_health)
            return float(new_job) * .5 + float(treatment and old_treatment is None) * .5 + healing * 5. + min(.2, distance)
        if action == ContextAction.TAKE_COVER:
            source = actor.last_seen if self.backend == 'campaign' else actor.last_seen_pos
            if source is None:
                return 0.
            farther = float(np.linalg.norm(actor.pos - source) - np.linalg.norm(before - source))
            grid = r.env.grid if self.backend == 'campaign' else r.core.map.grid
            occluded = not line_clear(grid, source, actor.pos)
            return max(0., farther) + float(occluded and distance > .01) * .25
        if action == ContextAction.COVER_ALLY:
            care = list(jobs.values())
            if not care:
                return min(.2, distance)
            # Cover is an offset from the patient. Moving away from the patient
            # can be the correct physical route toward that covering position.
            goal = getattr(actor, 'target', None)
            if goal is None:
                return 0.
            progress = float(np.linalg.norm(before - goal) - np.linalg.norm(actor.pos - goal))
            holding = actor.state == 'COVER MEDIC' and np.linalg.norm(actor.pos - goal) < .3
            return max(0., progress) + float(distance > .01 or holding) * .05
        if action == ContextAction.SECURE_INTEL:
            if self.backend != 'campaign':
                return 0.
            points = [point for index, point in enumerate(r.env._intel)
                      if r.env.episode_has_intel[index] and r.env.known_intel_present[index]]
            if not points:
                return 0.
            point = min(points, key=lambda p: np.linalg.norm(before - p))
        elif action == ContextAction.SEARCH_REPORT:
            point = actor.last_seen if self.backend == 'campaign' else actor.last_seen_pos
            if point is None:
                return 0.
        else:
            return 0.
        return max(0., float(np.linalg.norm(before - point) - np.linalg.norm(actor.pos - point)))

    def step(self, action):
        action = ContextAction(int(action))
        r, actor = self.runtime, self.actor
        observation = observe_context(r, actor.guard_id, actor)
        expected = expected_context_action(observation)
        correct = int(action) == expected
        before = actor.pos.copy()
        old_health, old_treatment = actor.vitals.health, actor.vitals.treatment
        old_jobs = set(r._medical_jobs if self.backend == 'campaign' else r._context_medical_jobs)
        consumed = False
        for _ in range(self.physics_frames):
            consumed |= bool(apply_context(r, actor.guard_id, actor, action))
            # Other participants physically travel and apply medicine. Their
            # neural intentions are outside this actor's training decision.
            if self.backend == 'campaign':
                for other in list(r.aux_guards):
                    if other is actor:
                        continue
                    if other.guard_id in r._medical_jobs or any(j.get('helper') == other.guard_id for j in r._medical_jobs.values()):
                        r._step_guard_medical(other, False)
                r._step_guard_recovery()
                r.env.steps += 1
            else:
                if actor.vitals.treatment is None:
                    actor.vitals.step_treatment(r.game_dt)
                actor.health = actor.vitals.health
                r.core.world_ticks += 1
            actor.last_seen_age += r.game_dt
        physical = self._progress_measure(action, before, old_health, old_treatment, old_jobs)
        reward = (1. if correct else -.8) + min(.5, physical) * .25
        if action != ContextAction.CONTINUE and not consumed:
            reward -= .15
        self.steps += 1
        self._episode_correct += int(correct)
        self._episode_progress += physical
        self._progress_events += int(physical > .005)
        observation = observe_context(r, actor.guard_id, actor)
        terminated = not actor.active if self.backend == 'campaign' else not actor.alive or actor.health <= 0
        truncated = self.steps >= self.horizon
        info = self._info(observation, action_correct=bool(correct), consumed_step=consumed,
                          action_expected_before=expected, step_physical_progress=physical)
        return observation, float(reward), bool(terminated), bool(truncated), info
