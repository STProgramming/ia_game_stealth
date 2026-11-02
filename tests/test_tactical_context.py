import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from shadow_ai import TacticalAction
from shadow_ai.soliton_security import SecurityGuard
from shadow_ai.tactical_context import ContextAction, apply_context, context_role, observe_context
from test_guard_medical import prepare
from test_interactive_tactics import game
from test_playable_actions import swat


class ContextPolicy:
    def __init__(self, choice):
        self.choice = choice
        self.observation_space = SimpleNamespace(shape=(32,))
        self.action_space = SimpleNamespace(n=6)

    def predict(self, observation, **kwargs):
        assert np.asarray(observation).shape == (32,)
        return int(self.choice), None


def test_optional_cover_keeps_one_goal_and_holds_it_instead_of_retreating_forever(game):
    game.env.grid[:] = 0
    guard = game.aux_guards[0]
    guard.pos = np.array([8., 8.], dtype=np.float32)
    guard.angle = np.pi
    guard.last_seen, guard.last_seen_age = np.array([12., 8.], dtype=np.float32), 0.
    guard.seeing_player = True
    assert apply_context(game, guard.guard_id, guard, ContextAction.TAKE_COVER)
    fixed_goal = guard._context_cover['goal'].copy()
    for _ in range(240):
        assert apply_context(game, guard.guard_id, guard, ContextAction.TAKE_COVER)
        assert np.array_equal(guard._context_cover['goal'], fixed_goal)
    assert np.linalg.norm(guard.pos - fixed_goal) < .25


def test_continue_releases_context_cover_and_resumes_search_from_legitimate_memory(game):
    guard = game.aux_guards[0]
    guard.state = 'TACTICAL COVER'
    guard._context_cover = {'source': guard.pos.copy(), 'goal': guard.pos.copy()}
    guard.last_seen, guard.last_seen_age = guard.pos + [2., 0.], 1.
    assert not apply_context(game, guard.guard_id, guard, ContextAction.CONTINUE)
    assert guard._context_cover is None and guard.state == 'SEARCH'
    assert guard.guard_id in game._searches


def test_context_observes_pending_recovery_to_avoid_wasting_a_second_kit(game):
    guard = game.aux_guards[0]
    guard.vitals.health = .6
    untreated = observe_context(game, guard.guard_id, guard)
    guard.vitals.regen_remaining = .25
    recovering = observe_context(game, guard.guard_id, guard)
    assert untreated[30] == 0 and recovering[30] > .5


def test_context_observation_uses_own_vitals_but_never_hidden_player_position(game):
    guard = game.aux_guards[0]
    guard.last_seen, guard.last_seen_age = None, 999.
    guard.seeing_player = False
    game.team_contact = None
    first = observe_context(game, guard.guard_id, guard)
    game.env.player_pos[:] = (60., 38.)
    assert np.array_equal(first, observe_context(game, guard.guard_id, guard))
    guard.vitals.health, guard.vitals.bleeding = .4, .7
    second = observe_context(game, guard.guard_id, guard)
    assert second.shape == (32,) and second.dtype == np.float32
    assert second[0] == pytest.approx(.4) and second[2] == pytest.approx(.7)
    assert np.max(np.abs(second)) <= 1.


def test_context_contact_is_frozen_to_actual_memory_not_hidden_current_player(game):
    guard = game.aux_guards[0]
    guard.last_seen, guard.last_seen_age = guard.pos + np.array([2., 1.]), 3.
    guard.seeing_player = False
    first = observe_context(game, guard.guard_id, guard)
    game.env.player_pos[:] = (60., 38.)
    assert np.array_equal(first, observe_context(game, guard.guard_id, guard))
    assert np.allclose(first[15:17], [2 / 11.5, 1 / 11.5])


def test_context_uses_guard_and_swat_weights_for_the_correct_campaign_actors(game, swat):
    actor = SecurityGuard('S8', np.array([8., 8.], dtype=np.float32), 0., ((8, 8),))
    assert context_role(game, game.aux_guards[0].guard_id, game.aux_guards[0]) == 'guard'
    assert context_role(game, actor.guard_id, actor) == 'swat'
    uid, unit = next(iter(swat.core.units.items()))
    assert context_role(swat, uid, unit) == 'swat'
    assert observe_context(swat, uid, unit)[9] == 1.


def test_manifest_loads_two_valid_context_models_and_respects_explicit_opt_out(game, tmp_path, monkeypatch):
    guard_path, swat_path = tmp_path / 'guard.zip', tmp_path / 'swat.zip'
    guard_path.write_bytes(b'guard')
    swat_path.write_bytes(b'swat')
    manifest = tmp_path / 'active.json'
    manifest.write_text(json.dumps(dict(version=1, guard=str(guard_path), swat=str(swat_path))), encoding='utf-8')
    monkeypatch.setenv('SHADOW_CONTEXT_MANIFEST', str(manifest))
    monkeypatch.setenv('SHADOW_CONTEXT_MODELS', 'on')
    with patch('stable_baselines3.PPO.load', side_effect=lambda path, **kw: ContextPolicy(1 if path == guard_path else 3)):
        game._reset_context_tactics()
    assert game._context_models['guard'].choice == 1
    assert game._context_models['swat'].choice == 3
    monkeypatch.setenv('SHADOW_CONTEXT_MODELS', 'off')
    with patch('stable_baselines3.PPO.load') as loader:
        game._reset_context_tactics()
    loader.assert_not_called()
    assert not game._context_models


def test_incompatible_or_corrupt_optional_model_does_not_break_the_base_game(game, tmp_path, monkeypatch):
    path = tmp_path / 'guard.zip'
    path.write_bytes(b'bad')
    manifest = tmp_path / 'active.json'
    manifest.write_text(json.dumps(dict(version=1, guard=str(path))), encoding='utf-8')
    monkeypatch.setenv('SHADOW_CONTEXT_MANIFEST', str(manifest))
    monkeypatch.setenv('SHADOW_CONTEXT_MODELS', 'on')
    with patch('stable_baselines3.PPO.load', side_effect=RuntimeError('bad checkpoint')):
        game._reset_context_tactics()
    assert game._context_models == {} and 'bad checkpoint' in game._context_load_error
    model = ContextPolicy(0)
    model.observation_space.shape = (43,)
    with patch('stable_baselines3.PPO.load', return_value=model):
        game._reset_context_tactics()
    assert game._context_models == {}


def test_policy_can_choose_care_and_cannot_abandon_an_application_midway(game):
    patient, _, station = prepare(game, with_medic=False)
    game._context_models = {'guard': ContextPolicy(ContextAction.SEEK_CARE)}
    active, consumed = game._step_context_tactics(patient.guard_id, patient)
    assert active and consumed and patient.vitals.treatment == 'MEDKIT'
    assert not game.pickups and station['id'] in game.medical_watch.consumed
    game._context_models['guard'] = ContextPolicy(ContextAction.CONTINUE)
    game.policy_tick += 10
    active, consumed = game._step_context_tactics(patient.guard_id, patient)
    assert active and consumed and patient.vitals.treatment == 'MEDKIT'
    assert len(game.context_decisions) == 1  # Forced continuity is not a new policy decision.


def test_continue_defers_new_care_but_emergency_and_legacy_fallback_are_preserved(game):
    patient, _, _ = prepare(game, with_medic=False)
    game._context_models = {'guard': ContextPolicy(ContextAction.CONTINUE)}
    game._step_guard(patient, TacticalAction.PATROL)
    assert patient.vitals.treatment is None
    patient.vitals.health = .15
    game._step_guard(patient, TacticalAction.PATROL)
    assert patient.vitals.treatment == 'MEDKIT'
    assert patient.guard_id in game._legacy_policy_overrides


def test_mandatory_hit_cover_preempts_context_choice_and_legacy_credit(game):
    patient, _, _ = prepare(game, with_medic=False)
    patient._incoming_cover = dict(source=np.array([4., 8.]), goal=np.array([11., 8.]), remaining=6., holding=1.)
    game._context_models = {'guard': ContextPolicy(ContextAction.SEEK_CARE)}
    game._step_guard(patient, TacticalAction.PATROL)
    assert patient.vitals.treatment is None and not game.context_decisions
    assert patient.guard_id in game._legacy_policy_overrides


def test_invalid_context_choice_is_not_credited_and_contact_cannot_be_invented(game):
    guard = game.aux_guards[0]
    guard.last_seen, game.team_contact = None, None
    game._context_models = {'guard': ContextPolicy(ContextAction.SEARCH_REPORT)}
    active, consumed = game._step_context_tactics(guard.guard_id, guard)
    assert active and not consumed and not game.context_decisions
    assert not apply_context(game, guard.guard_id, guard, ContextAction.TAKE_COVER)


def test_standalone_swat_care_consumes_shared_kit_and_stops_bleeding(swat):
    uid, unit = next(iter(swat.core.units.items()))
    station = next(item for item in swat.pickups if item.get('fixed'))
    unit.pos = station['pos'].copy()
    unit.local_visual = False
    unit.health = swat._unit_vitals[uid].health = .5
    swat._unit_vitals[uid].bleeding = .5
    assert apply_context(swat, uid, unit, ContextAction.SEEK_CARE)
    assert station not in swat.pickups and station['id'] in swat.medical_watch.consumed
    for _ in range(125):
        if uid not in swat._context_medical_jobs:
            break
        apply_context(swat, uid, unit, ContextAction.SEEK_CARE)
    assert swat._unit_vitals[uid].bleeding == 0 and unit.health >= .6 - 1e-6
    assert swat._unit_vitals[uid].regen_remaining > 0


def test_disarmed_actor_observation_still_has_a_valid_schema(game):
    actor = game.aux_guards[0]
    actor.weapon = None
    actor.ammo = 0
    assert observe_context(game, actor.guard_id, actor).shape == (32,)


@pytest.mark.parametrize('state,choice,age', [
    ('SEARCH REPORT', ContextAction.SEARCH_REPORT, 9.),
    ('TACTICAL COVER', ContextAction.TAKE_COVER, 6.),
    ('COVER MEDIC', ContextAction.COVER_ALLY, 3.),
])
def test_unavailable_context_order_releases_swat_to_bounded_search(game, state, choice, age):
    game.env.grid[:] = 0
    game.lead_alive = False
    game.aux_guards = []
    game.env.player_pos = np.array([40., 30.], dtype=np.float32)
    actor = SecurityGuard('S8', np.array([8., 8.], dtype=np.float32), 0., ((8., 8.),))
    actor.state = state
    actor.last_seen, actor.last_seen_age = np.array([10., 8.], dtype=np.float32), age
    game.swat_units = [('ASSAULT', actor)]
    game._context_models = {'swat': ContextPolicy(choice)}
    _, consumed = game._step_context_tactics(actor.guard_id, actor)
    assert not consumed and actor.state == 'SEARCH'
    assert actor.guard_id in game._searches
    from shadow_ai.playable_navigation import LocalSearch
    game._searches[actor.guard_id] = LocalSearch([], scan_seconds=.01)
    game._step_guard(actor)
    assert actor.state == 'SUPPORT INTEL'
    actor._security_support_remaining = 0.
    game._step_guard(actor)
    assert actor.state == 'WITHDRAW'


def test_expired_context_order_cannot_restart_hidden_sentry_pursuit(game):
    actor = game.aux_guards[0]
    game._publish_contact(np.array([20., 10.], dtype=np.float32), 'S0')
    actor.state = 'SEARCH REPORT'
    actor.last_seen, actor.last_seen_age = np.array([20., 10.], dtype=np.float32), 9.
    actor.seeing_player = False
    game._context_models = {'guard': ContextPolicy(ContextAction.SEARCH_REPORT)}
    _, consumed = game._step_context_tactics(actor.guard_id, actor)
    assert not consumed and actor.state == 'PROTECT INTEL'
    assert actor.guard_id not in game._searches
