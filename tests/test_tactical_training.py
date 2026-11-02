from unittest.mock import patch

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from shadow_ai.tactical_context import ContextAction, observe_context
from shadow_ai.tactical_training_env import TacticalTrainingEnv, SCENARIOS, STANDALONE_SCENARIOS, expected_context_action
from shadow_ai.soliton_security import line_clear


class HoldPolicy:
    def predict(self, observation, **kwargs):
        if np.asarray(observation).ndim > 1:
            return np.zeros(len(observation), dtype=int), None
        return 0, None


@pytest.fixture(scope='module', params=['guard', 'swat'])
def rehearsal(request):
    with patch('shadow_ai.interactive_runtime.PPO.load', return_value=HoldPolicy()):
        return TacticalTrainingEnv(mode=request.param, seed=331, horizon=64, training=False)


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_all_rehearsals_use_finite_shared_schema_and_real_walkable_geometry(rehearsal, scenario):
    observation, info = rehearsal.reset(seed=70113, options={'scenario': scenario})
    assert rehearsal.observation_space.contains(observation)
    assert observation.dtype == np.float32 and observation.shape == (32,)
    assert info['scenario'] == scenario
    assert 1 <= info['level'] <= 3
    assert rehearsal.runtime.env._free(*rehearsal.runtime.env._cell(rehearsal.actor.pos))
    _, reward, terminated, truncated, result = rehearsal.step(info['expected_action'])
    assert np.isfinite(reward) and not terminated and not truncated
    assert result['action_correct'] and result['action_accuracy'] == 1.


def test_seed_reproduces_observation_scenario_resources_and_action_target(rehearsal):
    first, first_info = rehearsal.reset(seed=543211)
    point = rehearsal.actor.pos.copy()
    second, second_info = rehearsal.reset(seed=543211)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(point, rehearsal.actor.pos)
    assert first_info == second_info


def test_unobserved_player_movement_and_unreported_theft_do_not_enter_policy_observation(rehearsal):
    first, _ = rehearsal.reset(seed=12311, options={'scenario': 'no_report'})
    runtime = rehearsal.runtime
    runtime.env.player_pos = np.array([55., 36.], dtype=np.float32)
    runtime.env.player_velocity[:] = [20., -20.]
    runtime.env.player_has_intel[:] = 1
    second = observe_context(runtime, rehearsal.actor.guard_id, rehearsal.actor)
    np.testing.assert_array_equal(first, second)
    assert expected_context_action(second) == ContextAction.CONTINUE


@pytest.mark.parametrize('scenario', ['wounded_station', 'wounded_medic', 'wounded_personal_kit'])
def test_medical_action_physically_heals_and_spends_one_finite_supply(rehearsal, scenario):
    _, _ = rehearsal.reset(seed=852, options={'scenario': scenario})
    runtime, actor = rehearsal.runtime, rehearsal.actor
    initial_health = actor.vitals.health
    initial_kits = sum(g.medkits for g in runtime.aux_guards + [u for _, u in runtime.swat_units])
    for _ in range(64):
        observation, _, _, _, info = rehearsal.step(expected_context_action(observe_context(runtime, actor.guard_id, actor)))
        if actor.vitals.bleeding == 0:
            break
    assert actor.vitals.bleeding == 0
    assert actor.vitals.health >= initial_health + .099
    assert actor.vitals.regen_remaining > 0
    assert info['physical_success'] and info['physical_applicable']
    if scenario == 'wounded_station':
        assert not runtime.pickups
        assert 'training-shared-kit' in runtime.medical_watch.consumed
        assert runtime.medical_watch.goal(actor.guard_id, actor.pos, False, runtime.navigator.path_length) is None
    else:
        assert sum(g.medkits for g in runtime.aux_guards + [u for _, u in runtime.swat_units]) == initial_kits - 1
    assert expected_context_action(observation) == ContextAction.CONTINUE


def test_spent_station_cannot_heal_or_create_a_false_physical_success(rehearsal):
    _, info = rehearsal.reset(seed=852, options={'scenario': 'spent_station'})
    before = rehearsal.actor.vitals.health
    for _ in range(4):
        _, _, _, _, info = rehearsal.step(ContextAction.SEEK_CARE)
    assert not info['physical_success'] and not info['consumed_step']
    assert not info['physical_applicable']
    assert rehearsal.actor.vitals.health == before


def test_camera_scenarios_generate_actual_body_evidence_and_theft_radio(rehearsal):
    _, info = rehearsal.reset(seed=19343, options={'scenario': 'camera_body'})
    runtime = rehearsal.runtime
    assert runtime.bodies[0].discovered
    assert runtime.security_cycle.phase == 'SUSPICIOUS'
    assert any('Telecamera' in item['text'] for item in runtime.radio_log.entries)
    assert info['expected_action'] == ContextAction.SECURE_INTEL
    _, info = rehearsal.reset(seed=19343, options={'scenario': 'camera_theft'})
    assert runtime.level == 2 and runtime._reported_missing_intel
    assert any("rubato l'intel" in item['text'] for item in runtime.radio_log.entries)
    assert info['expected_action'] == ContextAction.SECURE_INTEL


def test_cover_moves_away_from_only_the_remembered_source_without_crossing_walls(rehearsal):
    _, _ = rehearsal.reset(seed=39147, options={'scenario': 'empty_weapon_contact'})
    actor, runtime = rehearsal.actor, rehearsal.runtime
    source, start = actor.last_seen.copy(), actor.pos.copy()
    for _ in range(12):
        before = actor.pos.copy()
        _, _, _, _, info = rehearsal.step(ContextAction.TAKE_COVER)
        assert line_clear(runtime.env.grid, before, actor.pos)
    assert np.linalg.norm(actor.pos - source) > np.linalg.norm(start - source) + .3
    assert info['physical_success'] and info['physical_progress'] > .3


def test_action_target_and_physical_progress_are_reported_separately(rehearsal):
    _, info = rehearsal.reset(seed=953, options={'scenario': 'routine'})
    assert info['expected_action'] == ContextAction.CONTINUE
    _, reward, _, _, result = rehearsal.step(ContextAction.CONTINUE)
    assert reward == 1. and result['action_correct']
    assert result['physical_progress'] == 0 and not result['physical_applicable']
    rehearsal.reset(seed=953, options={'scenario': 'routine'})
    _, wrong_reward, _, _, result = rehearsal.step(ContextAction.SEARCH_REPORT)
    assert wrong_reward < 0 and not result['action_correct']
    assert not result['consumed_step']


@pytest.fixture(scope='module')
def standalone():
    with patch('shadow_ai.interactive_runtime.PPO.load', return_value=HoldPolicy()):
        return TacticalTrainingEnv(mode='swat', seed=331, horizon=64, training=False)


@pytest.mark.parametrize('scenario', STANDALONE_SCENARIOS)
def test_standalone_swat_rehearsals_execute_real_core_units(standalone, scenario):
    observation, info = standalone.reset(seed=25212, options={'scenario': scenario, 'backend': 'standalone'})
    assert info['backend'] == 'standalone' and observation[9] == 1
    assert not hasattr(standalone.runtime, 'env')
    assert standalone.observation_space.contains(observation)
    _, _, terminated, _, result = standalone.step(info['expected_action'])
    assert not terminated and result['action_correct']


def test_standalone_swat_consumes_shared_station_and_completes_real_treatment(standalone):
    observation, _ = standalone.reset(seed=2852, options={'scenario': 'wounded_station', 'backend': 'standalone'})
    runtime, actor = standalone.runtime, standalone.actor
    before = actor.health
    for _ in range(64):
        observation, _, _, _, info = standalone.step(expected_context_action(observation))
        if actor.vitals.bleeding == 0:
            break
    assert actor.health >= before + .099 and actor.vitals.bleeding == 0
    assert not runtime.pickups and 'training-shared-kit' in runtime.medical_watch.consumed
    assert info['physical_success']
    assert expected_context_action(observation) == ContextAction.CONTINUE


def test_standalone_observation_ignores_unseen_player_positions(standalone):
    observation, _ = standalone.reset(seed=1778, options={'scenario': 'no_report', 'backend': 'standalone'})
    standalone.runtime.core.player_pos[:] = [25., 10.]
    actual = observe_context(standalone.runtime, standalone.actor.guard_id, standalone.actor)
    np.testing.assert_array_equal(observation, actual)


@pytest.mark.parametrize('action', list(ContextAction))
@pytest.mark.parametrize('backend', ['campaign', 'standalone'])
def test_every_action_is_safe_even_when_not_available_in_current_backend(standalone, action, backend):
    standalone.reset(seed=28453, options={'scenario': 'routine', 'backend': backend})
    observation, reward, _, _, info = standalone.step(action)
    assert standalone.observation_space.contains(observation) and np.isfinite(reward)
    assert info['physical_progress'] == 0 if action != ContextAction.SECURE_INTEL else info['physical_progress'] >= 0
    if backend == 'standalone' and action == ContextAction.SECURE_INTEL:
        assert not info['consumed_step'] and info['physical_progress'] == 0


def test_random_swat_rollout_includes_every_action_on_every_available_scenario_and_backend(standalone):
    rng = np.random.default_rng(71153)
    for backend, scenarios in [('campaign', SCENARIOS), ('standalone', STANDALONE_SCENARIOS)]:
        for scenario in scenarios:
            standalone.reset(seed=71153, options={'scenario': scenario, 'backend': backend})
            for action in rng.permutation(len(ContextAction)):
                observation, reward, _, _, _ = standalone.step(int(action))
                assert standalone.observation_space.contains(observation) and np.isfinite(reward)


def test_mixed_swat_environment_satisfies_gymnasium_step_reset_contract():
    with patch('shadow_ai.interactive_runtime.PPO.load', return_value=HoldPolicy()):
        mixed = TacticalTrainingEnv(mode='swat', seed=393, training=True)
    check_env(mixed, skip_render_check=True)
    mixed.close()


@pytest.mark.parametrize('scenario', ['recovering_station', 'recovering_medic', 'recovering_personal_kit'])
def test_recovering_actors_do_not_consume_a_second_kit(rehearsal, scenario):
    observation, info = rehearsal.reset(seed=92453, options={'scenario': scenario})
    runtime, actor = rehearsal.runtime, rehearsal.actor
    stock = sum(g.medkits for g in runtime.aux_guards + [u for _, u in runtime.swat_units])
    stations = [item['id'] for item in runtime.pickups]
    before = actor.vitals.health
    assert observation[30] > 0 and observation[2] == 0
    assert info['expected_action'] == ContextAction.CONTINUE
    for _ in range(24):
        observation, _, _, _, info = rehearsal.step(expected_context_action(observation))
    assert actor.vitals.health > before
    assert not runtime._medical_jobs and actor.vitals.treatment is None
    assert sum(g.medkits for g in runtime.aux_guards + [u for _, u in runtime.swat_units]) == stock
    assert [item['id'] for item in runtime.pickups] == stations
    assert expected_context_action(observation) == ContextAction.CONTINUE


def test_standalone_recovery_leaves_shared_station_for_other_actors(standalone):
    observation, info = standalone.reset(seed=92453, options={'scenario': 'recovering_station', 'backend': 'standalone'})
    for _ in range(24):
        observation, _, _, _, info = standalone.step(expected_context_action(observation))
    assert info['expected_action'] == ContextAction.CONTINUE
    assert standalone.runtime.pickups and not standalone.runtime._context_medical_jobs
