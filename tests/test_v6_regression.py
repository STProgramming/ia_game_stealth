import numpy as np

from shadow_ai import ShadowProtocolEnv, TacticalAction


def _clear_visual(env: ShadowProtocolEnv) -> None:
    env.sense.visual = False
    env.sense.visual_candidate = False
    env.sense.visual_confidence = 0.0
    env.sense.visual_rel = (0.0, 0.0)
    env.visual_confidence = 0.0


def test_v6_observation_contract_is_43_features():
    env = ShadowProtocolEnv(seed=123)
    obs, _ = env.reset(seed=123)
    assert obs.shape == (43,)
    assert env.observation_space.shape == (43,)


def test_invalid_flank_is_negative_in_true_routine():
    env = ShadowProtocolEnv(seed=1)
    env.reset(seed=1, options={'scenario': 'routine'})
    _clear_visual(env)
    env.sense.sound = False
    env.last_seen_pos = None
    env.alert = 0

    decision = env._decision_context()
    assert env._tactical_phase(decision) == 'routine'

    reward = env._reward(
        TacticalAction.FLANK_RIGHT,
        env.guard_pos.copy(),
        decision,
    )
    assert reward < 0.0


def test_sound_investigation_beats_investigation_without_sound():
    env = ShadowProtocolEnv(seed=3)
    env.reset(seed=3, options={'scenario': 'routine'})
    _clear_visual(env)
    env.last_seen_pos = None
    env.alert = 0
    previous = env.guard_pos.copy()

    env.sense.sound = False
    invalid_decision = env._decision_context()
    invalid = env._reward(
        TacticalAction.INVESTIGATE_SOUND,
        previous,
        invalid_decision,
    )

    env.sense.sound = True
    env.sense.sound_rel = (1.0, 0.0)
    env.sense.sound_confidence = 0.8
    valid_decision = env._decision_context()
    assert env._tactical_phase(valid_decision) in {'sound', 'sound_tactical'}

    valid = env._reward(
        TacticalAction.INVESTIGATE_SOUND,
        previous,
        valid_decision,
    )
    assert valid > invalid


def test_last_seen_prefers_memory_action_over_blind_pursue():
    env = ShadowProtocolEnv(seed=4)
    env.reset(seed=4, options={'scenario': 'routine'})
    _clear_visual(env)
    env.sense.sound = False
    env.last_seen_pos = env.guard_pos + np.array([3.0, 0.0], dtype=np.float32)
    env.last_seen_velocity[:] = 0.0
    env.last_seen_age = 1.0
    env.alert = 2

    decision = env._decision_context()
    assert env._tactical_phase(decision) == 'last_seen'
    previous = env.guard_pos.copy()

    go_last = env._reward(
        TacticalAction.GO_LAST_SEEN,
        previous,
        decision,
    )
    pursue = env._reward(
        TacticalAction.PURSUE,
        previous,
        decision,
    )
    assert go_last > pursue


def test_arrest_still_requires_explicit_arrest_action():
    env = ShadowProtocolEnv(seed=5)
    env.reset(seed=5, options={'scenario': 'arrest'})

    env.player_compliant = True
    env.visual_confidence = 1.0
    env.sense = env._sense()
    env._update_knowledge()
    decision = env._decision_context()

    if not decision['visual']:
        # Create a deterministic close/visible configuration for the contract.
        env.guard_pos = np.array([10.0, 8.0], dtype=np.float32)
        env.player_pos = np.array([10.7, 8.0], dtype=np.float32)
        env.guard_angle = 0.0
        env.player_compliant = True
        env.visual_confidence = 1.0
        env.sense = env._sense()
        env._update_knowledge()
        decision = env._decision_context()

    assert decision['visual']
    terminal_reward, terminated = env._terminal_events(
        TacticalAction.PURSUE,
        decision,
    )
    assert not terminated
    assert terminal_reward == 0.0
