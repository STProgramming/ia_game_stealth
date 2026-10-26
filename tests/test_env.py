from stable_baselines3.common.env_checker import check_env

from shadow_ai import ShadowProtocolEnv


def test_sb3_env_contract():
    env = ShadowProtocolEnv(seed=123)
    check_env(env, warn=True)


def test_hidden_position_is_not_exposed_in_info():
    env = ShadowProtocolEnv(seed=123)
    _, info = env.reset()
    assert "player_pos" not in info
    assert "player_position" not in info


def test_observation_shape():
    env = ShadowProtocolEnv(seed=123)
    obs, _ = env.reset()
    assert obs.shape == (26,)
