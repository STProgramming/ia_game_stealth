from shadow_ai import (
    DoctrinePretrainEnv,
    ShadowProtocolEnv,
    TacticalAction,
)


def _reward_for(env, action):
    decision = env._decision_context()
    return env._reward(
        action,
        env.guard_pos.copy(),
        decision,
    )


def test_last_seen_prefers_intercept_over_search():
    env = DoctrinePretrainEnv(forced_skill="last_seen", seed=1)
    _, info = env.reset(seed=1)

    assert info["tactical_phase"] == "last_seen"
    assert (
        _reward_for(env, TacticalAction.INTERCEPT)
        > _reward_for(env, TacticalAction.SEARCH_LAST_SEEN)
    )


def test_search_prefers_search_over_intercept():
    env = DoctrinePretrainEnv(forced_skill="search", seed=2)
    _, info = env.reset(seed=2)

    assert info["tactical_phase"] == "search"
    assert (
        _reward_for(env, TacticalAction.SEARCH_LAST_SEEN)
        > _reward_for(env, TacticalAction.INTERCEPT)
    )


def test_confirmed_visual_reports_once_then_pursues():
    env = DoctrinePretrainEnv(
        forced_skill="visual_unreported",
        seed=3,
    )
    _, info = env.reset(seed=3)

    assert info["tactical_phase"] == "visual_unreported"
    assert (
        _reward_for(env, TacticalAction.RADIO_REPORT)
        > _reward_for(env, TacticalAction.PURSUE)
    )

    env._apply_action(TacticalAction.RADIO_REPORT)
    assert env.radio_reported is True
    assert env._tactical_phase(env._decision_context()) == "visual"
    assert (
        _reward_for(env, TacticalAction.PURSUE)
        > _reward_for(env, TacticalAction.RADIO_REPORT)
    )


def test_repeat_sound_prefers_local_support_when_buddy_available():
    env = DoctrinePretrainEnv(forced_skill="repeat_sound", seed=4)
    _, info = env.reset(seed=4)

    assert info["tactical_phase"] == "sound_tactical"
    assert (
        _reward_for(env, TacticalAction.REQUEST_LOCAL_SUPPORT)
        > _reward_for(env, TacticalAction.INVESTIGATE_SOUND)
    )
    assert (
        _reward_for(env, TacticalAction.REQUEST_LOCAL_SUPPORT)
        > _reward_for(env, TacticalAction.PROTECT_NEAREST_INTEL)
    )


def test_visual_reported_micro_skill_is_pursuit():
    env = DoctrinePretrainEnv(
        forced_skill="visual_reported",
        seed=5,
    )
    _, info = env.reset(seed=5)

    assert info["tactical_phase"] == "visual"
    _, _, terminated, truncated, result = env.step(
        int(TacticalAction.PURSUE)
    )
    assert terminated and not truncated
    assert result["doctrine_correct"] is True


def test_visual_unreported_micro_skill_is_radio_report():
    env = DoctrinePretrainEnv(
        forced_skill="visual_unreported",
        seed=6,
    )
    _, info = env.reset(seed=6)

    assert info["tactical_phase"] == "visual_unreported"
    _, _, terminated, truncated, result = env.step(
        int(TacticalAction.RADIO_REPORT)
    )
    assert terminated and not truncated
    assert result["doctrine_correct"] is True


def test_spaces_stay_compatible_with_v6_models():
    doctrine = DoctrinePretrainEnv(seed=7)
    full = ShadowProtocolEnv(seed=7)

    assert doctrine.observation_space.shape == (43,)
    assert full.observation_space.shape == (43,)
    assert doctrine.action_space.n == 16
    assert full.action_space.n == 16
