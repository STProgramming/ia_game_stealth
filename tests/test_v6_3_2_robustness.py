from shadow_ai import (
    DoctrinePretrainEnv,
    TacticalAction,
)


def _reward(env, action):
    d = env._decision_context()
    return env._reward(action, env.guard_pos.copy(), d)


def test_repeat_attention_reward_really_prefers_reinforcements():
    env = DoctrinePretrainEnv(
        forced_skill="attention_repeat3",
        seed=20,
    )
    env.reset(seed=20)

    assert (
        _reward(env, TacticalAction.CALL_REINFORCEMENTS)
        > _reward(env, TacticalAction.FOCUS_VISUAL)
    )


def test_robust_visual_reported_keeps_visual_phase_and_pursuit_contract():
    env = DoctrinePretrainEnv(
        forced_skill="visual_reported",
        robust_context=True,
        seed=21,
    )

    for seed in range(21, 61):
        _, info = env.reset(seed=seed)
        assert info["tactical_phase"] == "visual"
        assert env.radio_reported is True

        _, _, _, _, result = env.step(
            int(TacticalAction.PURSUE)
        )
        assert result["doctrine_correct"] is True


def test_robust_visual_unreported_keeps_radio_contract():
    env = DoctrinePretrainEnv(
        forced_skill="visual_unreported",
        robust_context=True,
        seed=22,
    )

    for seed in range(22, 62):
        _, info = env.reset(seed=seed)
        assert info["tactical_phase"] == "visual_unreported"
        assert env.radio_reported is False

        _, _, _, _, result = env.step(
            int(TacticalAction.RADIO_REPORT)
        )
        assert result["doctrine_correct"] is True


def test_robust_arrest_outranks_radio_and_theft_distractors():
    env = DoctrinePretrainEnv(
        forced_skill="arrest",
        robust_context=True,
        seed=23,
    )

    for seed in range(23, 63):
        _, info = env.reset(seed=seed)
        assert info["tactical_phase"] == "arrest"

        _, _, _, _, result = env.step(
            int(TacticalAction.ARREST)
        )
        assert result["doctrine_correct"] is True


def test_robust_last_seen_stays_last_seen_with_distractors():
    env = DoctrinePretrainEnv(
        forced_skill="last_seen",
        robust_context=True,
        seed=24,
    )

    for seed in range(24, 64):
        _, info = env.reset(seed=seed)
        assert info["tactical_phase"] == "last_seen"


def test_robust_search_stays_search_with_distractors():
    env = DoctrinePretrainEnv(
        forced_skill="search",
        robust_context=True,
        seed=25,
    )

    for seed in range(25, 65):
        _, info = env.reset(seed=seed)
        assert info["tactical_phase"] == "search"


def test_robust_repeat_sound_keeps_support_contract():
    env = DoctrinePretrainEnv(
        forced_skill="repeat_sound",
        robust_context=True,
        seed=26,
    )

    for seed in range(26, 66):
        _, info = env.reset(seed=seed)
        assert info["tactical_phase"] == "sound_tactical"
        assert env.buddy_available
        assert not env.buddy_supporting

        _, _, _, _, result = env.step(
            int(TacticalAction.REQUEST_LOCAL_SUPPORT)
        )
        assert result["doctrine_correct"] is True
