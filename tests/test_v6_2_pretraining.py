import pytest

from shadow_ai import (
    DoctrinePretrainEnv,
    ShadowProtocolEnv,
    TacticalAction,
)


@pytest.mark.parametrize(
    "skill,expected_phase,correct_action",
    [
        ("routine", "routine", TacticalAction.PATROL),
        ("peripheral_hint", "peripheral_hint", TacticalAction.FOCUS_VISUAL),
        ("attention", "attention", TacticalAction.FOCUS_VISUAL),
        (
            "attention_repeat3",
            "attention",
            TacticalAction.CALL_REINFORCEMENTS,
        ),
        (
            "suspicious_visual",
            "suspicious_visual",
            TacticalAction.INVESTIGATE_VISUAL,
        ),
        (
            "suspicious_known",
            "suspicious_visual",
            TacticalAction.REQUEST_LOCAL_SUPPORT,
        ),
        (
            "probable_visual",
            "probable_visual",
            TacticalAction.INVESTIGATE_VISUAL,
        ),
        (
            "probable_known",
            "probable_visual",
            TacticalAction.RADIO_REPORT,
        ),
        ("sound", "sound", TacticalAction.INVESTIGATE_SOUND),
        (
            "repeat_sound",
            "sound_tactical",
            TacticalAction.REQUEST_LOCAL_SUPPORT,
        ),
        ("visual", "visual", TacticalAction.PURSUE),
        ("arrest", "arrest", TacticalAction.ARREST),
        ("last_seen", "last_seen", TacticalAction.INTERCEPT),
        ("search", "search", TacticalAction.SEARCH_LAST_SEEN),
        (
            "theft_pre_call",
            "theft",
            TacticalAction.CALL_REINFORCEMENTS,
        ),
        (
            "theft_post_call",
            "theft",
            TacticalAction.PROTECT_NEAREST_INTEL,
        ),
    ],
)
def test_doctrine_skill_contract(skill, expected_phase, correct_action):
    env = DoctrinePretrainEnv(forced_skill=skill, seed=100)
    obs, info = env.reset(seed=100)

    assert obs.shape == (43,)
    assert info["tactical_phase"] == expected_phase

    _, reward, terminated, truncated, result = env.step(
        int(correct_action)
    )

    assert terminated
    assert not truncated
    assert result["doctrine_correct"] is True
    assert reward > 0.0


def test_pretrain_spaces_match_full_environment():
    doctrine = DoctrinePretrainEnv(seed=1)
    full = ShadowProtocolEnv(seed=1)

    assert doctrine.observation_space == full.observation_space
    assert doctrine.action_space == full.action_space


def test_wrong_doctrine_action_is_worse_than_correct_action():
    correct_env = DoctrinePretrainEnv(
        forced_skill="suspicious_visual",
        seed=3,
    )
    wrong_env = DoctrinePretrainEnv(
        forced_skill="suspicious_visual",
        seed=3,
    )

    correct_env.reset(seed=3)
    wrong_env.reset(seed=3)

    _, correct_reward, *_ = correct_env.step(
        int(TacticalAction.INVESTIGATE_VISUAL)
    )
    _, wrong_reward, *_ = wrong_env.step(
        int(TacticalAction.PATROL)
    )

    assert correct_reward > wrong_reward


def test_doctrine_visual_state_does_not_expose_hidden_player_position():
    env = DoctrinePretrainEnv(
        forced_skill="attention",
        seed=4,
    )
    _, info = env.reset(seed=4)

    assert "player_pos" not in info
    assert "player_position" not in info
