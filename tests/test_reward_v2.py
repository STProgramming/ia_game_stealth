from shadow_ai import ShadowProtocolEnv, TacticalAction


def test_invalid_flank_is_penalized():
    env = ShadowProtocolEnv(seed=1)
    env.reset(seed=1)
    env.last_seen_pos = None
    env.alert = 0
    env.sense.visual = False
    env.sense.sound = False

    decision = env._decision_context()
    previous = env.guard_pos.copy()

    reward = env._reward(
        TacticalAction.FLANK_RIGHT,
        previous,
        decision,
    )
    assert reward < -0.15


def test_routine_patrol_beats_flank():
    env = ShadowProtocolEnv(seed=2)
    env.reset(seed=2)
    env.last_seen_pos = None
    env.alert = 0
    env.sense.visual = False
    env.sense.sound = False

    decision = env._decision_context()
    previous = env.guard_pos.copy()

    patrol = env._reward(
        TacticalAction.PATROL,
        previous,
        decision,
    )
    flank = env._reward(
        TacticalAction.FLANK_LEFT,
        previous,
        decision,
    )
    assert patrol > flank


def test_sound_investigation_needs_sound():
    env = ShadowProtocolEnv(seed=3)
    env.reset(seed=3)
    env.sense.visual = False
    env.sense.sound = False

    previous = env.guard_pos.copy()
    invalid = env._reward(
        TacticalAction.INVESTIGATE_SOUND,
        previous,
        env._decision_context(),
    )

    env.sense.sound = True
    env.sense.sound_rel = (1.0, 0.0)
    env.sense.sound_confidence = 0.8

    valid = env._reward(
        TacticalAction.INVESTIGATE_SOUND,
        previous,
        env._decision_context(),
    )
    assert valid > invalid


def test_arrest_requires_compliance():
    env = ShadowProtocolEnv(seed=4)
    env.reset(seed=4)

    previous = env.guard_pos.copy()

    env.sense.visual = False
    env.player_compliant = False
    invalid = env._reward(
        TacticalAction.ARREST,
        previous,
        env._decision_context(),
    )

    env.sense.visual = True
    env.sense.visual_rel = (0.5, 0.0)
    env.player_compliant = True
    valid = env._reward(
        TacticalAction.ARREST,
        previous,
        env._decision_context(),
    )

    assert valid > invalid
