import numpy as np

from shadow_ai import ShadowProtocolEnv, TacticalAction


def test_search_has_real_sector_target():
    env = ShadowProtocolEnv(seed=1)
    env.reset(
        seed=1,
        options={"scenario": "search"},
    )

    anchor = env.last_seen_pos.copy()
    target = env._current_search_target()

    assert target is not None
    assert np.linalg.norm(target - anchor) > 0.5


def test_search_action_advances_waypoint_when_reached():
    env = ShadowProtocolEnv(seed=2)
    env.reset(
        seed=2,
        options={"scenario": "search"},
    )

    target = env._current_search_target()
    assert target is not None

    env.guard_pos = target.copy()
    before = env.search_waypoints_visited
    env._advance_search_pattern()

    assert env.search_waypoints_visited > before


def test_reinforcements_physically_spawn_after_delay():
    env = ShadowProtocolEnv(seed=3)
    env.reset(
        seed=3,
        options={"scenario": "theft"},
    )

    env.reinforcements_called = True
    env.reinforcement_timer = 0.0

    steps = int(
        env.cfg.reinforcement_delay_seconds / env.cfg.dt
    ) + 2

    for _ in range(steps):
        env._step_reinforcements()

    assert env.reinforcements_active
    assert (
        len(env.reinforcement_positions)
        == env.cfg.reinforcement_count
    )


def test_theft_prefers_call_before_team_is_called():
    env = ShadowProtocolEnv(seed=4)
    env.reset(
        seed=4,
        options={"scenario": "theft"},
    )

    env.sense.visual = False
    env.sense.sound = False
    env.last_seen_pos = None
    decision = env._decision_context()
    previous = env.guard_pos.copy()

    call = env._reward(
        TacticalAction.CALL_REINFORCEMENTS,
        previous,
        decision,
    )
    protect = env._reward(
        TacticalAction.PROTECT_NEAREST_INTEL,
        previous,
        decision,
    )

    assert call > protect


def test_theft_prefers_protect_after_team_is_called():
    env = ShadowProtocolEnv(seed=5)
    env.reset(
        seed=5,
        options={"scenario": "theft"},
    )

    env.reinforcements_called = True
    env.sense.visual = False
    env.sense.sound = False
    env.last_seen_pos = None
    decision = env._decision_context()
    previous = env.guard_pos.copy()

    call = env._reward(
        TacticalAction.CALL_REINFORCEMENTS,
        previous,
        decision,
    )
    protect = env._reward(
        TacticalAction.PROTECT_NEAREST_INTEL,
        previous,
        decision,
    )

    assert protect > call


def test_info_still_hides_live_player_position():
    env = ShadowProtocolEnv(seed=6)
    _, info = env.reset(seed=6)

    assert "player_pos" not in info
    assert "player_position" not in info
