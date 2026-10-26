import numpy as np

from shadow_ai import (
    DoctrinePretrainEnv,
    TacticalAction,
)


def _reward_for(env, action):
    decision = env._decision_context()
    return env._reward(
        action,
        env.guard_pos.copy(),
        decision,
    )


def test_decision_context_does_not_create_search_waypoints():
    env = DoctrinePretrainEnv(
        forced_skill="last_seen",
        seed=11,
    )
    env.reset(seed=11)

    assert env.search_waypoints == []

    first = env._decision_context()
    assert first["search_active"] is False
    assert first["search_target"] is None
    assert env.search_waypoints == []

    second = env._decision_context()
    assert second["search_active"] is False
    assert second["search_target"] is None
    assert env.search_waypoints == []


def test_last_seen_reward_prefers_intercept_after_info_read():
    env = DoctrinePretrainEnv(
        forced_skill="last_seen",
        seed=12,
    )
    _, info = env.reset(seed=12)

    # Reading reset/info must not mutate the phase.
    assert info["tactical_phase"] == "last_seen"
    assert env._tactical_phase(env._decision_context()) == "last_seen"

    assert (
        _reward_for(env, TacticalAction.INTERCEPT)
        > _reward_for(env, TacticalAction.SEARCH_LAST_SEEN)
    )


def test_search_action_cannot_start_sector_pattern_from_far_away():
    env = DoctrinePretrainEnv(
        forced_skill="last_seen",
        seed=13,
    )
    env.reset(seed=13)

    assert np.linalg.norm(
        env.guard_pos - env.last_seen_pos
    ) > 1.35
    assert env.search_waypoints == []

    target = env._choose_target(
        TacticalAction.SEARCH_LAST_SEEN
    )

    assert target is None
    assert env.search_waypoints == []
    assert env._tactical_phase(env._decision_context()) == "last_seen"


def test_search_pattern_can_start_after_reaching_memory():
    env = DoctrinePretrainEnv(
        forced_skill="last_seen",
        seed=14,
    )
    env.reset(seed=14)

    env.guard_pos = env.last_seen_pos.copy()

    assert env._tactical_phase(env._decision_context()) == "search"
    target = env._choose_target(
        TacticalAction.SEARCH_LAST_SEEN
    )

    assert target is not None
    assert len(env.search_waypoints) > 0


def test_legacy_visual_skill_alias_maps_to_reported_visual():
    env = DoctrinePretrainEnv(
        forced_skill="visual",
        seed=15,
    )
    _, info = env.reset(seed=15)

    assert info["tactical_phase"] == "visual"

    _, _, terminated, truncated, result = env.step(
        int(TacticalAction.PURSUE)
    )

    assert terminated
    assert not truncated
    assert result["doctrine_correct"] is True
