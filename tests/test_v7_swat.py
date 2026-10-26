
import numpy as np

from shadow_ai import (
    FacilitySecurityState,
    FacilityAlert,
    DeploymentReason,
    SwatAction,
    SwatDoctrinePretrainEnv,
    SwatParallelEnv,
    SwatRole,
    SwatSharedPolicyEnv,
    swat_deployment_reason,
)


def test_swat_shared_spaces_are_stable():
    env = SwatSharedPolicyEnv(seed=1)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (64,)
    assert env.action_space.n == 14
    assert len(env.core.units) == 5


def test_each_role_exists_once():
    env = SwatSharedPolicyEnv(seed=2)
    env.reset(seed=2)
    roles = [u.role for u in env.core.units.values()]
    assert set(roles) == {
        SwatRole.SHIELD,
        SwatRole.ASSAULT,
        SwatRole.UTILITY,
        SwatRole.HEAVY,
        SwatRole.LEADER,
    }


def test_info_never_exposes_hidden_player_position():
    env = SwatSharedPolicyEnv(seed=3)
    _, info = env.reset(seed=3)
    assert "player_pos" not in info
    assert "player_position" not in info


def test_utility_requires_legitimate_target_not_hidden_truth():
    env = SwatDoctrinePretrainEnv(
        forced_skill="utility_flash",
        seed=4,
    )
    obs, info = env.reset(seed=4)
    assert info["expected_actions"] == ["USE_FLASH"]
    _, _, terminated, truncated, result = env.step(
        int(SwatAction.USE_FLASH)
    )
    assert terminated and not truncated
    assert result["doctrine_correct"] is True


def test_visual_pursue_contract():
    env = SwatDoctrinePretrainEnv(
        forced_skill="visual_pursue",
        seed=5,
    )
    _, info = env.reset(seed=5)
    assert info["expected_actions"] == ["PURSUE"]
    _, _, _, _, result = env.step(int(SwatAction.PURSUE))
    assert result["doctrine_correct"] is True


def test_compliant_arrest_contract():
    env = SwatDoctrinePretrainEnv(
        forced_skill="compliant_arrest",
        seed=6,
    )
    _, _ = env.reset(seed=6)
    _, _, _, _, result = env.step(int(SwatAction.ARREST))
    assert result["doctrine_correct"] is True


def test_parallel_env_has_five_agents():
    env = SwatParallelEnv(seed=7)
    obs, infos = env.reset(seed=7)
    assert len(obs) == 5
    assert len(infos) == 5


def test_deployment_is_event_based_not_hidden_location():
    state = FacilitySecurityState(
        alert=FacilityAlert.CONFIRMED,
        guards_active=2,
        guards_down=0,
        reinforcement_requested=True,
    )
    assert (
        swat_deployment_reason(state)
        == DeploymentReason.CONFIRMED_ESCALATION
    )


def test_multiple_guards_down_escalates():
    state = FacilitySecurityState(
        alert=FacilityAlert.SEARCHING,
        guards_active=1,
        guards_down=2,
    )
    assert (
        swat_deployment_reason(state)
        == DeploymentReason.MULTIPLE_GUARDS_DOWN
    )


def test_shield_front_arc_reduces_incoming_damage():
    env = SwatSharedPolicyEnv(
        seed=8,
        scenario="confirmed_visual",
    )
    env.reset(seed=8)

    shield = env.core._role_unit(SwatRole.SHIELD)
    assert shield is not None

    # Zona con LOS libera.
    shield.pos = np.array(
        [10.0, 8.0],
        dtype=np.float32,
    )
    env.core.player_pos = np.array(
        [13.0, 8.0],
        dtype=np.float32,
    )

    assert env.core.map.line_clear(
        shield.pos,
        env.core.player_pos,
        env.core.smoke_clouds,
    )

    # Shield rivolto verso il player.
    shield.health = 1.0
    shield.angle = 0.0

    before = shield.health
    env.core._player_counterfire()
    shield_front_loss = before - shield.health

    # Stessa posizione/distanza, ma shield rivolto dalla parte opposta.
    shield.health = 1.0
    shield.alive = True
    shield.angle = np.pi

    before = shield.health
    env.core._player_counterfire()
    shield_back_loss = before - shield.health

    assert shield_front_loss > 0.0
    assert shield_back_loss > 0.0
    assert shield_front_loss < shield_back_loss

def test_armor_and_health_are_active_game_state():
    env = SwatSharedPolicyEnv(seed=9, scenario="confirmed_visual")
    env.reset(seed=9)
    heavy = env.core._role_unit(SwatRole.HEAVY)
    assert heavy is not None
    assert heavy.profile.armor > 1.0
    assert heavy.health == 1.0
