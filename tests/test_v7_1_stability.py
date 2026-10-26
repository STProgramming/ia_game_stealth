
import numpy as np

from shadow_ai import (
    SwatAction,
    SwatDoctrinePretrainEnv,
    SwatRole,
    SwatSharedPolicyEnv,
)


def test_flash_and_smoke_curriculum_states_are_distinguishable():
    flash = SwatDoctrinePretrainEnv(
        forced_skill="utility_flash",
        robust_context=False,
        seed=101,
    )
    smoke = SwatDoctrinePretrainEnv(
        forced_skill="utility_smoke",
        robust_context=False,
        seed=101,
    )
    fobs, _ = flash.reset(seed=101)
    sobs, _ = smoke.reset(seed=101)
    assert fobs.shape == sobs.shape == (64,)
    assert not np.allclose(fobs, sobs)


def test_smoke_contract_is_actually_rewardable():
    env = SwatDoctrinePretrainEnv(
        forced_skill="utility_smoke",
        robust_context=False,
        seed=102,
    )
    _, _ = env.reset(seed=102)
    _, reward, _, _, info = env.step(int(SwatAction.USE_SMOKE))
    assert info["doctrine_correct"] is True
    assert reward > 0.0


def test_reported_contact_can_train_pursuit_without_local_visual():
    env = SwatDoctrinePretrainEnv(
        forced_skill="reported_contact_pursue",
        robust_context=False,
        seed=103,
    )
    _, _ = env.reset(seed=103)
    unit = env.core.units[env.uid]
    assert not unit.local_visual
    assert env.core.blackboard.contact_pos is not None
    _, _, _, _, info = env.step(int(SwatAction.PURSUE))
    assert info["doctrine_correct"] is True


def test_reported_compliance_is_in_observation_and_arrest_approaches():
    env = SwatDoctrinePretrainEnv(
        forced_skill="reported_compliant_arrest",
        robust_context=False,
        seed=104,
    )
    obs, _ = env.reset(seed=104)
    unit = env.core.units[env.uid]
    before = float(
        np.linalg.norm(
            unit.pos - env.core.blackboard.contact_pos
        )
    )
    assert env.core.blackboard.player_compliant_known
    _, reward, _, _, info = env.step(int(SwatAction.ARREST))
    after = float(
        np.linalg.norm(
            unit.pos - env.core.blackboard.contact_pos
        )
    )
    assert info["doctrine_correct"] is True
    assert reward > 0.0
    assert after < before


def test_radio_deduplicates_stationary_contact():
    env = SwatSharedPolicyEnv(
        seed=105,
        scenario="arrest",
    )
    env.reset(seed=105)
    leader = env.core._role_unit(SwatRole.LEADER)
    assert leader is not None

    start = env.core.blackboard.radio_updates
    for _ in range(25):
        env.core._sense_unit(leader)
        env.core.blackboard.tick(
            env.core.cfg.dt,
            env.core.cfg.radio_memory_seconds,
        )
    end = env.core.blackboard.radio_updates

    assert end - start <= 2


def test_noncompliant_confirmed_contact_has_health_consequence():
    env = SwatSharedPolicyEnv(
        seed=106,
        scenario="confirmed_visual",
    )
    env.reset(seed=106)
    assault = env.core._role_unit(SwatRole.ASSAULT)
    assert assault is not None

    assault.pos = env.core.player_pos + np.array(
        [-1.0, 0.0],
        dtype=np.float32,
    )
    assault.angle = 0.0
    assault.ammo = max(assault.ammo, 5)
    env.core._sense_unit(assault)
    before = env.core.player_health
    env.core._automatic_contact_resolution()
    assert env.core.player_health < before
