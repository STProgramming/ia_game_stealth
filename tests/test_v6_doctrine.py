import math
import numpy as np

from shadow_ai import (
    PlayerStance,
    ShadowProtocolEnv,
    SWAT_PROFILES,
    SwatRole,
    TacticalAction,
)


def test_outer_sprint_spots_faster_than_crouched_slow():
    env = ShadowProtocolEnv(seed=1)
    env.reset(seed=1, options={'scenario': 'attention'})
    env.player_pos = env.guard_pos + np.array([9.0, 0.0], dtype=np.float32)
    env.guard_angle = 0.0

    env.player_stance = PlayerStance.CROUCHED
    env.player_current_speed = env.cfg.player_stealth_speed / env.cfg.dt
    slow = env._spot_rate(9.0, False, 0.15)

    env.player_stance = PlayerStance.STANDING
    env.player_current_speed = env.cfg.player_sprint_speed / env.cfg.dt
    fast = env._spot_rate(9.0, False, 1.0)
    assert fast > slow * 2.0


def test_peripheral_motion_matters_strongly():
    env = ShadowProtocolEnv(seed=2)
    env.reset(seed=2, options={'scenario': 'peripheral'})
    env.player_stance = PlayerStance.CROUCHED
    slow = env._spot_rate(5.5, True, 0.05)
    env.player_stance = PlayerStance.STANDING
    fast = env._spot_rate(5.5, True, 1.0)
    assert fast > slow * 4.0


def test_immediate_zone_is_instant_confirmation():
    env = ShadowProtocolEnv(seed=3)
    env.reset(seed=3, options={'scenario': 'visual'})
    env.player_pos = env.guard_pos + np.array([0.7, 0.0], dtype=np.float32)
    env.guard_angle = 0.0
    env.visual_confidence = 0.0
    packet = env._sense()
    assert packet.visual
    assert packet.visual_confidence == 1.0


def test_known_intruder_reacquisition_starts_from_half_confidence():
    env = ShadowProtocolEnv(seed=4)
    env.reset(seed=4, options={'scenario': 'attention'})
    env.intruder_known = True
    env.visual_confidence = 0.0
    # Use a guaranteed clear corridor. The previous (5,5)->(11,5)
    # crossed the arena wall at x=7, so visual_candidate=False was correct.
    env.guard_pos = np.array([14.5, 1.0], dtype=np.float32)
    env.player_pos = np.array([20.5, 1.0], dtype=np.float32)
    env.guard_angle = 0.0
    packet = env._sense()
    assert packet.visual_candidate
    assert packet.visual_confidence >= 0.50


def test_known_intruder_still_cannot_see_through_wall():
    env = ShadowProtocolEnv(seed=44)
    env.reset(seed=44, options={'scenario': 'attention'})
    env.intruder_known = True
    env.visual_confidence = 0.0
    # This line intentionally crosses the x=7 wall at y=5.
    env.guard_pos = np.array([5.0, 5.0], dtype=np.float32)
    env.player_pos = np.array([11.0, 5.0], dtype=np.float32)
    env.guard_angle = 0.0
    packet = env._sense()
    assert not packet.visual_candidate
    assert packet.visual_confidence < 0.50


def test_focus_is_stronger_at_night_with_flashlight():
    env = ShadowProtocolEnv(seed=5)
    env.reset(seed=5, options={'scenario': 'attention'})
    env.player_stance = PlayerStance.STANDING
    env.focus_visual_timer = 1.0
    env.is_night = False
    day = env._spot_rate(8.0, False, 0.5)
    env.is_night = True
    night = env._spot_rate(8.0, False, 0.5)
    assert night > day


def test_request_local_support_moves_buddy_into_support_mode():
    env = ShadowProtocolEnv(seed=6)
    env.reset(seed=6, options={'scenario': 'repeat_visual'})
    env.buddy_available = True
    env.buddy_pos = env.guard_pos + np.array([2.0, 0.0], dtype=np.float32)
    env.visual_anomaly_pos = env.guard_pos + np.array([4.0, 0.0], dtype=np.float32)
    env._apply_action(TacticalAction.REQUEST_LOCAL_SUPPORT)
    assert env.buddy_supporting
    assert env.buddy_target is not None


def test_swat_profiles_trade_speed_for_equipment_and_armor():
    shield = SWAT_PROFILES[SwatRole.SHIELD]
    heavy = SWAT_PROFILES[SwatRole.HEAVY]
    assert shield.has_shield
    assert shield.weapon == 'pistol'
    assert shield.move_multiplier < 1.0
    assert heavy.move_multiplier < 1.0
    assert heavy.armor > 1.0
    assert heavy.ammo_multiplier > 1.0
    assert heavy.weapon == 'shotgun'


def test_info_never_exposes_hidden_player_coordinates():
    env = ShadowProtocolEnv(seed=7)
    _, info = env.reset(seed=7)
    assert 'player_pos' not in info
    assert 'player_position' not in info
