import math
from unittest.mock import patch

import numpy as np
import pytest

from shadow_ai.interactive_runtime import InputState
from shadow_ai.playable_navigation import HumanMotion
from shadow_ai.soliton_security import SecurityGuard
from test_interactive_tactics import game
from test_playable_actions import swat


@pytest.mark.parametrize("guard_id", ("G0", "G1", "S0"))
@pytest.mark.parametrize("alert", ("SUSPICIOUS", "CONTACT", "SEARCH"))
def test_alerted_guards_run_and_spend_energy_even_while_searching(game, guard_id, alert):
    game.env.grid[:] = 0
    guard = SecurityGuard(guard_id, np.array([4., 4.], dtype=np.float32), 0, ((4, 4),))
    guard.state = "SEARCH"
    goal = np.array([30., 4.], dtype=np.float32)
    patrol_speed = float(game.env.cfg.guard_speed) / game.game_dt
    game.facility_alert = alert
    for _ in range(60):
        guard.pos, guard.angle = game._guard_move(guard, goal, patrol_speed * .82)
    assert guard.pos[0] - 4 > patrol_speed * 2
    assert guard.vitals.stamina == pytest.approx(1 - .14 * 2)
    assert game._motions[guard_id].speed > patrol_speed * 2

    guard.vitals.stamina = 0
    guard.vitals.exhausted = True
    start = guard.pos.copy()
    guard.pos, guard.angle = game._guard_move(guard, goal, patrol_speed * 2.4)
    assert np.linalg.norm(guard.pos - start) <= patrol_speed * .7 * game.game_dt + 1e-6


@pytest.mark.parametrize("body_first", (True, False))
def test_gunfire_supersedes_pending_body_reinforcements(game, body_first):
    game.env.grid[:] = 0
    game.env.player_pos = np.array([4., 4.], dtype=np.float32)
    victim = game.aux_guards[0]
    game._down_guard(victim.guard_id, lethal=True)
    if body_first:
        game._raise_evidence_alert("BODY FOUND", victim.pos, "G0")
        assert game._guard_dispatch is not None
    game.equipment.weapon, game.equipment.ammo = "PISTOL", 1
    game._player_equipment_action(InputState(fire=True, aim_x=4, aim_y=12))
    # Another observer can discover the same body after the shot.
    game._raise_evidence_alert("BODY FOUND", victim.pos, "G0")
    for _ in range(125):
        game._step_guard_dispatch()
        game._step_radio()
    assert game._guard_dispatch is None
    assert game._regular_reinforcements == 0
    assert len(game.swat_units) == 3


@pytest.mark.parametrize("stamina, fired", ((1., True), (.08, True), (.079, False), (0., False)))
def test_guard_shots_require_and_consume_energy(game, stamina, fired):
    game.env.grid[:] = 0
    game.lead_alive = False
    guard = SecurityGuard("G1", np.array([4., 4.], dtype=np.float32), 0, ((4, 4),))
    game.aux_guards = [guard]
    game.env.player_pos = np.array([8., 4.], dtype=np.float32)
    guard.vitals.stamina = stamina
    guard.aim_timer = 2.
    game._step_guard_weapon(guard, True)
    assert guard.ammo == (15 if fired else 16)
    assert guard.vitals.stamina == pytest.approx(stamina - (.08 if fired else 0))
    assert game.player_vitals.health == pytest.approx(.88 if fired else 1.)


@pytest.mark.parametrize("guard_id", ("G0", "G1", "S0"))
def test_frontal_knife_hit_prevents_immediate_shot_and_capture_then_allows_response(game, guard_id):
    game.env.grid[:] = 0
    game.env.player_pos = np.array([4., 4.], dtype=np.float32)
    guard = game._lead_patrol if guard_id == "G0" else SecurityGuard(
        guard_id, np.array([4.5, 4.], dtype=np.float32), math.pi, ((4.5, 4),))
    guard.pos = np.array([4.5, 4.], dtype=np.float32)
    guard.angle = math.pi
    game.env.guard_pos, game.env.guard_angle = guard.pos.copy(), guard.angle
    game.lead_alive = guard_id == "G0"
    game.aux_guards = [guard] if guard_id == "G1" else []
    game.swat_units = [("ASSAULT", guard)] if guard_id == "S0" else []
    guard.aim_timer = 2.
    game.tick(InputState(lethal=True))
    assert guard.vitals.health == pytest.approx(.75)
    assert guard.vitals.hit_stun_remaining > 0
    assert game.status == "RUNNING"
    assert guard.ammo == 16 and game.player_vitals.health == 1
    position = guard.pos.copy()
    for _ in range(6):
        game.tick(InputState())
        assert np.array_equal(guard.pos, position)
        assert game.status == "RUNNING"
        assert guard.ammo == 16 and game.player_vitals.health == 1
    # After the brief reaction, a healthy guard with energy can attack again.
    game.env.player_pos = np.array([8., 4.], dtype=np.float32)
    guard.angle = 0
    game.env.guard_angle = 0
    for _ in range(45):
        game.tick(InputState())
        if guard.ammo < 16:
            break
    assert guard.vitals.hit_stun_remaining == 0
    assert guard.ammo < 16


def test_standalone_swat_knife_stuns_and_shooting_consumes_energy(swat):
    swat.core.map.grid[:] = 0
    swat.core.player_pos = np.array([4., 4.], dtype=np.float32)
    unit = next(iter(swat.core.units.values()))
    for other in swat.core.units.values():
        other.alive = other is unit
    unit.pos = np.array([4.5, 4.], dtype=np.float32)
    unit.angle = math.pi
    swat._swat_melee(InputState(lethal=True))
    assert unit.health == pytest.approx(.75)
    assert swat._unit_vitals[unit.uid].hit_stun_remaining == .3
    ammo = unit.ammo
    swat._world_after_ai()
    assert unit.ammo == ammo and swat.core.player_health == 1
    assert not swat.core.player_captured
    swat._unit_vitals[unit.uid].step(False, .3)
    swat.core.player_pos = np.array([7., 4.], dtype=np.float32)
    unit.angle = 0
    swat._world_after_ai()
    assert unit.ammo == ammo - 1
    assert swat._unit_vitals[unit.uid].stamina == pytest.approx(1 - .08 / 2.5)


def test_swat_keeps_running_pace_when_changing_tactical_intent(game):
    game.env.grid[:] = 0
    game.facility_alert = "CONTACT"
    guard = SecurityGuard("S0", np.array([4., 4.], dtype=np.float32), 0, ((4, 4),))
    goal = np.array([30., 4.], dtype=np.float32)
    patrol_speed = float(game.env.cfg.guard_speed) / game.game_dt
    with patch.object(HumanMotion, 'move', return_value=(guard.pos.copy(), 0)) as move:
        for state, multiplier in (("RESPOND", 2.7), ("SEARCH", 1.4), ("STACK", 1.65), ("CLEAR ROOM", 1.65)):
            guard.state = state
            game._guard_move(guard, goal, patrol_speed * multiplier)
            assert move.call_args.args[5] == pytest.approx(patrol_speed * 2.8)
    assert guard.vitals.stamina == 1
    assert guard.vitals.fatigue == 0


def test_swat_short_corner_checks_do_not_accumulate_fatigue(game):
    game.env.grid[:] = 0
    game.facility_alert = "SEARCH"
    guard = SecurityGuard("S0", np.array([4., 4.], dtype=np.float32), 0, ((4, 4),))
    guard.state = "SEARCH"
    goal = np.array([35., 4.], dtype=np.float32)
    for _ in range(4):
        for _ in range(20):
            guard.pos, guard.angle = game._guard_move(guard, goal, 2.)
        guard.pos, guard.angle = game._guard_move(guard, None, 2.)
    assert .5 < guard.vitals.stamina < 1
    assert guard.vitals.fatigue == 0


def test_standalone_swat_movement_penalties_are_applied_once(swat):
    from shadow_ai.swat import SwatAction
    swat.core.map.grid[:] = 0
    unit = next(iter(swat.core.units.values()))
    for other in swat.core.units.values():
        other.alive = other is unit
    unit.pos = np.array([10., 10.], dtype=np.float32)
    swat.core.player_pos = np.array([4., 4.], dtype=np.float32)
    swat._unit_vitals[unit.uid].fatigue = .2
    swat._step_swat_search = lambda uid: False
    target = np.array([20., 10.], dtype=np.float32)

    def apply(uid, action):
        swat._move_swat_unit(unit, target, .65)

    with patch.object(swat.core, 'apply_action', side_effect=apply), patch.object(
            HumanMotion, 'move', side_effect=lambda nav, key, pos, angle, goal, speed, dt, turn: (pos + np.array([speed * dt, 0]), angle)):
        swat.cached_actions[unit.uid] = SwatAction.ADVANCE
        swat.policy_tick = 1
        before = unit.pos.copy()
        swat.tick(InputState())
    expected = .95 * unit.profile.move_multiplier * .8 * swat.game_dt
    assert unit.pos[0] - before[0] == pytest.approx(expected, abs=1e-6)
    assert swat._unit_vitals[unit.uid].stamina == pytest.approx(1 - .14 * swat.game_dt / 2.5)


@pytest.mark.parametrize('guard_id', ('G0', 'G1', 'S0'))
def test_missing_intel_requires_inspection_radio_and_negative_reply_before_caution(game, guard_id):
    game.env.grid[:] = 0
    index = game.env.episode_intel_order[0]
    point = game.env._intel[index]
    game.env.player_has_intel[index] = 1
    game.env.player_pos = np.array([2., 27.], dtype=np.float32)
    guard = game._lead_patrol if guard_id == 'G0' else SecurityGuard(
        guard_id, point + np.array([-1., 0.], dtype=np.float32), 0, ((4, 4),))
    guard.pos = point + np.array([-1., 0.], dtype=np.float32)
    guard.angle = 0
    game.env.guard_pos = guard.pos.copy() if guard_id == 'G0' else np.array([22., 12.], dtype=np.float32)
    if guard_id == 'G1':
        game.aux_guards[0] = guard
    elif guard_id == 'S0':
        game.swat_units = [('ASSAULT', guard)]
    game._inspect_guard_intel(guard)
    assert index in game._intel_checks
    assert 'Qualcuno' in game.guard_dialogue[guard_id]
    assert game.facility_alert == 'CALM' and not game.security_compromised
    assert game.env.known_intel_present[index] == 1
    for _ in range(35):
        game._step_intel_radio()
    assert game.facility_alert == 'CALM'
    game._step_intel_radio()
    assert game.facility_alert == 'SUSPICIOUS'
    assert game.env.known_intel_present[index] == 0
    assert any('Negativo' in line for line in game.guard_dialogue.values())
    assert all(g.suspicion >= .48 for g in game._active_guards())
    assert game.team_contact is None and game.radio_timer is None
    assert game._guard_dispatch is None
    updates = game.radio_updates
    game._inspect_guard_intel(guard)
    game._step_intel_radio()
    assert game.radio_updates == updates


@pytest.mark.parametrize('obstruction', ('far', 'wall', 'smoke', 'blind', 'down', 'absent_from_mission'))
def test_missing_intel_is_not_detected_without_an_inspection(game, obstruction):
    game.env.grid[:] = 0
    index = game.env.episode_intel_order[0]
    point = game.env._intel[index]
    guard = game.aux_guards[0]
    guard.pos, guard.angle = point + np.array([-1., 0.], dtype=np.float32), 0
    game.env.player_has_intel[index] = 1
    if obstruction == 'far':
        guard.pos[0] -= 2
    elif obstruction == 'wall':
        game.env.grid[game.env._cell(point)[1], game.env._cell(point)[0]] = 1
    elif obstruction == 'smoke':
        game.smoke_clouds = [dict(pos=point.copy(), radius=2., ttl=8.)]
    elif obstruction == 'blind':
        guard.blinded_timer = 2
    elif obstruction == 'down':
        guard.unconscious = True
    else:
        game.env.episode_has_intel[index] = 0
    game._inspect_guard_intel(guard)
    assert not game._intel_checks
    assert not game.security_compromised


def test_primary_does_not_bypass_intel_radio_confirmation(game):
    game.env.grid[:] = 0
    index = game.env.episode_intel_order[0]
    point = game.env._intel[index]
    game.env.player_has_intel[index] = 1
    game.env.guard_pos = point.copy()
    game.env.guard_angle = 0
    game.env.player_pos = np.array([2., 27.], dtype=np.float32)
    game.tick(InputState())
    assert index in game._intel_checks
    assert game.env.known_intel_present[index] == 1
    assert game.facility_alert == 'CALM'
    assert game.env.alert == 0


def test_losing_intel_inspector_cancels_call_and_reset_clears_reports(game):
    game.env.grid[:] = 0
    index = game.env.episode_intel_order[0]
    point = game.env._intel[index]
    game.env.player_has_intel[index] = 1
    guard = game.aux_guards[0]
    guard.pos, guard.angle = point.copy(), 0
    game._inspect_guard_intel(guard)
    guard.unconscious = True
    game._step_intel_radio()
    assert not game._intel_checks and not game.security_compromised
    game._reported_missing_intel.add(index)
    game.reset('routine')
    assert not game._intel_checks and not game._reported_missing_intel


def test_intel_confirmation_preserves_an_existing_combat_alert(game):
    game.env.grid[:] = 0
    game._player_security_event('GUNFIRE', np.array([4., 4.], dtype=np.float32))
    index = game.env.episode_intel_order[0]
    guard = game.aux_guards[0]
    guard.pos, guard.angle = game.env._intel[index].copy(), 0
    game.env.player_has_intel[index] = 1
    game._inspect_guard_intel(guard)
    for _ in range(36):
        game._step_intel_radio()
    assert game.facility_alert == 'CONTACT'
    assert game.threat_level == 2 and game.env.alert == 3
    assert game.team_contact_source == 'GUNFIRE'


def test_defence_does_not_know_an_intel_is_missing_before_confirmation(game):
    game.env.episode_has_intel[:] = 1
    game.env.player_has_intel[:] = 1
    game.env.known_intel_present[:] = 1
    game.team_contact = np.array([30., 12.], dtype=np.float32)
    guard = game.aux_guards[0]
    game._guard_goal_from_radio(guard)
    assert guard.state == 'PROTECT INTEL'
    assert np.array_equal(guard.target, game.env._intel[0])
