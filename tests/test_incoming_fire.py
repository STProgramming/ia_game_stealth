"""Surviving an unseen shot supplies a reaction and a fixed source estimate."""
import math
import numpy as np
import pytest

from shadow_ai.interactive_runtime import InputState
from shadow_ai.playable_equipment import Equipment
from shadow_ai.soliton_security import SecurityGuard, line_clear
from test_interactive_tactics import game
from test_playable_actions import swat


def hidden_target(game, guard_id='G1'):
    game.env.grid[:] = 0
    game.env.player_pos = np.array([4., 10.], dtype=np.float32)
    guard = SecurityGuard(guard_id, np.array([8., 10.], dtype=np.float32), 0., ((8, 10),))
    game.lead_alive = guard_id == 'G0'
    game.aux_guards = [guard] if guard_id == 'G1' else []
    game.swat_units = [('ASSAULT', guard)] if guard_id.startswith('S') else []
    if guard_id == 'G0':
        game._lead_patrol = guard
        game.env.guard_pos, game.env.guard_angle = guard.pos.copy(), guard.angle
    game.equipment = Equipment(weapon='PISTOL_SUPPRESSED', ammo=7,
        weapons={'PISTOL_SUPPRESSED'}, selected='PISTOL_SUPPRESSED')
    return guard


@pytest.mark.parametrize('guard_id', ('G0', 'G1', 'S0'))
def test_survivor_of_hidden_suppressed_hit_seeks_cover_and_reports(game, guard_id):
    guard = hidden_target(game, guard_id)
    game._player_equipment_action(InputState(fire=True, aim_x=8, aim_y=10))
    assert guard.vitals.health < 1.
    assert guard.state == 'TAKE COVER'
    assert guard._incoming_cover is not None
    assert np.linalg.norm(guard.target - np.array([4., 10.])) > 4.
    assert any(entry['text'] == 'SHOOT FIRE, SHOOT FIRE' for entry in game.radio_log.entries)
    if guard_id == 'G1':
        for _ in range(125):
            game._step_radio()
        assert len(game.swat_units) == 3
    game.env.player_pos[:] = (20., 20.)
    assert np.array_equal(guard._incoming_cover['source'], [4., 10.])


@pytest.mark.parametrize('penalty', ('stun', 'surprise'))
def test_surprise_penalty_prevents_instant_hidden_shot_reaction(game, penalty):
    guard = hidden_target(game)
    if penalty == 'stun':
        guard.vitals.hit_stun_remaining = .4
    else:
        guard.surprised_remaining = .4
    game._player_equipment_action(InputState(fire=True, aim_x=8, aim_y=10))
    assert guard.vitals.health < 1.
    assert getattr(guard, '_incoming_cover', None) is None
    assert not any(entry['text'] == 'SHOOT FIRE, SHOOT FIRE' for entry in game.radio_log.entries)


def test_visible_hit_does_not_masquerade_as_unseen_fire(game):
    guard = hidden_target(game)
    guard.angle = math.pi
    game._player_equipment_action(InputState(fire=True, aim_x=8, aim_y=10))
    assert getattr(guard, '_incoming_cover', None) is None


def test_cover_reaction_moves_away_and_blocks_firing(game):
    guard = hidden_target(game)
    game._player_equipment_action(InputState(fire=True, aim_x=8, aim_y=10))
    start = guard.pos.copy()
    ammo = guard.ammo
    for _ in range(30):
        game._step_guard(guard)
        game._step_guard_weapon(guard, True)
    assert guard.pos[0] > start[0]
    assert guard.ammo == ammo


@pytest.mark.parametrize('kind,field', [('GRENADE', 'grenade_count'), ('FLASH', 'flash_count'), ('SMOKE', 'smoke_count')])
def test_swat_throws_available_equipment_at_recorded_shot_origin(game, kind, field):
    guard = hidden_target(game, 'S0')
    guard.grenade_count = guard.flash_count = guard.smoke_count = 0
    setattr(guard, field, 1)
    game._player_equipment_action(InputState(fire=True, aim_x=8, aim_y=10))
    guard.pos = guard._incoming_cover['goal'].copy()
    game.env.player_pos[:] = (20., 20.)
    game._step_incoming_cover('S0', guard, guard.vitals)
    assert getattr(guard, field) == 0
    assert len(game.explosives) == 1
    bomb = game.explosives[0]
    assert bomb.kind == kind and bomb.owner == 'S0'
    assert np.allclose(bomb.target, [4., 10.], atol=.15)
    for _ in range(46):
        game._step_explosives()
    assert not game.explosives
    assert bool(game.smoke_clouds) == (kind == 'SMOKE')
    assert bool(game.flash_events) == (kind == 'FLASH')
    assert bool(game.blasts) == (kind == 'GRENADE')


def test_cover_goal_prefers_a_wall_over_open_retreat(game):
    game.env.grid[:] = 0
    source = np.array([4., 10.], dtype=np.float32)
    start = np.array([8., 10.], dtype=np.float32)
    game.env.grid[9, 9] = 1
    goal = game._incoming_cover_goal(start, source)
    assert not line_clear(game.env.grid, source, goal)
    assert np.isfinite(game.navigator.path_length(start, goal))


@pytest.mark.parametrize('kind,field', [('GRENADE', 'grenade_count'), ('FLASH', 'flash_count'), ('SMOKE', 'smoke_count')])
def test_swat_peeks_from_wall_cover_before_throwing_without_crossing_walls(game, kind, field):
    guard = hidden_target(game, 'S0')
    game.env.grid[9, 9] = 1
    guard.grenade_count = guard.flash_count = guard.smoke_count = 0
    setattr(guard, field, 1)
    source = game.env.player_pos.copy()
    game._player_equipment_action(InputState(fire=True, aim_x=8, aim_y=10))
    cover = guard._incoming_cover['goal'].copy()
    assert not line_clear(game.env.grid, source, cover)
    # Cover blocks a direct throw. The unit must physically reach a clear edge.
    guard.pos = cover.copy()
    game.env.player_pos[:] = (20., 20.)
    for _ in range(250):
        old = guard.pos.copy()
        game._step_incoming_cover('S0', guard, guard.vitals)
        assert line_clear(game.env.grid, old, guard.pos)
        if game.explosives:
            break
    assert len(game.explosives) == 1
    bomb = game.explosives[0]
    assert bomb.kind == kind
    assert np.linalg.norm(bomb.origin - cover) > .3
    assert line_clear(game.env.grid, bomb.origin, bomb.target)
    assert np.allclose(bomb.target, source)
    assert getattr(guard, field) == 0


def test_incoming_cover_cancels_previous_medical_assignment(game):
    guard = hidden_target(game)
    guard.state = 'WAIT MEDIC'
    game._medical_jobs[guard.guard_id] = dict(helper='G0', point=guard.pos.copy(), started=False)
    game._player_equipment_action(InputState(fire=True, aim_x=8, aim_y=10))
    assert guard._incoming_cover is not None
    assert guard.guard_id not in game._medical_jobs


def test_standalone_swat_reacts_to_hidden_hit_and_uses_flash(swat):
    swat.core.map.grid[:] = 0
    unit = next(u for u in swat.core.units.values() if not u.profile.has_shield)
    for other in swat.core.units.values():
        other.alive = other is unit
    unit.pos, unit.angle = np.array([8., 10.], dtype=np.float32), 0.
    unit.grenade_count = unit.smoke_count = 0
    unit.flash_count = 1
    swat.core.player_pos = np.array([4., 10.], dtype=np.float32)
    swat.equipment = Equipment(weapon='PISTOL_SUPPRESSED', ammo=7,
        weapons={'PISTOL_SUPPRESSED'}, selected='PISTOL_SUPPRESSED')
    swat._swat_player_equipment(InputState(fire=True, aim_x=8, aim_y=10))
    assert unit.health < 1.
    assert unit._incoming_cover is not None
    assert any(entry['text'] == 'SHOOT FIRE, SHOOT FIRE' for entry in swat.radio_log.entries)
    source = swat.core.player_pos.copy()
    unit.pos = unit._incoming_cover['goal'].copy()
    swat.core.player_pos[:] = (20., 20.)
    swat._step_incoming_cover(unit.uid, unit, swat._unit_vitals[unit.uid])
    assert len(swat.explosives) == 1
    assert swat.explosives[0].kind == 'FLASH'
    assert np.allclose(swat.explosives[0].target, source, atol=.15)
