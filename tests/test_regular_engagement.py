import math
from unittest.mock import patch

import numpy as np
import pytest

from shadow_ai import TacticalAction
from shadow_ai.soliton_security import SecurityGuard, can_see
from shadow_ai.playable_combat import lane_blocked
from test_interactive_tactics import game
from test_playable_actions import swat


def setup_contact(game):
    game.env.grid[:] = 0
    game.lead_alive = False
    game.aux_guards = []
    game.cameras = []
    game.spawn_safe, game.spawn_grace = False, 0.
    game.env.player_pos = np.array([11., 10.], dtype=np.float32)
    game.equipment.weapon = 'PISTOL'
    game.equipment.weapons.add('PISTOL')
    regular = SecurityGuard('G7', np.array([5., 10.], dtype=np.float32), 0., ((5, 10),))
    regular.visual_confidence = 1.
    regular.weapon, regular.ammo = 'RIFLE', 31
    regular.state = 'PROTECT INTEL'
    regular.target = np.array([5., 2.], dtype=np.float32)
    shield = SecurityGuard('S0', np.array([7., 10.], dtype=np.float32), 0., ((7, 10),))
    shield.has_shield = True
    game.aux_guards = [regular]
    game.swat_units = [('SHIELD', shield)]
    game.swat_phase = 'ACTIVE'
    return regular, shield


def test_sentry_engages_visible_player_beside_swat_instead_of_optional_post(game):
    regular, shield = setup_contact(game)
    start = regular.pos.copy()
    with patch.object(game, '_step_context_tactics', return_value=(True, True)) as context:
        for _ in range(55):
            game._step_guard(regular, TacticalAction.PROTECT_NEAREST_INTEL)
    assert not context.called
    assert np.linalg.norm(regular.pos - start) > .5
    assert regular.seeing_player
    assert not lane_blocked(regular.pos, game.env.player_pos, [shield.pos])
    assert regular.pos[0] < game.env.player_pos[0]
    before = regular.ammo
    for _ in range(40):
        game._step_guard_weapon(regular, True)
    assert regular.ammo < before


def test_sentry_lost_visual_returns_to_security_not_hidden_chase(game):
    regular, _ = setup_contact(game)
    game._step_guard(regular, TacticalAction.HOLD)
    known = regular.last_seen.copy()
    game.env.player_pos = np.array([30., 25.], dtype=np.float32)
    game._step_guard(regular, TacticalAction.PURSUE)
    assert not regular.seeing_player
    assert regular.state in {'PROTECT INTEL', 'SECURE EXFIL', 'WATCH MEDKIT', 'PATROL'}
    assert regular.guard_id not in game._searches
    assert not np.allclose(regular.target, game.env.player_pos)
    assert np.array_equal(regular.last_seen, known)


def test_swat_lost_visual_searches_fixed_last_contact(game):
    _, shield = setup_contact(game)
    shield.state = 'CHASE'
    shield.visual_confidence = 1.
    shield.last_seen = np.array([11., 10.], dtype=np.float32)
    shield.last_seen_age = 0.
    game.env.player_pos = np.array([30., 25.], dtype=np.float32)
    game.team_contact, game.team_contact_age = shield.last_seen.copy(), 2.
    game._step_guard(shield, TacticalAction.PURSUE)
    assert shield.state == 'SEARCH'
    assert shield.guard_id in game._searches
    assert not np.array_equal(shield.target, game.env.player_pos)


@pytest.mark.parametrize('state', ['SEARCH', 'FLANK', 'RESPOND', 'SUPPORT', 'TRACK BLOOD'])
def test_hidden_pursuit_orders_cannot_pull_sentries_from_security_posts(game, state):
    regular, _ = setup_contact(game)
    game._publish_contact(game.env.player_pos.copy(), 'S0')
    game.env.player_pos = np.array([30., 25.], dtype=np.float32)
    regular.state = state
    regular.last_seen = np.array([11., 10.], dtype=np.float32)
    regular.last_seen_age = 1.
    regular.target = regular.last_seen.copy()
    regular.blood_target = regular.last_seen.copy() if state == 'TRACK BLOOD' else None
    for _ in range(5):
        game._step_guard(regular, TacticalAction.PURSUE)
    assert regular.state in {'PROTECT INTEL', 'SECURE EXFIL', 'WATCH MEDKIT', 'PATROL'}
    assert regular.guard_id not in game._searches
    assert not np.array_equal(regular.target, game.env.player_pos)


@pytest.mark.parametrize('mode', ['campaign', 'swat'])
def test_announced_flash_blocks_allied_advance_until_burst(game, swat, mode):
    runtime = game if mode == 'campaign' else swat
    world, grid = runtime._combat_world()
    grid[:] = 0
    target = np.array([10.5, 10.], dtype=np.float32)
    if mode == 'campaign':
        regular, _ = setup_contact(game)
        game.swat_units.clear()
        regular.pos = np.array([6.25, 10.], dtype=np.float32)
        regular.state = 'CHASE'
        actor = regular
        def move():
            actor.pos, actor.angle = runtime._guard_move(actor, target, 3.)
    else:
        actor = next(iter(world.units.values()))
        actor.pos = np.array([6.25, 10.], dtype=np.float32)
        actor.angle = 0.
        def move():
            runtime._unit_vitals_stepped.clear()
            runtime._move_swat_unit(actor, target)
    runtime._throw_grenade(np.array([4., 11.5]), target, 'S1', kind='FLASH')
    for _ in range(46):
        move()
        runtime._step_explosives()
    assert actor.blinded_timer == 0
    assert np.linalg.norm(actor.pos - target) >= 3.
    assert not runtime.explosives
    start = actor.pos.copy()
    for _ in range(15):
        move()
    assert np.linalg.norm(actor.pos - target) < np.linalg.norm(start - target)


def test_flash_clearance_allows_escape_and_does_not_protect_from_player_flash(game):
    regular, _ = setup_contact(game)
    target = np.array([10., 10.], dtype=np.float32)
    game._throw_grenade(np.array([4., 10.]), target, 'S1', kind='FLASH')
    old = target + [1., 0.]
    outward = old + [1., 0.]
    assert np.array_equal(game._friendly_flash_move(regular, old, outward), outward)
    game.explosives[0].owner = 'PLAYER'
    assert np.array_equal(game._friendly_flash_move(regular, outward, old), old)


def test_sentries_keep_patrolling_separate_posts_after_swat_withdrawal(game):
    from test_security_distribution import stage
    guards = stage(game, count=7)
    game._publish_contact(np.array([20., 10.], dtype=np.float32), 'S0')
    game.swat_phase = 'WITHDRAWN'
    game.env.player_pos = np.array([60., 38.], dtype=np.float32)
    for guard in guards:
        game._step_guard(guard, TacticalAction.HOLD)
    before = [guard.target.copy() for guard in guards]
    game.env.steps += int(8.1 / game.game_dt)
    for guard in guards:
        game._step_guard(guard, TacticalAction.HOLD)
    assert all(guard.state == 'PROTECT INTEL' for guard in guards)
    assert all(np.linalg.norm(guard.target - old) > .5 for guard, old in zip(guards, before))
    assert all(np.linalg.norm(first.target - other.target) >= 1.1
               for index, first in enumerate(guards) for other in guards[index + 1:])


def test_sentry_faces_guarded_intel_and_can_report_theft_from_its_perimeter(game):
    from test_security_distribution import stage
    guard = stage(game, count=1)[0]
    game.level = 1
    game.env.episode_has_intel[:] = [1, 0, 0]
    game._publish_contact(np.array([20., 10.], dtype=np.float32), 'S0')
    game.env.player_pos = np.array([60., 38.], dtype=np.float32)
    game._guard_goal_from_radio(guard)
    guard.pos, guard.angle = guard.target.copy(), 0.
    for _ in range(90):
        game._step_guard(guard, TacticalAction.HOLD)
        game.env.steps += 1
    assert can_see(game.env.grid, guard.pos, guard.angle, game.env._intel[0], 11.5, 115.)
    game.env.player_has_intel[0] = 1
    game._step_guard(guard, TacticalAction.HOLD)
    assert 0 in game._intel_checks


def test_medical_perimeter_guard_reaches_its_slot_instead_of_stopping_near_centre(game):
    from test_security_distribution import stage
    from shadow_ai.playable_threats import MedicalSurveillance
    guard = stage(game, count=1)[0]
    game.env.known_intel_present[:] = 0
    station = dict(id='guarded-kit', kind='MEDKIT', fixed=True,
                   pos=np.array([8., 4.], dtype=np.float32))
    game.pickups = [station]
    game.medical_watch = MedicalSurveillance(game.pickups)
    game.medical_watch.hit(guard.guard_id)
    guard.pos = np.array([5., 4.], dtype=np.float32)
    post = game._security_assignment(guard)
    assert post['state'] == 'WATCH MEDKIT'
    guard.pos = post['goal'] - np.array([1.5, 0.], dtype=np.float32)
    guard.target, guard.state, guard.angle = post['goal'].copy(), post['state'], 0.
    game.env.player_pos = np.array([60., 38.], dtype=np.float32)
    start = guard.pos.copy()
    game._step_guard(guard, TacticalAction.HOLD)
    assert np.array_equal(guard.target, post['goal'])
    assert np.linalg.norm(guard.pos - start) > 0
