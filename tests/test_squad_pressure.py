import math

import numpy as np
import pytest

from shadow_ai.soliton_security import SecurityGuard, line_clear
from test_interactive_tactics import game
from test_playable_actions import swat


def campaign_squad(game):
    game.env.grid[:] = 0
    game.lead_alive = False
    game.aux_guards = []
    game.cameras = []
    game.spawn_safe, game.spawn_grace = False, 0.
    game.env.player_pos = np.array([10., 10.], dtype=np.float32)
    members = []
    for index, point in enumerate(((4., 10.), (3., 10.), (2.3, 10.))):
        actor = SecurityGuard(f'S{index}', np.array(point, dtype=np.float32), 0., (point,))
        actor.has_shield = index == 0
        actor.weapon = 'PISTOL' if index == 0 else 'SMG'
        actor.ammo = 16 if index == 0 else 31
        actor.seeing_player = True
        actor.visual_confidence = 1.
        actor.last_seen, actor.last_seen_age = game.env.player_pos.copy(), 0.
        actor.flash_count = 0 if index == 0 else 1
        actor.smoke_count = actor.grenade_count = 0
        actor.vitals.stamina_capacity = 2.5
        members.append(actor)
    game.swat_units = list(zip(('SHIELD', 'ASSAULT', 'LEADER'), members))
    game.team_contact, game.team_contact_age = game.env.player_pos.copy(), 0.
    game.swat_phase = 'ACTIVE'
    return members


@pytest.mark.parametrize('weapon', ['SMG', 'RIFLE', 'ONE_HAND_SMG'])
def test_repeated_automatic_hits_suppress_shield_fire_then_recover(game, weapon):
    shield, _, _ = campaign_squad(game)
    for _ in range(3):
        game._register_squad_pressure(game.env.player_pos, shield.pos, weapon, True)
        game._step_squad_pressure()
    assert shield.suppression_timer > 0
    snapshot = next(actor for actor in game.state()['swat'] if actor['id'] == 'S0')
    assert snapshot['suppression_timer'] > 0
    assert not game._squad_weapon_ready('S0', shield)
    ammo = shield.ammo
    assert not game._enemy_shot('S0', shield, shield.vitals)
    assert shield.ammo == ammo
    for _ in range(34):
        game._step_squad_pressure()
    assert shield.suppression_timer == 0 and game._squad_weapon_ready('S0', shield)


def test_pistol_hits_do_not_apply_automatic_fire_suppression(game):
    shield, _, _ = campaign_squad(game)
    for _ in range(8):
        game._register_squad_pressure(game.env.player_pos, shield.pos, 'PISTOL', True)
    assert getattr(shield, 'suppression_timer', 0.) == 0
    assert game._squad_weapon_ready('S0', shield)
    assert game._enemy_shot('S0', shield, shield.vitals)


def test_separated_rifle_shots_misses_and_shots_from_behind_do_not_pin_shield(game):
    shield, _, _ = campaign_squad(game)
    origin = game.env.player_pos.copy()
    for _ in range(3):
        game._register_squad_pressure(origin, shield.pos, 'RIFLE', True)
        for _ in range(23):
            game._step_squad_pressure()
    game._shield_burst_hits.clear()
    for _ in range(4):
        game._register_squad_pressure(origin + [0., 4.], shield.pos + [0., 4.], 'SMG', False)
        game._register_squad_pressure(np.array([0., 10.]), shield.pos, 'SMG', True)
        game._register_squad_pressure(origin, np.array([7.,10.]), 'SMG', False)
    assert getattr(shield, 'suppression_timer', 0.) == 0


def test_burst_suppression_also_works_for_dedicated_swat(swat):
    swat.core.map.grid[:] = 0
    uid = swat.core.agent_ids[0]
    actor = swat.core.units[uid]
    assert actor.profile.has_shield
    actor.pos, actor.angle = np.array([8.,8.], dtype=np.float32), 0.
    source = np.array([13.,8.], dtype=np.float32)
    for _ in range(3):
        swat._register_squad_pressure(source, actor.pos, 'SMG', True)
    assert actor.suppression_timer > 0 and not swat._squad_weapon_ready(uid, actor)
    snapshot = next(unit for unit in swat.state()['swat'] if unit['id'] == uid)
    assert snapshot['suppression_timer'] > 0


def test_visible_escape_moves_flanker_and_throws_one_finite_flash_at_recorded_point(game):
    _, flanker, _ = campaign_squad(game)
    game.env.player_velocity[:] = (2.,0.)
    game.player_actions.angle = 0.
    assert game._step_squad_breakout('S1', flanker)
    target = game._squad_breakouts['S1']['target'].copy()
    start = flanker.pos.copy()
    # Losing sight cannot disclose the new hidden position to the throw.
    game.env.player_pos[:] = (30.,30.)
    for _, actor in game.swat_units:
        actor.seeing_player = False
    for _ in range(180):
        old = flanker.pos.copy()
        game._step_squad_pressure()
        game._step_squad_breakout('S1', flanker)
        assert line_clear(game.env.grid, old, flanker.pos)
        if game.explosives:
            break
    assert np.linalg.norm(flanker.pos - start) > .5
    assert abs(float(flanker.pos[1] - 10.)) > 1.
    assert len(game.explosives) == 1 and flanker.flash_count == 0
    bomb = game.explosives[0]
    assert bomb.kind == 'FLASH' and np.array_equal(bomb.target, target)
    assert line_clear(game.env.grid, bomb.origin, target)
    for _ in range(10):
        game._step_squad_breakout('S1', flanker)
    assert len(game.explosives) == 1


@pytest.mark.parametrize('condition', ['hidden', 'facing_away', 'approaching', 'stationary', 'no_flash', 'wall'])
def test_breakout_requires_observed_retreat_and_safe_available_flash(game, condition):
    members = campaign_squad(game)
    game.env.player_velocity[:] = (2., 0.)
    game.player_actions.angle = 0.
    if condition == 'hidden':
        for actor in members:
            actor.seeing_player = False
    elif condition == 'facing_away':
        for actor in members:
            actor.angle = math.pi
    elif condition == 'approaching':
        game.env.player_velocity[:] = (-2., 0.)
        game.player_actions.angle = math.pi
    elif condition == 'stationary':
        game.env.player_velocity[:] = 0.
    elif condition == 'no_flash':
        for actor in members:
            actor.flash_count = 0
    else:
        game.env.grid[:, 7] = 1
    assert not game._step_squad_breakout('S1', members[1])
    assert not game._squad_breakouts and not game.explosives


def test_flash_is_not_thrown_into_nearby_teammates_or_through_cover(game):
    shield, flanker, _ = campaign_squad(game)
    assert not game._safe_flash_breakout(flanker, flanker.pos, shield.pos + [1.,0.])
    game.env.grid[:, 7] = 1
    assert not game._safe_flash_breakout(flanker, np.array([4.,11.5]), np.array([10.,10.]))
