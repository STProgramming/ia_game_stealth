import math
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pytest

from shadow_ai.interactive_runtime import InputState
from shadow_ai.playable_actions import spend_energy
from shadow_ai.playable_combat import EnemyMelee
from shadow_ai.playable_combat import MELEE_PROFILES
from shadow_ai.playable_equipment import Equipment
from shadow_ai.playable_vitals import PlayableVitals, SWAT_STAMINA_CAPACITY
from shadow_ai.soliton_security import SecurityGuard, line_clear
from test_interactive_tactics import game
from test_playable_actions import swat


def drop_only_rng():
    from itertools import cycle
    rolls = cycle([.99, .99, 0.])
    return SimpleNamespace(random=lambda: next(rolls), normal=lambda *args, **kwargs: np.zeros(kwargs['size']) if 'size' in kwargs else 0.)


def test_deployed_and_standalone_swat_have_two_and_half_times_the_endurance(game, swat):
    game._publish_contact(np.array([20., 12.], dtype=np.float32), 'C1')
    for _ in range(120):
        game._step_radio()
    units = [unit.vitals for _, unit in game.swat_units] + list(swat._unit_vitals.values())
    assert all(v.stamina_capacity == SWAT_STAMINA_CAPACITY == 2.5 for v in units)
    normal = PlayableVitals()
    for vitals in [normal] + units:
        for _ in range(270):
            vitals.step(True, 1 / 30, drain=.14, fatigue_on_pause=False)
    assert normal.exhausted
    assert all(v.can_sprint and v.stamina > .49 for v in units)
    before = units[0].stamina
    assert spend_energy(units[0], .08)
    assert before - units[0].stamina == pytest.approx(.08 / 2.5)


@pytest.fixture(params=['G0', 'G1', 'S0'])
def enemy(game, request):
    game.env.grid[:] = 0
    game.env.player_pos = np.array([4., 4.], dtype=np.float32)
    guard = SecurityGuard(request.param, np.array([4.5, 4.], dtype=np.float32), math.pi, ((4.5, 4),))
    game.lead_alive = request.param == 'G0'
    game.aux_guards = [guard] if request.param == 'G1' else []
    game.swat_units = [('ASSAULT', guard)] if request.param == 'S0' else []
    if request.param == 'G0':
        game._lead_patrol = guard
        game.env.guard_pos, game.env.guard_angle = guard.pos.copy(), guard.angle
    return guard


def test_every_guard_attempts_melee_then_knocks_down_armed_player_without_instant_capture(game, enemy):
    game.equipment.weapon, game.equipment.ammo = 'PISTOL', 3
    game.tick(InputState())
    assert game._melee_attacks[enemy.guard_id].windup > 0
    assert not game.player_actions.prone and game.status == 'RUNNING'
    for _ in range(30):
        game.tick(InputState())
        if game.player_actions.prone:
            break
    assert game.player_actions.knockdown_remaining > 0
    assert game.player_vitals.health == pytest.approx(.78 if enemy.guard_id.startswith('S') else .88)
    assert enemy.ammo == 16 and not game.shots
    assert game.status == 'RUNNING'
    landing = game.env.player_pos.copy()
    for _ in range(20):
        game._apply_player(InputState(right=True, fire=True))
        assert np.array_equal(game.env.player_pos, landing)
    assert game.equipment.ammo == 3
    for _ in range(30):
        game._apply_player(InputState(fire=True))
        if game.player_actions.movement_locked:
            assert game.equipment.ammo == 3
        else:
            break
    assert not game.player_actions.prone
    assert game.player_actions.rise_timer == 0


def test_knife_hit_interrupts_an_enemy_melee_windup(game, enemy):
    game.tick(InputState())
    assert game._melee_attacks[enemy.guard_id].windup > 0
    game.tick(InputState(lethal=True))
    assert enemy.vitals.hit_stun_remaining > 0
    assert game._melee_attacks[enemy.guard_id].windup == 0
    assert not game.player_actions.prone and game.player_vitals.health == 1


@pytest.mark.parametrize('blocked', ['wall', 'exhausted', 'stunned', 'blind', 'down', 'unarmed', 'out_of_range'])
def test_melee_requires_a_reachable_armed_player_and_an_able_guard(game, blocked):
    game.env.grid[:] = 0
    game.lead_alive = False
    guard = SecurityGuard('G1', np.array([3.4, 4.], dtype=np.float32), 0, ((3, 4),))
    game.aux_guards = [guard]
    game.env.player_pos = np.array([4.6, 4.], dtype=np.float32)
    guard.ammo = 0  # Melee does not require bullets.
    if blocked == 'wall':
        game.env.grid[:, 4] = 1
    elif blocked == 'exhausted':
        guard.vitals.stamina = 0
        guard.vitals.exhausted = True
    elif blocked == 'stunned':
        guard.vitals.hit_stun_remaining = 2
    elif blocked == 'blind':
        guard.blinded_timer = 2
    elif blocked == 'down':
        guard.unconscious = True
    elif blocked == 'unarmed':
        game.equipment.knife = False
    else:
        game.env.player_pos[0] = 7
    for _ in range(20):
        game._step_guard_weapon(guard, True)
    assert not game.player_actions.prone
    assert game.player_vitals.health == 1


def test_player_can_dodge_a_melee_attack_and_knockback_respects_walls(game):
    game.env.grid[:] = 0
    game.lead_alive = False
    guard = SecurityGuard('G1', np.array([4., 4.], dtype=np.float32), 0, ((4, 4),))
    game.aux_guards = [guard]
    game.env.player_pos = np.array([5., 4.], dtype=np.float32)
    guard.ammo = 0
    game._step_guard_weapon(guard, True)
    game.env.player_pos[0] = 7
    for _ in range(12):
        game._step_guard_weapon(guard, True)
    assert not game.player_actions.prone
    assert game._melee_attacks['G1'].windup == 0
    game.env.player_pos[0] = 5
    game.env.grid[:, 6] = 1
    before = game.env.player_pos.copy()
    for _ in range(12):
        game._step_guard_weapon(guard, True)
    assert game.player_actions.prone
    assert game.env.player_pos[0] < 5.5
    assert line_clear(game.env.grid, before, game.env.player_pos)
    assert guard.vitals.stamina == pytest.approx(.2)  # Both the dodged attempt and the hit cost energy.


def test_standalone_swat_uses_the_same_melee_and_get_up_rules(swat):
    swat.core.map.grid[:] = 0
    unit = next(iter(swat.core.units.values()))
    for other in swat.core.units.values():
        other.alive = other is unit
    unit.pos, unit.angle = np.array([4.5, 4.], dtype=np.float32), math.pi
    swat.core.player_pos = np.array([4., 4.], dtype=np.float32)
    ammo = unit.ammo
    for _ in range(30):
        swat._world_after_ai()
        if swat.player_actions.prone:
            break
    assert swat.player_actions.knockdown_remaining > 0
    assert swat.core.player_health == swat.player_vitals.health == pytest.approx(.68)
    assert not swat.core.player_captured
    assert unit.ammo == ammo
    assert swat._unit_vitals[unit.uid].stamina == pytest.approx(1 - .65 / 2.5)
    for _ in range(50):
        swat._apply_player(InputState())
    assert not swat.player_actions.prone


@pytest.mark.parametrize('level', [1, 2])
def test_next_level_loses_pistol_and_ammo_and_keeps_every_other_item(game, level):
    from game_server import GameSession
    game.level = level
    game.reset('routine')
    game.status = 'ESCAPED'
    game.equipment = Equipment(weapon='PISTOL', ammo=17, knife=True, flash=2, smoke=3,
                               bandages=4, medkits=2, casings=5, magazines=6,
                               hazmat=True, sample_vial=True, security_card=True,
                               weapons={'PISTOL', 'RIFLE'}, selected='PISTOL')
    original = game.equipment.snapshot()
    session = GameSession.__new__(GameSession)
    session.runtime, session.mode, session.revision = game, 'guard', 0
    with patch.object(session, '_start_episode_learning') as start:
        session.handle({'type': 'next_level'})
        start.assert_called_once()
    current = game.equipment.snapshot()
    assert game.level == level + 1 and game.status == 'RUNNING'
    assert current['weapon'] is None and current['ammo'] == 0
    assert current['weapons'] == ['RIFLE'] and current['selected'] == 'KNIFE'
    for key in original.keys() - {'weapon', 'ammo', 'weapons', 'selected'}:
        assert current[key] == original[key], key


def test_failed_or_final_level_does_not_drop_inventory(game):
    game.equipment = Equipment(weapon='PISTOL', ammo=7, weapons={'PISTOL'}, knife=True)
    before = game.equipment.snapshot()
    assert not game.advance_level()  # Current mission is still running.
    game.status = 'ESCAPED'
    game.level = 3
    assert not game.advance_level()
    assert game.equipment.snapshot() == before


@pytest.mark.parametrize('capacity, stamina, succeeds', [(1., .39, False), (1., .4, True), (2.5, .15, False), (2.5, .16, True)])
def test_melee_pays_heavy_energy_cost_when_the_attempt_starts(capacity, stamina, succeeds):
    vitals = PlayableVitals(stamina=stamina, stamina_capacity=capacity)
    attack = EnemyMelee()
    assert not attack.step(True, vitals, 1 / 30)
    assert vitals.stamina == pytest.approx(stamina - (.4 / capacity if succeeds else 0))
    hit = False
    for _ in range(10):
        hit |= attack.step(True, vitals, 1 / 30)
    assert hit == succeeds


def test_an_interrupted_melee_still_costs_vigour():
    vitals = PlayableVitals()
    attack = EnemyMelee()
    attack.step(True, vitals, 1 / 30)
    assert vitals.stamina == pytest.approx(.6)
    attack.step(False, vitals, 1 / 30)
    assert attack.windup == 0 and vitals.stamina == pytest.approx(.6)


@pytest.mark.parametrize('mode', ['game', 'swat'])
@pytest.mark.parametrize('kind', ['KNIFE', 'PISTOL', 'RIFLE', 'SMOKE', 'MEDKIT', 'SECURITY_CARD', 'HAZMAT', 'SAMPLE_VIAL'])
def test_fall_can_drop_selected_item_and_it_can_be_recovered_once(request, mode, kind):
    runtime = request.getfixturevalue(mode)
    runtime.rng = drop_only_rng()
    runtime.equipment = Equipment(knife=True, weapon=kind if kind in {'PISTOL', 'RIFLE'} else None,
                                 weapons={'PISTOL', 'RIFLE'}, ammo=17, smoke=2, medkits=2,
                                 security_card=True, hazmat=True, sample_vial=True, selected=kind)
    before = runtime.equipment.snapshot()
    if mode == 'game':
        core = runtime.env
        core.grid[:] = 0
        runtime.lead_alive = False
        enemy = SecurityGuard('G1', np.array([4.5, 4.], dtype=np.float32), math.pi, ((4, 4),))
        runtime.aux_guards = [enemy]
        runtime.campaign_items.add('SECURITY_CARD')
        combat = lambda: runtime._step_guard_weapon(enemy, True)
        collect = runtime._collect_pickups
    else:
        core = runtime.core
        core.map.grid[:] = 0
        enemy = next(iter(core.units.values()))
        for other in core.units.values():
            other.alive = other is enemy
        enemy.pos, enemy.angle = np.array([4.5, 4.], dtype=np.float32), math.pi
        combat, collect = runtime._world_after_ai, runtime._collect_swat_pickups
    enemy.ammo = 0
    core.player_pos = np.array([4., 4.], dtype=np.float32)
    for _ in range(30):
        combat()
        if runtime.player_actions.prone:
            break
    drops = [p for p in runtime.pickups if p['source'] == 'PLAYER']
    assert len(drops) == 1 and drops[0]['kind'] == kind
    assert runtime.player_actions.dropped_item == kind
    assert drops[0]['amount'] == (17 if kind in {'PISTOL', 'RIFLE'} else 1)
    if kind in {'PISTOL', 'RIFLE'}:
        assert kind not in runtime.equipment.weapons and runtime.equipment.ammo == 0
    if kind == 'SECURITY_CARD' and mode == 'game':
        assert 'SECURITY_CARD' not in runtime.campaign_items
    collect()
    assert any(p['source'] == 'PLAYER' for p in runtime.pickups)
    for _ in range(50):
        runtime._apply_player(InputState())
    collect()
    restored = runtime.equipment.snapshot()
    assert not any(p['source'] == 'PLAYER' for p in runtime.pickups)
    for key in before.keys() - {'weapon', 'selected'}:
        assert restored[key] == before[key], key
    collect()
    assert runtime.equipment.snapshot() == restored


@pytest.mark.parametrize('roll, dropped', [(0., True), (.2499, True), (.25, False), (.999, False)])
def test_item_drop_chance_is_twenty_five_percent(roll, dropped):
    equipment = Equipment(knife=True, selected='KNIFE')
    pickups = []
    result = equipment.drop_on_fall(SimpleNamespace(random=lambda: roll), np.array([4., 4.]), pickups, 1)
    assert (result == 'KNIFE') == dropped
    assert equipment.knife == (not dropped)
    assert len(pickups) == int(dropped)


@pytest.mark.parametrize('mode', ['game', 'swat'])
@pytest.mark.parametrize('stance', ['walk', 'sprint', 'crouch'])
def test_player_moves_six_percent_slower_in_both_modes(request, mode, stance):
    runtime = request.getfixturevalue(mode)
    core = runtime.env if mode == 'game' else runtime.core
    grid = core.grid if mode == 'game' else core.map.grid
    grid[:] = 0
    core.player_pos = np.array([4., 4.], dtype=np.float32)
    speeds = {'walk': 2.6, 'sprint': 3.7, 'crouch': 1.3} if mode == 'game' else {'walk': 1.7, 'sprint': 2.5, 'crouch': 1.7 * .55}
    for _ in range(10):
        runtime._apply_player(InputState(right=True, sprint=stance == 'sprint', crouch=stance == 'crouch'))
    actual_speed = (core.player_pos[0] - 4.) / (runtime.game_dt * 10)
    assert actual_speed == pytest.approx(speeds[stance] * .94, abs=1e-5)


def test_dropping_the_only_weapon_preserves_the_get_up_window(game, enemy):
    game.rng = drop_only_rng()
    for _ in range(30):
        game.tick(InputState())
        if game.player_actions.prone:
            break
    assert not game.equipment.armed and game.player_actions.prone
    enemy.pos = game.env.player_pos + np.array([.5, 0.], dtype=np.float32)
    enemy.angle = math.pi
    if enemy.guard_id == 'G0':
        game.env.guard_pos, game.env.guard_angle = enemy.pos.copy(), enemy.angle
    for _ in range(6):
        game.tick(InputState())
        assert game.status == 'RUNNING'


def test_standalone_swat_does_not_instantly_capture_a_disarmed_falling_player(swat):
    swat.rng = drop_only_rng()
    swat.core.map.grid[:] = 0
    unit = next(iter(swat.core.units.values()))
    for other in swat.core.units.values():
        other.alive = other is unit
    unit.pos, unit.angle = np.array([4.5, 4.], dtype=np.float32), math.pi
    swat.core.player_pos = np.array([4., 4.], dtype=np.float32)
    for _ in range(30):
        swat._world_after_ai()
        if swat.player_actions.prone:
            break
    assert not swat.equipment.armed and swat.player_actions.prone
    unit.pos = swat.core.player_pos + np.array([.5, 0.], dtype=np.float32)
    unit.angle = math.pi
    for _ in range(6):
        swat._world_after_ai()
        assert not swat.core.player_captured
