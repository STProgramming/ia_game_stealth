from pathlib import Path
from unittest.mock import patch
from collections import deque
import numpy as np
import pytest
from shadow_ai.interactive_runtime import InputState, InteractiveSwatRuntime
from shadow_ai.playable_actions import PlayerActions, projectile_damage
from shadow_ai.playable_vitals import PlayableVitals
from shadow_ai.playable_navigation import HumanMotion
from test_interactive_tactics import game, HoldPolicy


@pytest.fixture
def swat():
    with patch('shadow_ai.interactive_runtime.PPO.load', return_value=HoldPolicy()):
        runtime = InteractiveSwatRuntime(Path('fake.zip'))
    runtime.spawn_safe = False
    runtime.spawn_grace = 0
    return runtime


def test_player_faces_movement_and_ignores_mouse_when_stationary():
    actions = PlayerActions()
    pos = np.array([4., 4.])
    actions.step(InputState(aim_x=4, aim_y=9), pos, .1)
    assert actions.angle == 0
    actions.step(InputState(up=True, aim_x=9, aim_y=4), pos, .1)
    assert actions.angle == pytest.approx(-np.pi / 2)
    actions.step(InputState(aim_x=9, aim_y=4), pos, .1)
    assert actions.angle == pytest.approx(-np.pi / 2)


def test_stalled_pursuit_replans_and_moves_without_crossing_walls(game):
    guard = game.aux_guards[0]
    guard.pos = np.array([4., 4.], dtype=np.float32)
    guard.state = "CHASE"
    start = guard.pos.copy()
    goal = np.array([8., 4.], dtype=np.float32)
    with patch.object(HumanMotion, "move", side_effect=lambda nav, key, pos, angle, *args: (pos.copy(), angle)):
        for _ in range(40):
            guard.pos, guard.angle = game._guard_move(guard, goal, 2.)
            assert game.env._free(*game.env._cell(guard.pos))
    assert np.linalg.norm(guard.pos - start) > .05


@pytest.mark.parametrize("shield", (False, True))
def test_swat_shield_limits_actual_sight_and_reported_cone(swat, shield):
    uid = next(uid for uid, unit in swat.core.units.items() if unit.profile.has_shield == shield)
    unit = swat.core.units[uid]
    unit.pos = np.array([4., 4.], dtype=np.float32)
    unit.angle = 0
    swat.core.map.grid[:] = 0
    radians = np.deg2rad(47)
    swat.core.player_pos = unit.pos + np.array([np.cos(radians), np.sin(radians)]) * 6
    assert swat.core._unit_can_see_player(unit) == (not shield)
    swat.core.player_pos = unit.pos + np.array([6., 0.])
    assert swat.core._unit_can_see_player(unit)


def test_f_tap_is_lethal_only_on_release_and_hold_only_chokes():
    actions = PlayerActions()
    pos = np.array([4., 4.])
    pressed = InputState(attack_held=True)
    actions.step(pressed, pos, .1)
    assert not pressed.lethal and not pressed.choke
    released = InputState()
    actions.step(released, pos, .03)
    assert released.lethal and not released.choke
    pressed = InputState(attack_held=True)
    actions.step(pressed, pos, .3)
    assert pressed.choke and not pressed.lethal
    released = InputState()
    actions.step(released, pos, .03)
    assert not released.lethal and not released.choke


def test_frontal_knife_costs_energy_and_takes_quarter_health(game):
    game.lead_alive = False
    guard = game.aux_guards[0]
    game.aux_guards = [guard]
    game.env.player_pos = np.array([6., 4.], dtype=np.float32)
    guard.pos = np.array([5., 4.], dtype=np.float32)
    guard.angle = 0
    game._process_takedown(InputState(lethal=True))
    assert guard.vitals.health == .75 and guard.active
    assert game.player_vitals.stamina == pytest.approx(.78)
    assert guard.visual_confidence == 1 and game.team_contact is not None
    game._process_takedown(InputState(lethal=True))
    assert guard.vitals.health == .75  # cooldown stops button spam
    game.player_actions.melee_cooldown = 0
    game.player_vitals.stamina = .1
    game._process_takedown(InputState(lethal=True))
    assert guard.vitals.health == .75


def test_rear_knife_kills_and_choke_uses_more_energy(game):
    game.lead_alive = False
    guard = game.aux_guards[0]
    game.aux_guards = [guard]
    game.env.player_pos = np.array([4., 4.], dtype=np.float32)
    guard.pos = np.array([5., 4.], dtype=np.float32)
    guard.angle = 0
    game._process_takedown(InputState(lethal=True))
    assert not guard.alive and guard.vitals.health == 0
    game.reset('routine')
    game.lead_alive = False
    guard = game.aux_guards[0]
    game.aux_guards = [guard]
    game.env.player_pos = np.array([4., 4.], dtype=np.float32)
    guard.pos = np.array([5., 4.], dtype=np.float32)
    guard.angle = 0
    for _ in range(92):
        game._process_takedown(InputState(choke=True))
    assert guard.unconscious and guard.vitals.health == 1
    assert game.player_vitals.stamina < .3


def test_low_energy_interrupts_choke(game):
    game.lead_alive = False
    guard = game.aux_guards[0]
    game.env.player_pos = np.array([4., 4.], dtype=np.float32)
    guard.pos = np.array([5., 4.], dtype=np.float32)
    guard.angle = 0
    game.player_vitals.stamina = .15
    game._process_takedown(InputState(choke=True))
    assert game.takedown
    game._process_takedown(InputState(choke=True))
    assert game.takedown is None and not guard.unconscious


def test_dive_stops_at_walls_costs_energy_and_makes_noise(game):
    game.env.grid[:] = 0
    game.env.grid[:, 6] = 1
    game.env.player_pos = np.array([4., 4.], dtype=np.float32)
    game._apply_player(InputState(right=True, dive=True, aim_x=10, aim_y=4))
    assert 4. < game.env.player_pos[0] < 5.5
    assert game.player_vitals.stamina < .73
    assert any(n.source == 'DIVE IMPACT' for n in game.noise_events)
    assert game.state()['player']['diving']
    stamina = game.player_vitals.stamina
    game._apply_player(InputState(dive=True))
    assert game.player_vitals.stamina == stamina


def test_health_gauges_require_player_facing_los_and_no_smoke(game):
    game.env.grid[:] = 0
    game.env.player_pos = np.array([4., 4.], dtype=np.float32)
    guard = game.aux_guards[0]
    guard.pos = np.array([8., 4.], dtype=np.float32)
    game.player_actions.angle = 0
    assert game._guard_state(guard)['observed']
    game.player_actions.angle = np.pi
    assert not game._guard_state(guard)['observed']
    game.player_actions.angle = 0
    game.env.grid[:, 6] = 1
    assert not game._guard_state(guard)['observed']
    game.env.grid[:] = 0
    game.smoke_clouds = [dict(pos=np.array([6., 4.]), radius=1., ttl=2.)]
    assert not game._guard_state(guard)['observed']


@pytest.mark.parametrize('weapon,armor_loss', [('PISTOL', .16), ('RIFLE', .32)])
def test_armor_absorbs_damage_and_weapon_controls_degradation(weapon, armor_loss):
    vitals = PlayableVitals()
    health, armor, _ = projectile_damage(1., 1., vitals, False, 0., np.array([5., 4.]), np.array([6., 4.]), weapon)
    assert armor == pytest.approx(1. - armor_loss)
    assert .94 < health < 1
    bare_health, _, _ = projectile_damage(1., 0., vitals, False, 0., np.array([5., 4.]), np.array([6., 4.]), weapon)
    assert bare_health < health


def test_shield_blocks_frontal_health_damage_but_drains_vigour():
    vitals = PlayableVitals()
    health, armor, effect = projectile_damage(1., 1., vitals, True, 0., np.array([5., 4.]), np.array([6., 4.]), 'RIFLE')
    assert health == armor == 1 and effect == 'SHIELD'
    assert vitals.stamina == pytest.approx(.7)
    rear_health, rear_armor, _ = projectile_damage(1., 1., vitals, True, 0., np.array([5., 4.]), np.array([4., 4.]), 'RIFLE')
    assert rear_health < 1 and rear_armor < 1


def test_bandage_selected_in_backpack_is_used_by_quick_e(game):
    game.pickups = []
    game.player_vitals.take_hit(.12)
    game.equipment.bandages = 1
    game._apply_player(InputState(interact_held=True, selected_item='BANDAGE'))
    assert game.equipment.bandages == 1
    game._apply_player(InputState())
    assert game.player_vitals.treatment == 'BANDAGE' and game.equipment.bandages == 0
    for _ in range(29):
        game._apply_player(InputState())
    assert game.player_vitals.bleeding == 0


@pytest.mark.parametrize('mode', ['guard', 'swat'])
def test_fixed_medical_station_requires_held_e_and_is_consumed_after_four_seconds(game, swat, mode):
    runtime = game if mode == 'guard' else swat
    station = next(i for i in runtime.pickups if i['kind'] == 'MEDKIT')
    if mode == 'guard':
        runtime.env.player_pos = station['pos'].copy()
    else:
        runtime.core.player_pos = station['pos'].copy()
        runtime.core.player_health = .4
    runtime.player_vitals.health = .4
    runtime.pickups = [station]
    runtime._apply_player(InputState(interact=True))
    assert runtime.equipment.medkits == 0 and runtime.pickups
    for _ in range(60):
        runtime._apply_player(InputState(interact_held=True))
    assert runtime.pickups and .4 < runtime.player_vitals.health < .5
    for _ in range(70):
        runtime._apply_player(InputState(interact_held=True))
    assert not runtime.pickups
    assert runtime.player_vitals.health >= .5 - 1e-6
    assert runtime.player_vitals.regen_remaining > .24


def test_releasing_e_interrupts_station_without_consuming_it(game):
    station = next(i for i in game.pickups if i['kind'] == 'MEDKIT')
    game.pickups = [station]
    game.env.player_pos = station['pos'].copy()
    game.player_vitals.take_hit(.12)
    for _ in range(40):
        game._apply_player(InputState(interact_held=True))
    assert game.player_vitals.treatment == 'MEDKIT'
    game._apply_player(InputState())
    assert game.player_vitals.treatment is None
    assert game.pickups and game.player_vitals.bleeding > 0


def test_selected_casing_launches_one_pulse_and_cannot_be_thrown_again(game):
    game.equipment.casings = 1
    game.equipment.select('CASING')
    game._craft_or_throw(InputState(throw=True, aim_x=12, aim_y=27))
    assert game.equipment.casings == 0 and len(game.diversions) == 1
    assert game.diversions[0].pulses_left == 1
    game._craft_or_throw(InputState(throw=True, aim_x=12, aim_y=27))
    assert len(game.diversions) == 1


def test_swat_knife_and_selected_magazine_work(swat):
    swat.core.map.grid[:] = 0
    unit = next(iter(swat.core.units.values()))
    swat.core.player_pos = np.array([6., 4.], dtype=np.float32)
    unit.pos = np.array([5., 4.], dtype=np.float32)
    unit.angle = 0
    swat._swat_melee(InputState(lethal=True))
    assert unit.health == .75 and swat.player_vitals.stamina == pytest.approx(.78)
    swat.equipment.magazines = 1
    swat.equipment.select('MAGAZINE')
    swat._swat_player_equipment(InputState(throw=True, aim_x=10, aim_y=4))
    assert swat.equipment.magazines == 0 and swat._thrown


def test_all_contextual_supplies_are_reachable_from_spawn(game, swat):
    for runtime in (game, swat):
        nav = runtime.navigator
        start = nav.cell(runtime.env.player_pos if runtime is game else runtime.core.player_pos)
        visited, queue = {start}, deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                point = x+dx,y+dy
                if point not in visited and nav.free(*point):
                    visited.add(point); queue.append(point)
        for item in runtime.pickups:
            assert nav.cell(item['pos']) in visited
            assert item['zone']
            if item['kind'] == 'MEDKIT':
                assert item['fixed'] and item['zone'] == 'INFERMERIA'


def test_e_near_intel_does_not_also_throw_selected_grenade(game):
    game.level = 1
    game.reset('routine')
    game.pickups = []
    game.env.player_pos = game.env._intel[0].copy()
    game.equipment.flash = 1
    game.equipment.select('FLASH')
    game._apply_player(InputState(interact=True))
    assert game.env.player_has_intel[0] == 1
    assert game.equipment.flash == 1 and not game.flash_events


def test_even_a_nearby_enemy_behind_player_has_hidden_vitals(game):
    game.env.grid[:] = 0
    game.env.player_pos = np.array([4., 4.], dtype=np.float32)
    game.player_actions.angle = 0
    assert not game._player_observes(np.array([3.7, 4.]))


def test_backpack_remembers_multiple_weapons_and_selects_only_owned_items(game):
    game.pickups = [dict(id='p',kind='PISTOL',amount=8,pos=game.env.player_pos.copy(),source='MAP'),
                    dict(id='r',kind='RIFLE',amount=6,pos=game.env.player_pos.copy(),source='G9')]
    game._collect_pickups()
    assert game.equipment.weapons == {'PISTOL','RIFLE'}
    assert game.equipment.select('PISTOL') and game.equipment.weapon == 'PISTOL'
    assert game.equipment.select('KNIFE') and game.equipment.weapon is None
    assert not game.equipment.select('BANDAGE')


@pytest.mark.parametrize("runtime", ["game", "swat"])
def test_dive_requires_explicit_timed_get_up(request, runtime):
    game = request.getfixturevalue(runtime)
    core = game.env if runtime == "game" else game.core
    grid = core.grid if runtime == "game" else core.map.grid
    grid[:] = 0
    core.player_pos = np.array([4., 4.], dtype=np.float32)
    game._apply_player(InputState(right=True, dive=True))
    landing = core.player_pos.copy()
    for _ in range(40):
        game._apply_player(InputState())
    assert game.player_actions.prone
    np.testing.assert_array_equal(core.player_pos, landing)
    game._apply_player(InputState(right=True, dive=True))
    assert game.player_actions.snapshot()["rising"]
    for _ in range(10):
        game._apply_player(InputState(right=True))
        np.testing.assert_array_equal(core.player_pos, landing)
    for _ in range(15):
        game._apply_player(InputState(right=True))
    assert not game.player_actions.prone
    assert core.player_pos[0] > landing[0]


@pytest.mark.parametrize("runtime", ["game", "swat"])
def test_extended_rear_takedown_range_and_walls(request, runtime):
    game = request.getfixturevalue(runtime)
    if runtime == "game":
        game.lead_alive = False
        target = game.aux_guards[0]
        game.aux_guards = [target]
        core, grid = game.env, game.env.grid
        melee = game._process_takedown
    else:
        core, grid = game.core, game.core.map.grid
        target = next(iter(core.units.values()))
        for unit in core.units.values():
            unit.alive = unit is target
        melee = game._swat_melee
    grid[:] = 0
    core.player_pos = np.array([4., 4.], dtype=np.float32)
    target.pos = np.array([5.9, 4.], dtype=np.float32)
    target.angle = 0
    grid[:, 5] = 1
    melee(InputState(choke=True))
    assert game.takedown is None
    grid[:] = 0
    target.pos[0] = 6.1
    melee(InputState(choke=True))
    assert game.takedown is None
    target.pos[0] = 5.9
    melee(InputState(choke=True))
    assert game.takedown is not None
    melee(InputState(choke=True))
    assert game.takedown is not None


def test_player_can_outrun_regular_guards_and_walk_faster_than_patrols(game, swat):
    assert game.env.cfg.player_walk_speed > game.env.cfg.guard_speed * 1.4
    assert game.env.cfg.player_sprint_speed > game.env.cfg.guard_speed * 2.4
    assert swat.core.cfg.player_walk_speed > max(.95 * u.profile.move_multiplier for u in swat.core.units.values())



def test_caution_increases_guard_movement_speed(game):
    game.env.grid[:] = 0
    game.env.player_pos = np.array([40., 25.], dtype=np.float32)
    guard = game.aux_guards[0]
    guard.pos = np.array([4., 4.], dtype=np.float32)
    guard.angle = 0
    guard.state = "PATROL"
    guard.patrol = [np.array([8., 4.], dtype=np.float32)]
    game.facility_alert = "CALM"
    with patch.object(game, "_guard_move", return_value=(guard.pos.copy(), guard.angle)) as move:
        game._step_aux_guard(guard)
        normal_speed = move.call_args.args[2]
        game.facility_alert = "SUSPICIOUS"
        game._step_aux_guard(guard)
        assert move.call_args.args[2] == pytest.approx(normal_speed * 1.4)


@pytest.mark.parametrize("runtime", ["game", "swat"])
def test_movement_gets_player_up_after_dive(request, runtime):
    game = request.getfixturevalue(runtime)
    core = game.env if runtime == "game" else game.core
    grid = core.grid if runtime == "game" else core.map.grid
    grid[:] = 0
    core.player_pos = np.array([4., 4.], dtype=np.float32)
    game._apply_player(InputState(right=True, dive=True))
    landing = core.player_pos.copy()
    for _ in range(20):
        game._apply_player(InputState(right=True, sprint=True))
        np.testing.assert_array_equal(core.player_pos, landing)
    for _ in range(20):
        game._apply_player(InputState(right=True, sprint=True))
    assert not game.player_actions.movement_locked
    assert core.player_pos[0] > landing[0]


def test_early_get_up_press_is_buffered_until_landing():
    actions = PlayerActions()
    pos = np.array([4., 4.], dtype=np.float32)
    grid = np.zeros((20, 20), dtype=np.int8)
    actions.dive(InputState(dive=True), pos, grid, PlayableVitals())
    actions.step(InputState(dive=True), pos, .03)
    for _ in range(40):
        actions.step(InputState(), pos, .03)
    assert not actions.movement_locked


@pytest.mark.parametrize("runtime", ["game", "swat"])
@pytest.mark.parametrize("wall", [False, True])
def test_knife_miss_is_visible_and_only_impacts_make_noise(request, runtime, wall):
    game = request.getfixturevalue(runtime)
    if runtime == "game":
        core, grid, melee = game.env, game.env.grid, game._process_takedown
        game.lead_alive = False
        game.aux_guards = []
    else:
        core, grid, melee = game.core, game.core.map.grid, game._swat_melee
        for unit in core.units.values():
            unit.alive = False
    grid[:] = 0
    core.player_pos = np.array([4., 4.], dtype=np.float32)
    if wall:
        grid[:, 6] = 1
    melee(InputState(lethal=True))
    assert game.player_actions.snapshot()["knife_remaining"] > 0
    assert game.player_actions.knife_impact == wall
    assert game.player_vitals.stamina == pytest.approx(.78)
    if runtime == "game":
        assert any(n.source == "KNIFE IMPACT" for n in game.noise_events) == wall
    else:
        assert (game._gunshot_cue is not None) == wall
        if wall:
            assert game._gunshot_cue["radius"] == 6.
    melee(InputState(lethal=True))
    assert game.player_vitals.stamina == pytest.approx(.78)
    for _ in range(12):
        game.player_actions.step(InputState(), core.player_pos, game.game_dt)
    assert game.player_actions.snapshot()["knife_remaining"] == 0


def test_heavy_breathing_at_zero_energy_attracts_guard_at_rest(game):
    game.env.grid[:] = 0
    game.env.player_pos = np.array([4., 4.], dtype=np.float32)
    game.env.player_current_speed = 0
    game.player_vitals.stamina = 0
    game.player_vitals.exhausted = True
    guard = game.aux_guards[0]
    guard.pos = np.array([7., 4.], dtype=np.float32)
    guard.angle = 0
    assert game.player_vitals.snapshot()["panting"]
    assert game._audible_cue(guard.pos, guard.guard_id)[1] == "HEAVY BREATHING"
    assert game._audible_cue(np.array([10., 4.])) is None
    game._step_aux_guard(guard)
    assert guard.state == "INVESTIGATE"
    assert game._guard_state(guard)["vision_status"] == "SUSPICIOUS"
    assert game.team_contact is None
    game.player_vitals.stamina = .35
    game.player_vitals.exhausted = False
    assert not game.player_vitals.panting
    assert game._audible_cue(guard.pos, guard.guard_id) is None


def test_swat_hears_heavy_breathing_within_five_cells(swat):
    swat.core.map.grid[:] = 0
    swat.core.player_pos = np.array([4., 4.], dtype=np.float32)
    swat.core.player_velocity[:] = 0
    unit = next(iter(swat.core.units.values()))
    unit.pos = np.array([7., 4.], dtype=np.float32)
    unit.angle = 0
    swat.player_vitals.stamina = 0
    swat.player_vitals.exhausted = True
    swat._sense_swat_units()
    assert unit.local_sound and not unit.local_visual
    unit.pos = np.array([10., 4.], dtype=np.float32)
    swat._sense_swat_units()
    assert not unit.local_sound



def test_pistol_shot_triggers_global_alert_without_visual_contact(game):
    game.env.grid[:] = 0
    game.env.player_pos = np.array([4., 4.], dtype=np.float32)
    for guard in game.aux_guards:
        guard.pos = np.array([40., 20.], dtype=np.float32)
        guard.visual_confidence = 0
    game.equipment.weapon = "PISTOL"
    game.equipment.ammo = 2
    game._player_equipment_action(InputState(fire=True, aim_x=12, aim_y=4))
    game._update_facility_alert()
    assert game.equipment.ammo == 1
    assert game.facility_alert == "CONTACT"
    assert game.threat_level == 2
    assert game.team_contact_source == "GUNFIRE"
    assert all(g.suspicion == 1 and g.state == "RESPOND" for g in game.aux_guards)
    assert all(g.visual_confidence == 0 for g in game.aux_guards)
    assert all(g["combat_alert"] for g in game.state()["guards"])


def test_smoke_raises_caution_without_dispatching_swat_or_lowering_alert(game):
    game.equipment.smoke = 2
    game._player_equipment_action(InputState(use_smoke=True, aim_x=12, aim_y=4))
    game._update_facility_alert()
    assert game.equipment.smoke == 1
    assert game.facility_alert == "SUSPICIOUS"
    assert game.threat_level == 1
    assert game.team_contact is None
    assert game.swat_phase == "QUIET" and game.radio_timer is None
    assert all(g.suspicion >= .48 for g in game.aux_guards)
    game._player_security_event("GUNFIRE", game.env.player_pos)
    game._player_equipment_action(InputState(use_smoke=True, aim_x=12, aim_y=4))
    game._update_facility_alert()
    assert game.threat_level == 2 and game.facility_alert == "CONTACT"


@pytest.mark.parametrize("source", ["PISTOL", "SMOKE"])
def test_missing_equipment_does_not_raise_alert(game, source):
    game.equipment.weapon = "PISTOL"
    game.equipment.ammo = 0
    game.equipment.smoke = 0
    game._player_equipment_action(InputState(fire=source == "PISTOL", use_smoke=source == "SMOKE", aim_x=12, aim_y=4))
    game._update_facility_alert()
    assert game.facility_alert == "CALM"
    assert not game.security_compromised


def test_standalone_swat_pistol_shares_audio_and_smoke_cannot_lower_alert(swat):
    swat.core.map.grid[:] = 0
    swat.core.player_pos = np.array([4., 4.], dtype=np.float32)
    swat.equipment.weapon = "PISTOL"
    swat.equipment.ammo = 2
    swat.equipment.smoke = 2
    swat._swat_player_equipment(InputState(use_smoke=True, aim_x=12, aim_y=4))
    assert swat._equipment_alert == "SUSPICIOUS"
    swat._swat_player_equipment(InputState(fire=True, aim_x=12, aim_y=4))
    assert swat._equipment_alert == "CONTACT"
    assert swat._gunshot_cue["radius"] > swat.core.cfg.width
    swat._swat_player_equipment(InputState(use_smoke=True, aim_x=12, aim_y=4))
    assert swat._equipment_alert == "CONTACT"
