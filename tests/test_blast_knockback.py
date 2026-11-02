import numpy as np
import pytest

from shadow_ai.blast_physics import blast_impulse, MAX_BLAST_PUSH
from shadow_ai.playable_combat import EnemyMelee
from shadow_ai.playable_navigation import HumanMotion
from shadow_ai.soliton_security import SecurityGuard, line_clear
from test_interactive_tactics import game
from test_playable_actions import swat


@pytest.fixture(params=['game', 'swat'])
def blast_runtime(request):
    runtime = request.getfixturevalue(request.param)
    world, grid = runtime._combat_world()
    grid[:] = 0
    runtime.spawn_safe, runtime.spawn_grace = False, 0.
    world.player_pos = np.array([25., 20.], dtype=np.float32)
    if hasattr(runtime, 'env'):
        runtime.lead_alive = False
        runtime.aux_guards, runtime.swat_units, runtime.cameras = [], [], []
    else:
        for actor in runtime.core.units.values():
            actor.alive = False
    return runtime


def target(runtime, point, role='guard'):
    if hasattr(runtime, 'env'):
        uid = 'G8' if role == 'guard' else 'S8'
        actor = SecurityGuard(uid, np.asarray(point, dtype=np.float32), 0., (tuple(point),))
        if role == 'guard':
            runtime.aux_guards = [actor]
        else:
            actor.has_shield = True
            runtime.swat_units = [('SHIELD', actor)]
        return uid, actor, actor.vitals
    uid, actor = next(iter(runtime.core.units.items()))
    actor.alive = True
    actor.pos = np.asarray(point, dtype=np.float32)
    return uid, actor, runtime._unit_vitals[uid]


def detonate(runtime, point=(6., 6.), kind='GRENADE'):
    point = np.array(point, dtype=np.float32)
    runtime._throw_grenade(point.copy(), point.copy(), 'PLAYER', kind=kind)
    runtime.explosives[-1].fuse = 0.
    runtime._step_explosives()


def test_radial_impulse_falls_with_distance_and_points_away_from_the_blast():
    grid = np.zeros((16, 16), dtype=np.uint8)
    center = np.array([6., 6.], dtype=np.float32)
    near = np.array([7., 6.], dtype=np.float32)
    far = np.array([8.5, 6.], dtype=np.float32)
    near_goal, near_strength = blast_impulse(grid, center, near)
    far_goal, far_strength = blast_impulse(grid, center, far)
    assert near_strength > far_strength > 0
    assert near_goal[0] - near[0] > far_goal[0] - far[0] > 0
    assert near_goal[1] == far_goal[1] == 6.
    opposite, _ = blast_impulse(grid, center, np.array([5., 6.]))
    assert opposite[0] < 5. and opposite[1] == 6.


def test_impulse_at_exact_center_is_finite_and_uses_the_incoming_direction():
    grid = np.zeros((16, 16), dtype=np.uint8)
    center = np.array([6., 6.], dtype=np.float32)
    goal, strength = blast_impulse(grid, center, center, fallback=np.array([0., 2.]))
    assert strength == 1. and np.all(np.isfinite(goal))
    assert goal[0] == 6. and goal[1] == pytest.approx(6. + MAX_BLAST_PUSH)


def test_impulse_sweeps_into_a_wall_without_snapping_through_it():
    grid = np.zeros((16, 16), dtype=np.uint8)
    grid[:, 8] = 1
    start = np.array([7.25, 6.], dtype=np.float32)
    goal, strength = blast_impulse(grid, np.array([6., 6.]), start)
    assert strength > 0 and 7.25 < goal[0] < 7.5
    assert goal[1] == 6. and line_clear(grid, start, goal)
    assert not grid[int(round(goal[1])), int(round(goal[0]))]


def test_diagonal_corner_collision_and_map_bounds_stop_the_impulse():
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[5, 5] = 1
    start = np.array([4., 4.], dtype=np.float32)
    goal, _ = blast_impulse(grid, np.array([3., 3.]), start)
    assert goal[0] < 4.5 and goal[1] < 4.5 and line_clear(grid, start, goal)
    grid[:] = 0
    for start, center in [([.1, 5.], [1., 5.]), ([8.9, 5.], [8., 5.]),
                          ([5., .1], [5., 1.]), ([5., 8.9], [5., 8.])]:
        goal, _ = blast_impulse(grid, np.asarray(center), np.asarray(start))
        assert np.all(goal >= 0) and np.all(goal <= 9)


def test_occlusion_and_blast_radius_prevent_any_impulse():
    grid = np.zeros((16, 16), dtype=np.uint8)
    center = np.array([6., 6.], dtype=np.float32)
    outside = np.array([9., 6.], dtype=np.float32)
    goal, strength = blast_impulse(grid, center, outside)
    np.testing.assert_array_equal(goal, outside)
    assert strength == 0
    grid[:, 7] = 1
    hidden = np.array([8., 6.], dtype=np.float32)
    goal, strength = blast_impulse(grid, center, hidden)
    np.testing.assert_array_equal(goal, hidden)
    assert strength == 0


def test_close_fragmentation_pushes_player_and_reuses_fall_timing_without_extra_damage(blast_runtime):
    runtime = blast_runtime
    world, _ = runtime._combat_world()
    world.player_pos = np.array([7., 6.], dtype=np.float32)
    world.player_velocity[:] = (2., 1.)
    detonate(runtime)
    assert runtime.player_vitals.health == pytest.approx(.7)
    assert world.player_pos[0] == pytest.approx(8.4) and world.player_pos[1] == 6.
    assert np.linalg.norm(world.player_velocity) == 0
    assert runtime.player_actions.prone and runtime.player_actions.knockdown_remaining == .75
    assert runtime.player_vitals.hit_stun_remaining >= 1.4
    if not hasattr(runtime, 'env'):
        assert world.player_health == pytest.approx(.7)


@pytest.mark.parametrize('role', ['guard', 'swat'])
def test_fragmentation_displaces_and_staggers_actors_and_invalidates_old_motion(blast_runtime, role):
    runtime = blast_runtime
    uid, actor, vitals = target(runtime, [7., 6.], role)
    actor.aim_timer = 1.
    runtime._motions[uid] = HumanMotion(speed=2.)
    runtime.navigator.paths[uid] = (np.array([12., 6.]), [np.array([8., 6.])])
    runtime._melee_attacks[uid] = EnemyMelee(windup=.2)
    detonate(runtime)
    assert vitals.health == pytest.approx(.7)
    assert actor.pos[0] == pytest.approx(8.4) and actor.pos[1] == 6.
    assert vitals.hit_stun_remaining > .5 and actor.aim_timer == 0
    assert runtime._motions[uid].speed == 0 and uid not in runtime.navigator.paths
    assert runtime._melee_attacks[uid].windup == 0
    if not hasattr(runtime, 'env'):
        assert actor.health == pytest.approx(.7)


def test_primary_guard_adapter_tracks_the_actual_blast_destination(game):
    game.env.grid[:] = 0
    game.aux_guards, game.swat_units = [], []
    game.env.player_pos = np.array([25., 20.], dtype=np.float32)
    game.env.guard_pos = game._lead_patrol.pos = np.array([7., 6.], dtype=np.float32)
    detonate(game)
    assert game.env.guard_pos[0] == pytest.approx(8.4)
    np.testing.assert_array_equal(game.env.guard_pos, game._lead_patrol.pos)


def test_wall_shields_both_player_and_enemy_from_damage_and_displacement(blast_runtime):
    runtime = blast_runtime
    world, grid = runtime._combat_world()
    grid[:, 7] = 1
    world.player_pos = np.array([8., 6.], dtype=np.float32)
    _, actor, vitals = target(runtime, [8., 6.])
    detonate(runtime)
    assert runtime.player_vitals.health == vitals.health == 1.
    assert world.player_pos[0] == actor.pos[0] == 8.
    assert not runtime.player_actions.prone and vitals.hit_stun_remaining == 0


def test_fatal_blast_leaves_actor_and_body_loot_at_collision_safe_fall_position(blast_runtime):
    runtime = blast_runtime
    uid, actor, vitals = target(runtime, [7., 6.])
    vitals.health = .1
    if not hasattr(runtime, 'env'):
        actor.health = .1
    detonate(runtime)
    assert vitals.health == 0 and actor.pos[0] == pytest.approx(8.4)
    assert not actor.alive
    loot = [item for item in runtime.pickups if item['source'] == uid]
    assert loot and all(np.array_equal(item['pos'], actor.pos) for item in loot)
    if hasattr(runtime, 'env'):
        body = next(body for body in runtime.bodies if body.guard_id == uid)
        np.testing.assert_array_equal(body.pos, actor.pos)


@pytest.mark.parametrize('kind', ['SMOKE', 'FLASH'])
def test_smoke_and_flash_never_add_fragmentation_damage_or_knockback(blast_runtime, kind):
    runtime = blast_runtime
    world, _ = runtime._combat_world()
    world.player_pos = np.array([7., 6.], dtype=np.float32)
    _, actor, vitals = target(runtime, [7., 6.])
    detonate(runtime, kind=kind)
    assert runtime.player_vitals.health == vitals.health == 1.
    assert world.player_pos[0] == actor.pos[0] == 7.
    assert not runtime.player_actions.prone and vitals.hit_stun_remaining == 0
