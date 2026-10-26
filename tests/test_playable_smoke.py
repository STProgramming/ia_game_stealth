import numpy as np
import pytest
from collections import deque
from shadow_ai.playable_smoke import smoke_density, smoke_visibility, smoke_escape
from shadow_ai.playable_vitals import PlayableVitals
from shadow_ai.playable_facility import build_research_grid
from shadow_ai.interactive_runtime import InputState
from shadow_ai import TacticalAction
from shadow_ai.swat import SmokeCloud
from test_interactive_tactics import game
from test_playable_actions import swat


def test_smoke_drains_vigour_then_caps_health_damage_and_recovers_outside():
    vitals = PlayableVitals()
    vitals.step_smoke(1., 1.)
    assert vitals.health == 1. and vitals.stamina == pytest.approx(.65)
    for _ in range(100):
        vitals.step_smoke(1., .1)
        vitals.step(False, .1)
    assert vitals.stamina == 0 and not vitals.can_sprint and not vitals.can_aim
    assert vitals.health == pytest.approx(.9)
    assert vitals.bleeding == 0
    vitals.step_smoke(0., .1)
    vitals.step(False, 6.)
    assert vitals.can_sprint and vitals.can_aim


def test_smoke_visibility_is_reduced_by_95_percent_including_crossing_rays():
    clouds = [dict(pos=np.array([4.,4.]),radius=2.,ttl=8.)]
    assert smoke_visibility(np.array([4.,4.]),np.array([5.,4.]),clouds) == .05
    assert smoke_visibility(np.array([1.,4.]),np.array([8.,4.]),clouds) == .05
    assert smoke_visibility(np.array([1.,1.]),np.array([8.,1.]),clouds) == 1.


def test_guard_leaves_smoke_using_a_reachable_path_and_stops_aiming(game):
    guard = game.aux_guards[0]
    game.env.grid[:] = 0
    game.env.player_pos = np.array([25.,25.],dtype=np.float32)
    guard.pos = np.array([4.,4.],dtype=np.float32)
    guard.aim_timer = .8
    game.smoke_clouds = [dict(pos=guard.pos.copy(),radius=2.,ttl=8.)]
    goal = smoke_escape(game.navigator,guard.pos,game.smoke_clouds)
    assert goal is not None and game.navigator.path_length(guard.pos,goal) < float('inf')
    for _ in range(180):
        guard.vitals.step_smoke(smoke_density(guard.pos,game.smoke_clouds),game.game_dt)
        game._step_aux_guard(guard)
        if not smoke_density(guard.pos,game.smoke_clouds):
            break
    assert not smoke_density(guard.pos,game.smoke_clouds)
    assert guard.aim_timer == 0 and guard.vitals.health >= .9


def test_player_in_smoke_cannot_shoot_or_sprint_with_no_vigour(game):
    game.player_vitals.stamina = .001
    game.equipment.weapon = 'PISTOL'
    game.equipment.ammo = 3
    game.smoke_clouds = [dict(pos=game.env.player_pos.copy(),radius=3.,ttl=8.)]
    game._apply_player(InputState(right=True,sprint=True,fire=True))
    assert game.player_vitals.exhausted and not game.player_vitals.can_aim
    assert game.equipment.ammo == 3
    assert game.env.player_current_speed <= game.env.cfg.player_walk_speed / game.game_dt


def test_swat_uses_smoke_escape_and_does_not_shoot_exhausted(swat):
    swat.core.map.grid[:] = 0
    unit = next(iter(swat.core.units.values()))
    unit.pos = np.array([10.,10.],dtype=np.float32)
    swat.core.player_pos = np.array([11.,10.],dtype=np.float32)
    for other in swat.core.units.values():
        if other.uid != unit.uid:
            other.alive = False
    swat._unit_vitals[unit.uid].stamina = 0
    swat.core.smoke_clouds = [SmokeCloud((10.,10.),radius=2.,ttl=8.)]
    start = unit.pos.copy()
    ammo = unit.ammo
    # Skip batched policy inference: this test isolates the tactical response.
    swat.policy_tick = 1
    swat.tick(InputState())
    assert swat.actions[unit.uid] == 'ESCAPE SMOKE'
    assert unit.ammo == ammo and swat.core.player_health == 1
    assert swat._unit_vitals[unit.uid].exhausted
    for _ in range(50):
        swat._move_swat_unit(unit,unit.smoke_goal)
    assert np.linalg.norm(unit.pos-start) > .05


def test_primary_keeps_moving_to_remembered_contact_despite_hold_policy(game):
    game.env.grid[:] = 0
    game.aux_guards = []
    game.env.guard_pos = np.array([4.,4.],dtype=np.float32)
    game.env.guard_angle = 0
    game.env.player_pos = np.array([30.,25.],dtype=np.float32)
    game.env.last_seen_pos = np.array([8.,4.],dtype=np.float32)
    game.env.last_seen_age = 1.
    game.last_action = TacticalAction.HOLD
    start = game.env.guard_pos.copy()
    for _ in range(30):
        game.tick(InputState())
    assert np.linalg.norm(game.env.guard_pos-start) > .15


def test_partial_spot_investigation_searches_instead_of_sticking_at_target(game):
    guard = game.aux_guards[0]
    guard.pos = np.array([4.,4.],dtype=np.float32)
    guard.target = guard.pos.copy()
    guard.state = 'VERIFY'
    guard.visual_confidence = .3
    game.env.player_pos = np.array([30.,25.],dtype=np.float32)
    game._step_aux_guard(guard)
    assert guard.state == 'SEARCH'
    assert guard.guard_id in game._searches


@pytest.mark.parametrize('compact', [False,True])
def test_research_floorplan_has_no_disconnected_free_rooms(compact):
    grid = build_research_grid(32 if compact else 48,20 if compact else 30,compact)
    seen = {(0,0)}
    queue = deque(seen)
    while queue:
        x,y = queue.popleft()
        for nx,ny in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]) and not grid[ny,nx] and (nx,ny) not in seen:
                seen.add((nx,ny)); queue.append((nx,ny))
    assert len(seen) == np.count_nonzero(grid == 0)
