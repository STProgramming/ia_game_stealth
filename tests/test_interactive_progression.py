from pathlib import Path
from unittest.mock import patch
import numpy as np
from shadow_ai.interactive_runtime import InteractiveGuardRuntime, InteractiveSwatRuntime, InputState
from shadow_ai import TacticalAction
from shadow_ai.playable_navigation import local_search_points
from shadow_ai.soliton_security import SecurityGuard, can_see, line_clear, GUARD_FOV, GUARD_VISION_RANGE
import pytest

class Policy:
    def predict(self, observation, **kwargs):
        if observation.ndim == 2:
            return np.full(len(observation), int(TacticalAction.HOLD)), None
        return int(TacticalAction.HOLD), None

def runtime():
    with patch("shadow_ai.interactive_runtime.PPO.load", return_value=Policy()):
        return InteractiveGuardRuntime(Path("fake.zip"))

def test_radio_dispatch_and_reset():
    r = runtime()
    r.spawn_grace = 0
    r.spawn_safe = False
    point = np.array([20., 12.], dtype=np.float32)
    r._publish_contact(point, "C1")
    assert not r.swat_units
    assert r.swat_phase == "SPOTTED"
    for _ in range(30):
        r._step_radio()
    assert r.swat_phase == "RADIO"
    assert not r.swat_units
    for _ in range(60):
        r._step_radio()
    assert r.swat_phase == "DISPATCH"
    assert not r.swat_units
    for _ in range(30):
        r._step_radio()
    assert len(r.swat_units) == 3
    assert r.swat_phase == "ACTIVE"
    assert all(np.array_equal(u.target, point) for _, u in r.swat_units)
    r.env.player_pos = np.array([2., 27.], dtype=np.float32)
    for _ in range(60):
        r.tick(InputState())
    assert len(r.state()["swat"]) == 3
    r.reset("routine")
    assert not r.swat_units

def test_levels_require_objectives():
    r = runtime()
    assert len(r.aux_guards) == 2
    assert len(r.state()["map"]["intel"]) == 1
    r.env.player_pos = r.env._exit.copy()
    r.tick(InputState())
    assert r.status == "RUNNING"
    r.env.player_has_intel[0] = 1
    r.tick(InputState())
    assert r.status == "ESCAPED"
    r.level = 3
    r.reset("routine")
    assert len(r.aux_guards) == 7
    assert len(r.cameras) == 6
    assert len(r.state()["map"]["intel"]) == 1
    assert r.state()["map"]["intel"][0]["name"] == "Campione biologico"

def test_regular_guard_secures_sector_after_radio():
    r = runtime()
    r.spawn_grace = 0
    r.spawn_safe = False
    r._publish_contact(np.array([44., 22.], dtype=np.float32), "C1")
    guard = r.aux_guards[0]
    r._guard_goal_from_radio(guard)
    assert guard.state == "PROTECT INTEL"
    assert np.array_equal(guard.target, r.env._intel[0])


def test_spawn_protection_all_levels_and_scenarios():
    r = runtime()
    for level in (1, 2, 3):
        r.level = level
        for scenario in r.scenarios:
            r.reset(scenario)
            g = r.aux_guards[0]
            r.env.player_pos = g.pos.copy()
            r.env.player_current_speed = 2.5
            r.tick(InputState())
            assert r.status == "RUNNING"
            assert r.team_contact is None
            assert not r.swat_units
            assert r._audible_cue(g.pos) is None


def test_heard_steps_reach_other_side_of_wall():
    r = runtime()
    r.spawn_grace = 0
    r.spawn_safe = False
    r.cameras = []
    r.lead_alive = False
    g = r.aux_guards[0]
    r.aux_guards = [g]
    g.pos = np.array([5., 20.], dtype=np.float32)
    r.env.player_pos = np.array([7., 20.], dtype=np.float32)
    r.env.player_current_speed = 1.5
    r._step_aux_guard(g)
    assert g.state == "INVESTIGATE"
    r.env.player_current_speed = 0
    for _ in range(600):
        r._step_aux_guard(g)
        assert r.env._free(*r.env._cell(g.pos))
        if g.pos[0] > 6.5:
            break
    assert g.pos[0] > 6.5


def test_patrol_route_crosses_partition_without_sticking():
    r = runtime()
    pos = np.array([30., 3.], dtype=np.float32)
    goal = np.array([36., 13.], dtype=np.float32)
    for _ in range(600):
        pos = r._navigate("test", pos, goal, 0.05)
        assert r.env._free(*r.env._cell(pos))
    assert np.linalg.norm(pos - goal) < .1


def test_primary_sensor_does_not_accumulate_during_spawn():
    r = runtime()
    for level in (1, 2, 3):
        r.level = level
        r.reset("routine")
        r.env.player_pos = r.env.guard_pos.copy()
        for _ in range(75):
            r.tick(InputState())
            assert r.status == "RUNNING"
            assert r.env.visual_confidence == 0.0
            assert not r.env.sense.visual
            assert r.team_contact is None


def test_swat_spawn_protection_every_scenario():
    with patch("shadow_ai.interactive_runtime.PPO.load", return_value=Policy()):
        r = InteractiveSwatRuntime(Path("fake.zip"))
    for scenario in r.scenarios:
        r.reset(scenario)
        unit = next(iter(r.core.units.values()))
        r.core.player_pos = unit.pos.copy()
        health = r.core.player_health
        for _ in range(75):
            r.tick(InputState())
            assert r.core.player_health == health
            assert not r.core.player_captured
            assert not r.core.player_neutralized
            assert r.status == "RUNNING"


def test_evasion_caution_and_recontact():
    r = runtime()
    r.spawn_grace = 0
    r.spawn_safe = False
    r._publish_contact(np.array([20., 12.], dtype=np.float32), "C1")
    r.team_contact_age = .6
    r._update_facility_alert()
    assert r.facility_alert == "SEARCH"
    r.team_contact_age = 13
    r._update_facility_alert()
    assert r.facility_alert == "SUSPICIOUS"
    r._publish_contact(np.array([21., 12.], dtype=np.float32), "C1")
    assert r.facility_alert == "CONTACT"
    r.team_contact_age = 33
    r._update_facility_alert()
    assert r.facility_alert == "SUSPICIOUS"
    r.team_contact_age = 300
    r._update_facility_alert()
    assert r.facility_alert == "SUSPICIOUS"
    assert r.state()["security"]["high_guard"]
    r.reset("routine")
    assert not r.state()["security"]["high_guard"]
    assert r.facility_alert == "CALM"


def test_swat_enters_and_moves_toward_radio_location():
    r = runtime()
    r.spawn_grace = 0
    r.spawn_safe = False
    r._publish_contact(np.array([30., 12.], dtype=np.float32), "C1")
    for _ in range(120):
        r._step_radio()
    assert len(r.swat_units) == 3
    positions = [u.pos.copy() for _, u in r.swat_units]
    r.env.player_pos = np.array([2., 27.], dtype=np.float32)
    for _ in range(360):
        for _, unit in r.swat_units:
            r._step_aux_guard(unit)
            assert r.env._free(*r.env._cell(unit.pos))
    assert all(np.linalg.norm(u.pos - start) > 5 for (_, u), start in zip(r.swat_units, positions))
    assert all(u.state != "PROTECT INTEL" for _, u in r.swat_units)


@pytest.mark.parametrize("route_index", range(7))
def test_all_patrols_finish_a_full_route(route_index):
    r = runtime()
    r.level = 3
    r.reset("routine")
    r.spawn_grace = 0
    r.spawn_safe = False
    r.env.player_pos = np.array([0., 29.], dtype=np.float32)
    g = r.aux_guards[route_index]
    for target in g.patrol[1:] + g.patrol[:1]:
        goal = np.asarray(target, dtype=np.float32)
        for _ in range(1200):
            before = g.pos.copy()
            g.pos = r._navigate(g.guard_id, g.pos, goal, .06)
            assert r.env._free(*r.env._cell(g.pos))
            if np.linalg.norm(g.pos - goal) < .1:
                break
        assert np.linalg.norm(g.pos - goal) < .1, (route_index, target, before, g.pos)


def test_spawn_positions_navigable_and_movement_all_scenarios():
    r = runtime()
    for scenario in r.scenarios:
        for direction in (InputState(up=True), InputState(down=True), InputState(left=True), InputState(right=True)):
            r.reset(scenario)
            assert r.env._free(*r.env._cell(r.env.player_pos)), scenario
            assert r.env._free(*r.env._cell(r.env.guard_pos)), scenario
            old = r.env.player_pos.copy()
            r.tick(direction)
            assert np.linalg.norm(r.env.player_pos - old) > 0, (scenario, direction)


def test_cameras_mount_on_wall_faces_and_point_outward():
    r = runtime()
    r.level = 3
    r.reset("routine")
    for camera in r.cameras:
        assert r.env._free(*r.env._cell(camera.pos))
        walls = np.argwhere(r.env.grid)
        distances = [np.linalg.norm(camera.pos - np.array([x, y])) for y, x in walls]
        assert abs(min(distances) - .58) < .01, camera.camera_id
        front = camera.pos + np.array([np.cos(camera.center_angle), np.sin(camera.center_angle)]) * .4
        behind = camera.pos - np.array([np.cos(camera.center_angle), np.sin(camera.center_angle)]) * .2
        assert r.env._free(*r.env._cell(front))
        assert not r.env._free(*r.env._cell(behind))


@pytest.mark.parametrize("level", (1, 2, 3))
def test_spawn_stays_safe_after_grace_in_every_guard_scenario(level):
    r = runtime()
    r.level = level
    for scenario in r.scenarios:
        r.reset(scenario)
        spawn = r.env.player_pos.copy()
        assert not any(can_see(r.env.grid, g.pos, g.angle, spawn, 11.5, 122)
                       for g in r.aux_guards)
        for _ in range(180):
            r.tick(InputState())
        assert r.spawn_grace == 0
        assert r.spawn_protected
        assert r.state()["gameplay"]["spawn_protected"]
        assert r.status == "RUNNING"
        assert r.team_contact is None
        assert not r.env.sense.visual
        assert not r.swat_units


def test_guard_spawn_can_be_left_and_protection_does_not_return():
    r = runtime()
    for _ in range(90):
        r.tick(InputState(down=True))
    assert r.spawn_protected
    for _ in range(180):
        r.tick(InputState(right=True))
        if not r.spawn_protected:
            break
    assert not r.spawn_protected
    assert not r.spawn_safe
    r.env.player_pos = np.array([2., 27.], dtype=np.float32)
    r.tick(InputState())
    assert not r.spawn_protected
    r.reset("routine")
    assert r.spawn_protected


def test_swat_spawn_stays_safe_and_can_be_left_in_all_scenarios():
    with patch("shadow_ai.interactive_runtime.PPO.load", return_value=Policy()):
        r = InteractiveSwatRuntime(Path("fake.zip"))
    for scenario in r.scenarios:
        r.reset(scenario)
        for _ in range(120):
            r.tick(InputState())
        assert r.spawn_grace == 0
        assert r.spawn_protected
        assert r.status == "RUNNING"
        assert r.core.player_health == 1.0
        assert not any(u.local_visual for u in r.core.units.values())
        # The alcove has an actual navigable route into the facility.
        pos = r.core.player_pos.copy()
        goal = np.array([8., 2.], dtype=np.float32)
        for _ in range(800):
            pos = r.navigator.move("exit-test", pos, goal, .04)
            assert r.core.map.free(*r.core.map.cell(pos))
            if np.linalg.norm(pos - goal) < .01:
                break
        assert np.linalg.norm(pos - goal) < .01
        r.core.player_pos = pos
        r.tick(InputState())
        assert not r.spawn_protected
        r.core.player_pos = np.array([2., 2.], dtype=np.float32)
        r.tick(InputState())
        assert not r.spawn_protected


def test_guard_swat_search_visits_local_corners_and_scans():
    r = runtime()
    r.spawn_grace = 0
    r.spawn_safe = False
    anchor = np.array([9., 20.], dtype=np.float32)
    r.env.player_pos = np.array([2., 27.], dtype=np.float32)
    g = SecurityGuard("S0", anchor.copy(), 0., ((46, 10), (46, 12)))
    r._start_guard_search(g, anchor)
    r.swat_units = [("SHIELD", g)]
    r.team_contact = anchor.copy()
    r.team_contact_age = 30
    r._update_facility_alert()
    assert r.facility_alert == "SEARCH"
    visited = set()
    scans = []
    for _ in range(8000):
        old, angle = g.pos.copy(), g.angle
        r._step_aux_guard(g)
        assert r.env._free(*r.env._cell(g.pos))
        assert line_clear(r.env.grid, old, g.pos)
        visited.add(g.search_step)
        if np.linalg.norm(g.pos - old) < 1e-6 and abs(g.angle - angle) > .01:
            scans.append(g.angle)
        if g.state == "WITHDRAW":
            break
    assert g.state == "WITHDRAW"
    assert len(visited) >= 5
    assert len(scans) >= 100
    assert np.array_equal(g.last_seen, anchor)


def test_search_plan_uses_geometry_and_not_hidden_player_position():
    r = runtime()
    anchor = np.array([18., 20.], dtype=np.float32)
    r.env.player_pos = np.array([2., 27.], dtype=np.float32)
    first = local_search_points(r.env.grid, anchor)
    r.env.player_pos = np.array([44., 2.], dtype=np.float32)
    second = local_search_points(r.env.grid, anchor)
    assert np.array_equal(first, second)
    assert len(first) == 6
    for point in first:
        assert r.env._free(*r.env._cell(point))
        assert np.linalg.norm(point - anchor) <= 7.1


def test_new_contact_redirects_swat_search_without_duplicate_restarts():
    r = runtime()
    r.spawn_grace = 0
    r.spawn_safe = False
    anchor = np.array([20., 12.], dtype=np.float32)
    r._publish_contact(anchor, "C1")
    for _ in range(120):
        r._step_radio()
    g = r.swat_units[0][1]
    r._start_guard_search(g, anchor)
    search = r._searches[g.guard_id]
    r._publish_contact(anchor, "C1")
    assert r._searches[g.guard_id] is search
    assert g.state == "SEARCH"
    new = np.array([30., 12.], dtype=np.float32)
    r._publish_contact(new, "C1")
    assert g.state == "RESPOND"
    assert g.guard_id not in r._searches
    assert np.array_equal(g.target, new)


def test_swat_mode_completes_search_after_radio_report_expires():
    with patch("shadow_ai.interactive_runtime.PPO.load", return_value=Policy()):
        r = InteractiveSwatRuntime(Path("fake.zip"))
    r.reset("lost_contact")
    r.spawn_grace = 0
    r.spawn_safe = False
    uid = r.core.agent_ids[0]
    unit = r.core.units[uid]
    unit.pos = np.array([15., 10.], dtype=np.float32)
    assert r._step_swat_search(uid)
    search = r._searches[uid]
    r.core.blackboard.contact_pos = None
    r.core.blackboard.contact_age = 100
    for point in search.points:
        unit.pos = point.copy()
        for _ in range(73):
            assert r._step_swat_search(uid)
    assert search.index == len(search.points)
    assert r.actions[uid] == "SEARCH COMPLETE"
    unit.local_visual = True
    assert not r._step_swat_search(uid)
    assert not r._searches


def test_navigation_clamps_targets_and_never_crosses_walls():
    r = runtime()
    pos = np.array([2., 27.], dtype=np.float32)
    goal = np.array([-10., 40.], dtype=np.float32)
    for _ in range(100):
        old = pos.copy()
        pos = r._navigate("edge", pos, goal, .06)
        assert 0 <= pos[0] <= 47 and 0 <= pos[1] <= 29
        assert line_clear(r.env.grid, old, pos)
    assert np.linalg.norm(pos - [0, 29]) < .01


@pytest.mark.parametrize("mode", ("guard", "swat"))
def test_navigation_reaches_goals_from_fractional_positions(mode):
    with patch("shadow_ai.interactive_runtime.PPO.load", return_value=Policy()):
        r = (InteractiveGuardRuntime if mode == "guard" else InteractiveSwatRuntime)(Path("fake.zip"))
    nav = r.navigator
    rng = np.random.default_rng(713)
    free = np.argwhere(nav.grid == 0)
    for i in range(30):
        cells = free[rng.integers(0, len(free), size=2)]
        pos = nav.goal(cells[0][::-1] + rng.uniform(-.4, .4, size=2))
        goal = nav.goal(cells[1][::-1] + rng.uniform(-.4, .4, size=2))
        start = pos.copy()
        for _ in range(1800):
            old = pos.copy()
            pos = nav.move(f"fraction-{i}", pos, goal, .08)
            assert nav.free(*nav.cell(pos))
            assert line_clear(nav.grid, old, pos)
            if np.linalg.norm(pos - goal) < .01:
                break
        assert np.linalg.norm(pos - goal) < .01, (mode, start, goal, pos)


@pytest.mark.parametrize("alert", ("CALM", "SUSPICIOUS", "SEARCH", "CONTACT"))
def test_guard_cones_share_sensor_geometry_in_every_phase(alert):
    r = runtime()
    r.level = 3
    r.reset("routine")
    r.facility_alert = alert
    r.swat_units = [("SHIELD", SecurityGuard("S0", np.array([46., 10.]), 0., ((46, 10),)))]
    snapshot = r.state()
    for guard in snapshot["guards"] + snapshot["swat"]:
        assert guard["fov"] == GUARD_FOV == r.env.cfg.main_fov_degrees == r.env.cfg.peripheral_fov_degrees
        assert guard["vision_range"] == GUARD_VISION_RANGE == r.env.cfg.vision_outer
    assert all(guard["vision_status"] == "NORMAL" for guard in snapshot["guards"] + snapshot["swat"])


def test_patrols_pause_scan_and_resume_with_bounded_turns():
    r = runtime()
    guard = r.aux_guards[0]
    start = guard.pos.copy()
    stationary, walking = 0, 0
    for _ in range(240):
        old, angle = guard.pos.copy(), guard.angle
        r._step_aux_guard(guard)
        delta = abs(np.arctan2(np.sin(guard.angle-angle), np.cos(guard.angle-angle)))
        assert delta <= 2 * r.game_dt + 1e-6
        assert line_clear(r.env.grid, old, guard.pos)
        if np.linalg.norm(guard.pos-old) < 1e-6:
            stationary += 1
        else:
            walking += 1
    assert stationary >= 50
    assert walking >= 50
    assert np.linalg.norm(guard.pos-start) > 1.0


def test_primary_patrol_also_pauses_and_walks():
    r = runtime()
    r.spawn_grace = 0
    r.spawn_safe = False
    r.aux_guards, r.cameras = [], []
    r.env.player_pos = np.array([0., 29.], dtype=np.float32)
    actions = set()
    for _ in range(240):
        old, angle = r.env.guard_pos.copy(), r.env.guard_angle
        r.tick(InputState())
        actions.add(r.state(False)["guard"]["action"])
        assert line_clear(r.env.grid, old, r.env.guard_pos)
        delta = abs(np.arctan2(np.sin(r.env.guard_angle-angle), np.cos(r.env.guard_angle-angle)))
        assert delta <= 2 * r.game_dt + 1e-6
    assert {"OBSERVE", "PATROL"} <= actions


def test_patrol_pause_is_interrupted_by_a_real_sighting():
    r = runtime()
    r.spawn_grace = 0
    r.spawn_safe = False
    guard = r.aux_guards[0]
    r.env.player_pos = np.array([0., 29.], dtype=np.float32)
    r._step_aux_guard(guard)
    assert guard.state == "OBSERVE"
    r.env.player_pos = guard.pos + np.array([1., 0.], dtype=np.float32)
    r._step_aux_guard(guard)
    assert guard.state == "CHASE"
    assert guard.pause_remaining == 0
    assert r.security_compromised


def test_swat_withdraws_off_map_and_can_be_dispatched_again():
    r = runtime()
    r.spawn_grace = 0
    r.spawn_safe = False
    r.env.player_pos = np.array([0., 29.], dtype=np.float32)
    point = np.array([44., 10.], dtype=np.float32)
    r._publish_contact(point, "C1")
    for _ in range(120):
        r._step_radio()
    r.team_contact_age = 100
    deployed = [unit for _, unit in r.swat_units]
    for unit in deployed:
        unit.pos = np.array([44., 13.], dtype=np.float32)
        r._start_guard_search(unit, unit.pos)
        r._searches[unit.guard_id].index = len(r._searches[unit.guard_id].points)
    for _ in range(1200):
        for _, unit in r.swat_units:
            old = unit.pos.copy()
            r._step_aux_guard(unit)
            if unit.state not in {"LEAVING", "EXITED"}:
                assert line_clear(r.env.grid, old, unit.pos)
        r._step_swat_departures()
        r._update_facility_alert()
        if not r.swat_units:
            break
    assert not r.swat_units
    assert all(unit.pos[0] >= r.env.cfg.width for unit in deployed)
    assert r.swat_phase == "WITHDRAWN"
    assert r.facility_alert == "SUSPICIOUS"
    r._publish_contact(point, "C1")
    assert r.swat_phase == "SPOTTED"
    for _ in range(120):
        r._step_radio()
    assert len(r.swat_units) == 3


def test_new_radio_contact_interrupts_swat_withdrawal():
    r = runtime()
    r.spawn_grace = 0
    r.spawn_safe = False
    unit = SecurityGuard("S0", np.array([40., 10.]), 0., ((46, 10),))
    unit.state = "WITHDRAW"
    r.swat_units = [("SHIELD", unit)]
    r.swat_phase = "RETURNING"
    point = np.array([30., 12.], dtype=np.float32)
    r._publish_contact(point, "C1")
    r._step_swat_departures()
    assert unit.state == "RESPOND"
    assert np.array_equal(unit.target, point)
    assert r.swat_phase == "ACTIVE"


def test_standalone_swat_leaves_and_stops_sensing_after_search():
    with patch("shadow_ai.interactive_runtime.PPO.load", return_value=Policy()):
        r = InteractiveSwatRuntime(Path("fake.zip"))
    r.spawn_grace = 0
    r.spawn_safe = False
    r._searched_contact = np.array([15., 10.], dtype=np.float32)
    r.core.blackboard.contact_pos = None
    r.core.blackboard.contact_age = 100
    for index, uid in enumerate(r.core.agent_ids):
        unit = r.core.units[uid]
        unit.pos = np.array([30., 7.+index*.4], dtype=np.float32)
        for _ in range(400):
            r._step_swat_withdrawal(uid)
            if uid in r._departed:
                break
        assert uid in r._departed
        r.core.player_pos = unit.pos.copy()
        assert not r._swat_can_see_player(unit)
    assert not r.state()["swat"]
    assert r.state()["security"]["swat_phase"] == "WITHDRAWN"
