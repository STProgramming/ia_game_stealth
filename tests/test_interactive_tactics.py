from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from shadow_ai import TacticalAction
from shadow_ai.interactive_runtime import InteractiveGuardRuntime, InputState
from shadow_ai.soliton_security import SecurityGuard, line_clear


class HoldPolicy:
    def predict(self, observation, **kwargs):
        return int(TacticalAction.HOLD), None


@pytest.fixture
def game():
    with patch("shadow_ai.interactive_runtime.PPO.load", return_value=HoldPolicy()):
        runtime = InteractiveGuardRuntime(Path("fake.zip"))
    runtime.level = 3
    runtime.reset("routine")
    runtime.spawn_safe = False
    runtime.spawn_grace = 0
    runtime.cameras = []
    return runtime


def test_first_partial_spot_is_a_solo_investigation(game):
    source = game.aux_guards[0]
    source.pos = np.array([4., 4.], dtype=np.float32)
    source.angle = 0
    source.visual_confidence = .25
    game.env.player_pos = np.array([8., 4.], dtype=np.float32)
    game._step_aux_guard(source)
    game._update_facility_alert()
    assert source.state == "VERIFY"
    assert game.facility_alert == "CALM"
    assert game.team_contact is None
    assert game.swat_phase == "QUIET"
    assert not any(g.state == "SUPPORT" for g in game.aux_guards)
    for _ in range(35):
        game._step_support_radio()
    assert not any(g.state == "SUPPORT" for g in game.aux_guards)
    game._step_support_radio()
    assert not game._support_calls
    assert not any(g.state == "SUPPORT" for g in game.aux_guards)
    assert game.radio_updates == 0
    assert game.facility_alert == "CALM"
    game._request_support(source.guard_id, source.target)
    assert not game._support_calls


def test_radio_call_is_cancelled_if_guard_is_incapacitated(game):
    source = game.aux_guards[0]
    game.security_compromised = True
    game._request_support(source.guard_id, source.pos)
    source.unconscious = True
    for _ in range(40):
        game._step_support_radio()
    assert not game._support_calls
    assert game.radio_updates == 0
    assert not any(g.state == "SUPPORT" for g in game.aux_guards)


def test_security_assigns_real_intel_and_exfil_and_adapts_to_theft(game):
    game._publish_contact(np.array([30., 12.], dtype=np.float32), "G0")
    for g in game.aux_guards:
        game._guard_goal_from_radio(g)
    assert any(g.state == "SECURE EXFIL" and np.array_equal(g.target, game.env._exit)
               for g in game.aux_guards)
    assert any(g.state == "PROTECT INTEL" and np.array_equal(g.target, game.env._intel[2])
               for g in game.aux_guards)
    game.env.player_has_intel[:] = 1
    game.env.known_intel_present[:] = 0
    for g in game.aux_guards:
        game._guard_goal_from_radio(g)
        assert g.state == "SECURE EXFIL"
        assert np.array_equal(g.target, game.env._exit)


def deploy(game, point):
    game._publish_contact(np.array(point, dtype=np.float32), "C1")
    for _ in range(120):
        game._step_radio()
    game.env.player_pos = np.array([2., 27.], dtype=np.float32)


def test_swat_follows_searching_guard_without_hidden_player_truth(game):
    deploy(game, [20, 12])
    game.team_contact_age = 1
    tracker = game.aux_guards[0]
    tracker.pos = np.array([22., 12.], dtype=np.float32)
    tracker.last_seen_age = 2
    game._start_guard_search(tracker, np.array([24., 12.], dtype=np.float32))
    tracker.target = np.array([25., 12.], dtype=np.float32)
    unit = game.swat_units[0][1]
    game._step_aux_guard(unit)
    assert unit.state == "RESPOND"
    assert np.array_equal(unit.target, tracker.target)
    assert not np.array_equal(unit.target, game.env.player_pos)


@pytest.mark.parametrize("point", ([24, 6], [44, 22]))
def test_tactical_stack_grenades_and_distinct_corner_searches(game, point):
    deploy(game, point)
    for _, unit in game.swat_units:
        assert game._step_tactical_entry(unit)
    plans = [game._room_entries[u.guard_id] for _, u in game.swat_units]
    assert all(np.array_equal(plan["door"], plans[0]["door"]) for plan in plans)
    assert len({tuple(plan["stage"]) for plan in plans}) == 3
    for _, unit in game.swat_units:
        unit.pos = game._room_entries[unit.guard_id]["stage"].copy()
    # The player can be affected only at the reported position, inside the room.
    game.env.player_pos = game.team_contact.copy()
    for _ in range(20):
        for _, unit in game.swat_units:
            game._step_tactical_entry(unit)
    assert len(game.flash_events) == len(game.smoke_clouds) == 1
    assert game.player_flash_timer > 0
    assert all(p["phase"] == "CLEAR ROOM" for p in plans)
    first_corners = {tuple(p["search"].points[1]) for p in plans}
    assert len(first_corners) == 3
    for _, unit in game.swat_units:
        plan = game._room_entries[unit.guard_id]
        for point in plan["search"].points:
            unit.pos = point.copy()
            for _ in range(27):
                game._step_tactical_entry(unit)
        assert plan["phase"] == "DONE"
        assert unit.state == "SEARCH"
    assert len(game.flash_events) == len(game.smoke_clouds) == 1


def test_smoke_blocks_sensors_but_not_navigation_and_reset_clears_effects(game):
    game.env.guard_pos = np.array([4., 4.], dtype=np.float32)
    game.env.guard_angle = 0
    game.env.player_pos = np.array([8., 4.], dtype=np.float32)
    game.env.visual_confidence = 1
    game.smoke_clouds = [{"pos": np.array([6., 4.]), "radius": 1.5, "ttl": 5.}]
    packet = game._sense_primary()
    assert not packet.visual_candidate
    assert not packet.visual
    assert game.env._line_clear(game.env.guard_pos, game.env.player_pos)
    moved = game.navigator.move("smoke-test", game.env.guard_pos, game.env.player_pos, .1)
    assert np.linalg.norm(moved - game.env.guard_pos) > 0
    assert line_clear(game.env.grid, game.env.guard_pos, moved)
    game.player_flash_timer = 2.4
    game._apply_player(InputState(right=True))
    assert game.env.player_current_speed == pytest.approx(game.env.cfg.player_walk_speed / game.game_dt * .28, abs=2e-5)
    assert game.state()["player"]["flashed"]
    assert len(game.state()["smoke"]) == 1
    game.reset("routine")
    assert not game.smoke_clouds and not game.flash_events
    assert not game._room_entries and not game._support_calls
    assert game.player_flash_timer == 0


def test_flash_does_not_affect_player_through_room_wall(game):
    deploy(game, [24, 6])
    # Less than three units from the blast, but outside the closed side wall.
    game.env.player_pos = np.array([27., 6.], dtype=np.float32)
    for _, unit in game.swat_units:
        game._step_tactical_entry(unit)
    for _, unit in game.swat_units:
        unit.pos = game._room_entries[unit.guard_id]["stage"].copy()
    for _ in range(20):
        for _, unit in game.swat_units:
            game._step_tactical_entry(unit)
    assert len(game.flash_events) == 1
    assert game.player_flash_timer == 0


def test_effects_expire_and_tick_preserves_sensor_occlusion(game):
    game.env.player_pos = np.array([2., 27.], dtype=np.float32)
    game.lead_alive = False
    game.aux_guards = []
    game.player_flash_timer = game.game_dt
    game.smoke_clouds = [{"pos": np.array([6., 4.]), "radius": 1.5, "ttl": game.game_dt}]
    game.flash_events = [{"pos": np.array([6., 4.]), "ttl": game.game_dt}]
    game.tick(InputState())
    assert not game.state()["smoke"]
    assert not game.state()["flashes"]
    assert not game.state()["player"]["flashed"]
