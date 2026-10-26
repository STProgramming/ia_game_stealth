import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from shadow_ai.playable_vitals import PlayableVitals
from shadow_ai.interactive_runtime import InputState, InteractiveSwatRuntime
from shadow_ai.soliton_security import BodyEvidence
from test_interactive_tactics import game, HoldPolicy


def test_stamina_exhaustion_delayed_recovery_and_repeated_fatigue():
    vitals = PlayableVitals()
    for _ in range(190):
        vitals.step(True, 1 / 30)
    assert vitals.exhausted and not vitals.can_sprint
    assert vitals.stamina == 0
    assert vitals.fatigue == pytest.approx(.08)
    for _ in range(50):
        vitals.step(False, 1 / 30)
    assert vitals.stamina == 0
    for _ in range(130):
        vitals.step(False, 1 / 30)
    assert 0.35 <= vitals.stamina < 1
    assert vitals.can_sprint
    first_speed = vitals.speed_multiplier
    for _ in range(20):
        vitals.step(True, 1 / 30)
    vitals.step(False, 1 / 30)
    assert vitals.fatigue == pytest.approx(.16)
    assert vitals.speed_multiplier < first_speed


def test_boost_only_starts_on_new_confirmed_contact(game):
    game.env.player_pos = np.array([22., 12.], dtype=np.float32)
    game._apply_player(InputState(right=True, sprint=True))
    normal = game.env.player_current_speed
    point = game.env.player_pos.copy()
    game._publish_contact(point, "G0")
    assert game.player_vitals.boost_remaining == 2.5
    game._apply_player(InputState(right=True, sprint=True))
    assert game.env.player_current_speed == pytest.approx(normal * 1.65, abs=6e-5)
    remaining = game.player_vitals.boost_remaining
    game._publish_contact(game.env.player_pos, "G0")
    assert game.player_vitals.boost_remaining == remaining
    for _ in range(75):
        game._apply_player(InputState())
    assert game.player_vitals.boost_remaining == 0


@pytest.mark.parametrize("tension", ("previous_enemy", "repeated_strong", "first_strong"))
def test_search_buddy_requires_prior_tension_or_repeated_eighty_percent(game, tension):
    source = game.aux_guards[0]
    if tension == "previous_enemy":
        game.security_compromised = True
    else:
        game._register_spot(source.guard_id, .8)
        for _ in range(10):
            game._register_spot(source.guard_id, .9)
        assert game._strong_spots[source.guard_id] == 1
        if tension == "repeated_strong":
            game._register_spot(source.guard_id, 0)
            game._register_spot(source.guard_id, .8)
    game._request_support(source.guard_id, source.pos)
    assert bool(game._support_calls) == (tension != "first_strong")
    for _ in range(36):
        game._step_support_radio()
    buddies = [g for g in game.aux_guards if g.state == "SUPPORT"]
    assert len(buddies) == (0 if tension == "first_strong" else 1)
    assert not game.swat_units


@pytest.mark.parametrize("lethal", (True, False))
def test_body_discovery_calls_regular_reinforcements_and_accelerates_weapons(game, lethal):
    guard = game.aux_guards[0]
    guard.pos = np.array([4., 4.], dtype=np.float32)
    guard.angle = 0
    body = BodyEvidence(999, "G99", np.array([7., 4.], dtype=np.float32), 0., lethal, 0.)
    game.bodies.append(body)
    before = len(game.aux_guards)
    assert game._detect_evidence(guard)
    assert body.discovered and game.threat_level == 2
    assert game._guard_dispatch is not None
    assert game._support_calls
    assert not game.swat_units and game.radio_timer is None
    for _ in range(60):
        game._step_guard_dispatch()
    assert len(game.aux_guards) == before + 2
    assert game._regular_reinforcements == 2
    game.env.player_pos = np.array([8., 4.], dtype=np.float32)
    for _ in range(11):
        game._step_guard_weapon(guard, True)
    assert game.player_vitals.health < 1
    assert guard.aim_timer < .95
    assert game.shots


def test_health_and_stamina_are_real_state_and_takedown_respects_walls(game):
    guard = game.aux_guards[0]
    guard.pos = np.array([5., 4.], dtype=np.float32)
    guard.angle = 0
    game.env.player_pos = np.array([4., 4.], dtype=np.float32)
    candidate = game.state()["takedown_candidate"]
    assert candidate and candidate["guard_id"] == guard.guard_id
    game.equipment.knife = True
    game._process_takedown(InputState(lethal=True))
    assert guard.vitals.health == 0 and not guard.active
    assert game.state()["guards"][1]["health"] == 0
    assert "stamina" in game.state()["guards"][0]
    assert "stamina" in game.state()["player"]
    other = game.aux_guards[1]
    other.pos = np.array([6.55, 20.], dtype=np.float32)
    other.angle = 0
    game.env.player_pos = np.array([5.45, 20.], dtype=np.float32)
    assert game.env._free(*game.env._cell(other.pos))
    assert game.env._free(*game.env._cell(game.env.player_pos))
    assert game.state()["takedown_candidate"] is None


def test_guard_cannot_sprint_forever(game):
    guard = game.aux_guards[0]
    guard.pos = np.array([4., 4.], dtype=np.float32)
    guard.state = "CHASE"
    guard.vitals.stamina = .002
    goal = np.array([7., 4.], dtype=np.float32)
    for _ in range(4):
        guard.pos, guard.angle = game._guard_move(guard, goal, 2.0)
    assert guard.vitals.exhausted
    assert guard.vitals.stamina == 0
    assert guard.vitals.recovery_delay > 0


def test_dedicated_swat_mode_shares_stamina_and_confirmed_contact_boost():
    with patch("shadow_ai.interactive_runtime.PPO.load", return_value=HoldPolicy()):
        runtime = InteractiveSwatRuntime(Path("fake.zip"))
    runtime.spawn_safe = False
    runtime.spawn_grace = 0
    unit = runtime.core.units[runtime.core.agent_ids[0]]
    runtime.core.player_pos = unit.pos.copy()
    runtime._apply_player(InputState(down=True, sprint=True))
    normal = float(np.linalg.norm(runtime.core.player_velocity))
    with patch.object(runtime.core, "_automatic_contact_resolution", return_value=0):
        runtime._world_after_ai()
    assert runtime.player_vitals.boost_remaining == 2.5
    runtime._apply_player(InputState(down=True, sprint=True))
    assert np.linalg.norm(runtime.core.player_velocity) == pytest.approx(normal * 1.65, abs=5e-5)
    remaining = runtime.player_vitals.boost_remaining
    with patch.object(runtime.core, "_automatic_contact_resolution", return_value=0):
        runtime._world_after_ai()
    assert runtime.player_vitals.boost_remaining == remaining
    vitals = runtime._unit_vitals[unit.uid]
    vitals.stamina = .002 / vitals.stamina_capacity
    runtime._unit_vitals_stepped.clear()
    runtime._move_swat_unit(unit, runtime.navigator.goal(unit.pos + np.array([0., 1.], dtype=np.float32)))
    assert vitals.exhausted and vitals.stamina == 0
    snapshot = runtime.state()
    assert snapshot["swat"][0]["exhausted"]
    assert snapshot["player"]["stamina"] < 1
