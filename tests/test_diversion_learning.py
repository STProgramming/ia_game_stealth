import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from stable_baselines3 import PPO
import torch

from shadow_ai import ShadowProtocolEnv
from shadow_ai.episode_learning import EpisodeLearning
from shadow_ai.interactive_runtime import InteractiveGuardRuntime, InputState
from shadow_ai.soliton_security import NoiseEvent


class Policy:
    def predict(self, observation, **kwargs):
        return 0, None


def runtime():
    with patch("shadow_ai.interactive_runtime.PPO.load", return_value=Policy()):
        r = InteractiveGuardRuntime(Path("fake.zip"))
    r.spawn_safe = False
    r.spawn_grace = 0
    r.lead_alive = False
    r.cameras = []
    r.env.player_pos = np.array([2., 27.], dtype=np.float32)
    for i, g in enumerate(r.aux_guards):
        g.pos = np.array([20. + i * 2, 12.], dtype=np.float32)
    return r


def noise(r, device, event=1):
    r.noise_events = [NoiseEvent(event, np.array([20., 12.], dtype=np.float32),
                                 10.5, .72, "NOISEMAKER", device_id=device)]
    r._coordinate_diversions()


def test_normal_two_guards_choose_one_without_global_caution():
    r = runtime()
    noise(r, 10)
    assert len(r._diversion_assignments[10]) == 1
    for g in r.aux_guards:
        r._step_aux_guard(g)
    r._update_facility_alert()
    assert r.facility_alert == "CALM"
    assert not r.security_compromised
    assert "resta di guardia" in r.guard_dialogue[r.aux_guards[0].guard_id]


def test_normal_single_guard_investigates():
    r = runtime()
    r.aux_guards = r.aux_guards[:1]
    noise(r, 10)
    r._step_aux_guard(r.aux_guards[0])
    r._update_facility_alert()
    assert r.aux_guards[0].state in {"INVESTIGATE", "SEARCH"}
    assert r.facility_alert == "CALM"


def test_device_pulses_do_not_count_as_repeated_diversions():
    r = runtime()
    for i in range(3):
        noise(r, 10, i + 1)
    assert set(r._diversion_counts.values()) == {1}
    assert not r.security_compromised


def test_second_device_causes_caution_and_escort():
    r = runtime()
    noise(r, 10)
    noise(r, 11)
    r._update_facility_alert()
    assert r.facility_alert == "SUSPICIOUS"
    assert r.security_compromised
    assignments = r._diversion_assignments[11]
    assert len(assignments) == 2
    assert not np.array_equal(*assignments.values())
    assert any("qualcosa di strano" in text for text in r.guard_dialogue.values())


def test_high_guard_uses_two_distinct_approaches():
    r = runtime()
    r.security_compromised = True
    noise(r, 10)
    assert len(r._diversion_assignments[10]) == 2
    assert not np.array_equal(*r._diversion_assignments[10].values())


def test_primary_participates_in_assignment_without_global_caution():
    r = runtime()
    r.lead_alive = True
    r.env.guard_pos = np.array([20., 12.], dtype=np.float32)
    noise(r, 10)
    assert "G0" in r._diversion_assignments[10]
    r._inject_primary_audio()
    r._update_facility_alert()
    assert r.facility_alert == "CALM"


def test_reset_forgets_diversion_history():
    r = runtime()
    noise(r, 10)
    noise(r, 11)
    r.reset("routine")
    assert not r._diversion_counts
    assert not r._diversion_assignments
    assert not r.security_compromised


def test_different_investigators_do_not_trigger_repeated_diversion_alarm():
    r = runtime()
    noise(r, 10)
    r.aux_guards[0].pos, r.aux_guards[1].pos = r.aux_guards[1].pos, r.aux_guards[0].pos
    noise(r, 11)
    assert not r.security_compromised


def visible_launch_runtime():
    r = runtime()
    r.equipment.casings = 1
    r.env.player_pos = np.array([20., 12.], dtype=np.float32)
    for index, guard in enumerate(r.aux_guards):
        guard.pos = np.array([20. + index * 2, 10.], dtype=np.float32)
        guard.angle = math.atan2(2., -index * 2.)
    return r


def test_seen_launch_immediately_coordinates_on_estimated_source():
    r = visible_launch_runtime()
    r.equipment.selected = "CASING"
    r._craft_or_throw(InputState(throw=True, aim_x=27, aim_y=12))
    assert len(r._launch_responses) == 2
    assert r.security_compromised
    assert all(g.state == "FLANK" for g in r.aux_guards)
    goals = [response["goal"] for response in r._launch_responses.values()]
    assert not np.array_equal(*goals)
    for response in r._launch_responses.values():
        assert np.linalg.norm(response["source"] - np.array([20., 12.])) < 1.5
        assert np.linalg.norm(response["goal"] - np.array([27., 12.])) > 4
    assert r.team_contact is None  # A visible throw is a cue, not a confirmed player contact.
    assert r.learning_events[-1]["type"] == "observed_launch"


def test_seen_launch_memory_does_not_follow_hidden_player_or_device_pulses():
    r = visible_launch_runtime()
    r.equipment.selected = "CASING"
    r._craft_or_throw(InputState(throw=True, aim_x=27, aim_y=12))
    source = r._launch_responses[r.aux_guards[0].guard_id]["source"].copy()
    goal = r.aux_guards[0].target.copy()
    r.env.player_pos = np.array([2., 27.], dtype=np.float32)
    for _ in range(40):
        r._step_diversions()
        r._step_aux_guard(r.aux_guards[0])
    assert np.array_equal(r._launch_responses[r.aux_guards[0].guard_id]["source"], source)
    assert np.array_equal(r.aux_guards[0].target, goal)
    assert r._audible_cue(r.aux_guards[0].pos, r.aux_guards[0].guard_id) is None


@pytest.mark.parametrize("reason", ["wall", "facing_away", "inactive", "spawn"])
def test_unseen_launch_does_not_reveal_source(reason):
    r = visible_launch_runtime()
    r.aux_guards = r.aux_guards[:1]
    guard = r.aux_guards[0]
    if reason == "wall":
        r.env.grid[11, 20] = 1
    elif reason == "facing_away":
        guard.angle = -math.pi / 2
    elif reason == "inactive":
        guard.unconscious = True
    else:
        r.spawn_grace = 1
    r.equipment.selected = "CASING"
    r._craft_or_throw(InputState(throw=True, aim_x=27, aim_y=12))
    assert not r._launch_responses
    assert not r.security_compromised


def test_primary_observes_launch_and_keeps_estimated_source():
    r = visible_launch_runtime()
    r.aux_guards = []
    r.lead_alive = True
    r.env.guard_pos = np.array([20., 10.], dtype=np.float32)
    r.env.guard_angle = math.pi / 2
    r.equipment.selected = "CASING"
    r.tick(InputState(throw=True, aim_x=27, aim_y=12))
    assert "G0" in r._launch_responses
    assert r.state()["guard"]["action"] == "FLANK"
    goal = r._launch_responses["G0"]["goal"].copy()
    r.env.player_pos = np.array([2., 27.], dtype=np.float32)
    for _ in range(10):
        r.tick(InputState())
    assert np.array_equal(r._primary_audio_goal, goal)


@pytest.mark.parametrize("outcome", ["CAPTURED", "ESCAPED"])
def test_game_session_finishes_once_and_reuses_learned_checkpoint(tmp_path, outcome):
    from game_server import GameSession
    model = PPO("MlpPolicy", ShadowProtocolEnv(), n_steps=8, batch_size=8, device="cpu")
    path = tmp_path / "base.zip"
    model.save(path)
    with patch("game_server.ROOT", tmp_path):
        session = GameSession(path, None)
        r = session.runtime
        r.spawn_safe = False
        r.spawn_grace = 0
        r.env.player_pos = r.env._exit.copy()
        r.env.player_has_intel[0] = 1
        with patch.object(r.env, "_terminal_events", return_value=(False, outcome == "CAPTURED")):
            state = session.tick()
        assert state["status"] == outcome
        result = session.learning.future.result(timeout=30)
        assert result["status"] == "TRAINED", result
        session.tick()
        assert len(list((tmp_path / ".runtime" / "episodes").glob("*.json"))) == 1
        session.handle({"type": "reset"})
        updated = PPO.load(result["checkpoint"])
        assert all(torch.equal(value, updated.policy.state_dict()[key])
                   for key, value in session.runtime.model.policy.state_dict().items())


@pytest.mark.parametrize("outcome", ["CAPTURED", "ESCAPED"])
def test_completed_match_saves_schema_and_changes_weights(tmp_path, outcome):
    env = ShadowProtocolEnv()
    model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, device="cpu")
    original = {key: value.clone() for key, value in model.policy.state_dict().items()}
    learning = EpisodeLearning(model, "guard", tmp_path)
    observation, _ = env.reset()
    for i in range(8):
        learning.record(observation, i % 4)
    learning.finish(outcome, 1, 71)
    result = learning.future.result(timeout=30)
    assert result["status"] == "TRAINED", result
    schema = json.loads(Path(result["schema"]).read_text(encoding="utf-8"))
    assert schema["outcome"] == outcome
    assert len(schema["samples"]) == 8
    updated = PPO.load(result["checkpoint"])
    assert any(not torch.equal(original[key], value)
               for key, value in updated.policy.state_dict().items())
    assert all(torch.equal(original[key], value) for key, value in model.policy.state_dict().items())
    learning.finish(outcome, 1, 71)
    assert len(list(tmp_path.glob("*.json"))) == 1
