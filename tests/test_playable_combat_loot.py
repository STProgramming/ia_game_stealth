from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from shadow_ai.interactive_runtime import InputState, InteractiveSwatRuntime
from shadow_ai.soliton_security import SecurityGuard
from shadow_ai.playable_combat import lane_blocked
from test_interactive_tactics import game, HoldPolicy


def test_confirmed_contact_requests_nearby_guard_instead_of_swat(game):
    game.lead_alive = False
    spotter, helper = game.aux_guards[:2]
    game.aux_guards = [spotter, helper]
    spotter.pos = np.array([4., 4.], dtype=np.float32)
    helper.pos = np.array([6., 4.], dtype=np.float32)
    point = np.array([8., 4.], dtype=np.float32)
    game._publish_contact(point, spotter.guard_id)
    assert game.radio_timer is None
    assert game.swat_phase == "LOCAL SUPPORT"
    assert game._support_calls[spotter.guard_id]["helper"] == helper.guard_id
    for _ in range(36):
        game._step_support_radio()
    assert helper.state == "SUPPORT" and np.array_equal(helper.target, point)
    assert not game.swat_units


@pytest.mark.parametrize("reason", ("far", "unconscious", "unreachable"))
def test_swat_is_dispatched_when_no_available_guard_is_nearby(game, reason):
    game.lead_alive = False
    spotter, helper = game.aux_guards[:2]
    game.aux_guards = [spotter, helper]
    spotter.pos = np.array([4., 4.], dtype=np.float32)
    helper.pos = np.array([6., 4.], dtype=np.float32)
    if reason == "far":
        helper.pos = np.array([40., 4.], dtype=np.float32)
    elif reason == "unconscious":
        helper.unconscious = True
    else:
        game.env.grid[:, 5] = 1
    game._publish_contact(np.array([7., 4.], dtype=np.float32), spotter.guard_id)
    assert game.swat_phase == "SPOTTED"
    for _ in range(120):
        game._step_radio()
    assert len(game.swat_units) == 3


def test_losing_assigned_helper_escalates_to_swat(game):
    game.lead_alive = False
    spotter, helper = game.aux_guards[:2]
    game.aux_guards = [spotter, helper]
    spotter.pos = np.array([4., 4.], dtype=np.float32)
    helper.pos = np.array([6., 4.], dtype=np.float32)
    game._publish_contact(np.array([8., 4.], dtype=np.float32), spotter.guard_id)
    helper.unconscious = True
    for _ in range(36):
        game._step_support_radio()
    assert game.swat_phase == "SPOTTED"
    assert game.radio_timer is not None


@pytest.mark.parametrize("confidence", (.3, 1.0))
def test_hold_policy_cannot_freeze_primary_guard_on_visual(game, confidence):
    game.aux_guards = []
    game.env.guard_pos = np.array([4., 4.], dtype=np.float32)
    game.env.guard_angle = 0
    game.env.player_pos = np.array([8., 4.], dtype=np.float32)
    game.env.visual_confidence = confidence
    game.env.sense = game._sense_primary()
    start = game.env.guard_pos.copy()
    for _ in range(30):
        game.tick(InputState())
    assert np.linalg.norm(game.env.guard_pos - start) > .2
    assert game.state()["guard"]["action"] in {"CHASE", "VERIFY"}


@pytest.mark.parametrize("unit_id", ("G1", "S0"))
def test_guard_and_reinforcement_hold_fire_and_flank_blocking_teammate(game, unit_id):
    game.env.grid[:] = 0
    game.lead_alive = False
    shooter = SecurityGuard(unit_id, np.array([4., 4.], dtype=np.float32), 0, ((4, 4),))
    blocker = SecurityGuard("G2", np.array([6., 4.], dtype=np.float32), 0, ((6, 4),))
    game.aux_guards = [blocker] + ([shooter] if unit_id.startswith("G") else [])
    game.swat_units = [("SHIELD", shooter)] if unit_id.startswith("S") else []
    game.env.player_pos = np.array([8., 4.], dtype=np.float32)
    shooter.visual_confidence = 1
    shooter.last_seen = game.env.player_pos.copy()
    for _ in range(40):
        game._step_guard_weapon(shooter, True)
    assert game.player_vitals.health == 1 and shooter.ammo == 16
    assert shooter.fire_goal is not None
    assert not lane_blocked(shooter.fire_goal, game.env.player_pos, [blocker.pos])
    moved_sideways = False
    for _ in range(180):
        game._step_aux_guard(shooter)
        moved_sideways |= abs(float(shooter.pos[1]) - 4) > .5
        game._step_guard_weapon(shooter, shooter.visual_confidence >= 1)
        if game.player_vitals.health < 1:
            break
    assert moved_sideways and game.player_vitals.health < 1
    assert blocker.vitals.health == 1


def test_start_has_knife_and_map_supplies_are_reachable_and_consumed_once(game):
    snapshot = game.state()
    assert snapshot["player"]["diversions"] == snapshot["player"]["scrap"] == 0
    assert snapshot["player"]["weapon"] is None and snapshot["player"]["knife"]
    assert snapshot["player"]["ammo"] == snapshot["player"]["flash"] == snapshot["player"]["smoke"] == 0
    kinds = {p["kind"] for p in game.pickups}
    assert kinds == {"MAGAZINE", "SMOKE", "BANDAGE", "MEDKIT", "HAZMAT", "SAMPLE_VIAL", "SNACK"}
    assert not kinds & {"SCRAP", "DIVERSION", "KNIFE", "PISTOL", "AMMO"}
    for item in game.pickups:
        assert game.env._free(*game.env._cell(item["pos"]))
        assert game.env._astar(game.env.player_pos, item["pos"])
    item = next(p for p in game.pickups if p["kind"] == "MAGAZINE")
    game.env.player_pos = item["pos"].copy()
    game._apply_player(InputState(interact=True))
    assert game.equipment.magazines == item["amount"]
    game._apply_player(InputState(interact=True))
    assert game.equipment.magazines == item["amount"]
    game.equipment.selected = "MAGAZINE"
    game._craft_or_throw(InputState(throw=True, aim_x=12, aim_y=26))
    assert game.equipment.magazines == item["amount"] - 1
    assert len(game.diversions) == 1


def test_unarmed_player_can_stun_and_loot_but_cannot_use_knife(game):
    game.equipment.knife = False
    guard = game.aux_guards[0]
    guard.pos = np.array([5., 4.], dtype=np.float32)
    guard.angle = 0
    game.env.player_pos = np.array([4., 4.], dtype=np.float32)
    assert not game.state()["takedown_candidate"]["can_lethal"]
    game._process_takedown(InputState(lethal=True))
    assert guard.active and guard.vitals.health == 1
    for _ in range(92):
        game._process_takedown(InputState(choke=True))
    assert guard.unconscious and guard.vitals.health == 1
    loot = [p for p in game.pickups if p["source"] == guard.guard_id]
    assert {p["kind"] for p in loot} == {"PISTOL", "MAGAZINE"}
    assert game.equipment.weapon is None
    game._apply_player(InputState(interact=True))
    assert game.equipment.weapon == "PISTOL" and game.equipment.ammo == 16
    assert not game.equipment.knife
    assert not any(p["source"] == guard.guard_id for p in game.pickups)


def test_weapon_pickup_shooting_damage_ammo_and_enemy_loot(game):
    game.lead_alive = False
    guard = game.aux_guards[0]
    guard.pos = np.array([8., 4.], dtype=np.float32)
    game.aux_guards = [guard]
    game.env.player_pos = np.array([4., 4.], dtype=np.float32)
    game._player_equipment_action(InputState(fire=True, aim_x=8, aim_y=4))
    assert guard.vitals.health == 1
    donor = SecurityGuard("G9", np.array([4., 5.], dtype=np.float32), 0, ((4, 5),))
    game.aux_guards.append(donor)
    game._drop_guard_equipment(donor.guard_id, donor.pos)
    donor.alive = False
    pistol = next(p for p in game.pickups if p["kind"] == "PISTOL")
    game.env.player_pos = pistol["pos"].copy()
    game._collect_pickups()
    game.env.player_pos = np.array([4., 4.], dtype=np.float32)
    for _ in range(3):
        game.equipment.fire_cooldown = 0
        game._player_equipment_action(InputState(fire=True, aim_x=8, aim_y=4))
    assert not guard.alive and guard.vitals.health == 0
    assert game.equipment.ammo == 13
    assert any(p["source"] == guard.guard_id for p in game.pickups)
    assert any(event.source == "GUNSHOT" for event in game.noise_events)


def test_shots_and_pickups_do_not_cross_walls(game):
    game.lead_alive = False
    guard = game.aux_guards[0]
    guard.pos = np.array([10., 18.], dtype=np.float32)
    game.aux_guards = [guard]
    game.env.player_pos = np.array([8., 18.], dtype=np.float32)
    game.equipment.weapon, game.equipment.ammo = "PISTOL", 1
    game._player_equipment_action(InputState(fire=True, aim_x=10, aim_y=18))
    assert guard.vitals.health == 1 and game.equipment.ammo == 0
    game.pickups = [{"id":"behind-wall","kind":"AMMO","pos":np.array([9.55,18.]),"amount":12,"source":"MAP"}]
    game.env.player_pos = np.array([8.45,18.], dtype=np.float32)
    game._collect_pickups()
    assert game.equipment.ammo == 0 and len(game.pickups) == 1


def test_standalone_swat_does_not_fire_through_squad_and_can_collect_weapon():
    with patch("shadow_ai.interactive_runtime.PPO.load", return_value=HoldPolicy()):
        runtime = InteractiveSwatRuntime(Path("fake.zip"))
    runtime.spawn_safe = False
    runtime.spawn_grace = 0
    runtime.core.map.grid[:] = 0
    units = list(runtime.core.units.values())
    units[0].pos = np.array([4., 10.], dtype=np.float32)
    units[1].pos = np.array([6., 10.], dtype=np.float32)
    for i, unit in enumerate(units[2:]):
        unit.pos = np.array([23.+i, 20.], dtype=np.float32)
    for unit in units:
        unit.angle = 0
        unit.ammo = 0
    units[0].ammo = 3
    runtime.core.player_pos = np.array([8., 10.], dtype=np.float32)
    runtime._world_after_ai()
    assert runtime.core.player_health == 1 and units[0].ammo == 3
    assert runtime._step_swat_firing_position(units[0].uid)
    assert runtime.actions[units[0].uid] == "REPOSITION"
    units[-1].ammo = 7
    runtime._down_swat_player_target(units[-1], False)
    pistol = next(p for p in runtime.pickups if p["kind"] == units[-1].weapon)
    runtime.core.player_pos = pistol["pos"].copy()
    runtime._apply_player(InputState(interact=True))
    assert runtime.equipment.weapon == units[-1].weapon and runtime.equipment.ammo == 7
