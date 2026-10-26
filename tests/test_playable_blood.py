from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from shadow_ai.playable_vitals import PlayableVitals
from shadow_ai.playable_blood import BloodTrail, observe_blood
from shadow_ai.soliton_security import BloodMark
from shadow_ai.interactive_runtime import InputState, InteractiveSwatRuntime
from test_interactive_tactics import game, HoldPolicy


def test_running_wound_loses_more_health_vigour_and_drops_than_resting():
    results = []
    for speed in (0., 7.):
        vitals = PlayableVitals()
        vitals.take_hit(.12)
        trail, marks = BloodTrail(), []
        for i in range(30):
            old = np.array([i * speed * .1, 2.])
            pos = old + np.array([speed * .1, 0.])
            loss = vitals.step_bleeding(.1, speed)
            trail.deposit(marks, old, pos, loss, lambda: len(marks) + 1)
        results.append((vitals, marks))
    resting, running = results
    assert resting[0].health < .88
    assert running[0].health < resting[0].health
    assert running[0].stamina < resting[0].stamina < 1
    assert running[0].fatigue > resting[0].fatigue > 0
    assert len(running[1]) > len(resting[1]) > 0
    assert any(m.pos[0] > 0 for m in running[1])


def test_treatment_stops_loss_and_unhurt_player_does_not_bleed():
    vitals = PlayableVitals()
    assert vitals.step_bleeding(10., 8.) == 0
    vitals.take_hit(.12)
    vitals.treat_wound()
    health, stamina = vitals.health, vitals.stamina
    assert vitals.step_bleeding(10., 8.) == 0
    assert (vitals.health, vitals.stamina) == (health, stamina)


def test_track_visible_intermediate_and_share_old_evidence():
    grid = np.zeros((15, 15), dtype=np.int32)
    grid[:, 7] = 1
    marks = [BloodMark(1, np.array([4., 4.])), BloodMark(2, np.array([6., 4.])),
             BloodMark(3, np.array([8., 4.]))]
    assert observe_blood(marks, grid, np.array([3., 4.]), 0, 'G1').mark_id == 2
    assert observe_blood(marks, grid, np.array([3., 4.]), 0, 'G2') is None
    assert not marks[2].discovered_by
    assert observe_blood(marks, grid, np.array([8., 3.]), np.pi / 2, 'G1', 2).mark_id == 3
    assert observe_blood(marks, grid, np.array([8., 3.]), np.pi / 2, 'G2') is None


def test_player_wound_deposits_blood_and_medkit_stops_it(game):
    game.env.player_pos = np.array([4., 4.], dtype=np.float32)
    game.player_vitals.take_hit(.12)
    for _ in range(20):
        game._apply_player(InputState())
    assert game.blood and game.player_vitals.health < .88
    assert game.state()['player']['bleeding'] > 0
    game.pickups = [dict(id='med', kind='MEDKIT', amount=1, source='map', pos=game.env.player_pos.copy())]
    before_health = game.player_vitals.health
    game._collect_pickups()
    assert game.player_vitals.health == before_health
    assert game.player_vitals.bleeding > 0 and game.equipment.medkits == 1
    game._apply_player(InputState(use_medkit=True))
    for _ in range(119):
        game._apply_player(InputState())
    assert game.player_vitals.bleeding == 0 and game.equipment.medkits == 0
    before = len(game.blood)
    game._apply_player(InputState())
    assert len(game.blood) == before


def test_contaminated_blood_does_not_create_a_wound(game):
    game.env.player_pos = np.array([4., 4.], dtype=np.float32)
    game.bloody_timer = 10
    game._leave_blood_trail(np.array([3., 4.], dtype=np.float32))
    assert game.blood and game.player_vitals.health == 1
    assert game.player_vitals.bleeding == 0


def test_guard_follows_lost_player_blood_and_does_not_reinvestigate(game):
    guard = game.aux_guards[0]
    game.env.grid[:] = 0
    game.env.player_pos = np.array([40., 20.], dtype=np.float32)
    guard.pos = np.array([4., 4.], dtype=np.float32)
    guard.angle = 0
    guard.state = 'CHASE'
    guard.last_seen = np.array([5., 4.], dtype=np.float32)
    game.blood = [BloodMark(100, np.array([6., 4.], dtype=np.float32))]
    game._step_aux_guard(guard)
    assert guard.state == 'TRACK BLOOD'
    assert guard.evidence_id == 100
    assert guard.pos[0] > 4
    assert not game._detect_evidence(game.aux_guards[1])
    guard.pos = guard.target.copy()
    game._step_aux_guard(guard)
    assert guard.state == 'SEARCH'
    assert not game._detect_evidence(guard)


def test_primary_guard_assigns_blood_route(game):
    game.env.grid[:] = 0
    game.env.guard_pos = np.array([4., 4.], dtype=np.float32)
    game.env.guard_angle = 0
    game.env.sense.visual = False
    game.blood = [BloodMark(100, np.array([6., 4.], dtype=np.float32))]
    game._primary_evidence()
    assert game._lead_patrol.state == 'TRACK BLOOD'
    assert np.array_equal(game._lead_patrol.target, game.blood[0].pos)


def test_standalone_swat_hit_bleeds_tracks_and_can_be_treated():
    with patch('shadow_ai.interactive_runtime.PPO.load', return_value=HoldPolicy()):
        runtime = InteractiveSwatRuntime(Path('fake.zip'))
    runtime.spawn_safe = False
    runtime.spawn_grace = 0
    runtime.core.map.grid[:] = 0
    runtime.core.player_pos = np.array([8., 10.], dtype=np.float32)
    from types import SimpleNamespace
    runtime.rng = SimpleNamespace(normal=lambda *args, **kwargs: 0.)
    shooter = next(iter(runtime.core.units.values()))
    for other in runtime.core.units.values():
        other.alive = other is shooter
    shooter.pos, shooter.angle, shooter.weapon = np.array([4.,10.],dtype=np.float32), 0., 'PISTOL'
    runtime._world_after_ai()
    assert runtime.player_vitals.bleeding > 0
    for _ in range(20):
        runtime._apply_player(InputState())
    assert runtime.blood and runtime.core.player_health < .88
    unit = next(iter(runtime.core.units.values()))
    unit.local_visual = False
    unit.pos = np.array([4., 10.], dtype=np.float32)
    unit.angle = 0
    assert runtime._step_swat_blood(unit.uid)
    assert runtime.actions[unit.uid] == 'TRACK BLOOD'
    assert runtime.state()['blood']
    runtime.pickups = [dict(id='med', kind='MEDKIT', amount=1, source='map', pos=runtime.core.player_pos.copy())]
    before_health = runtime.core.player_health
    runtime._collect_swat_pickups()
    assert runtime.core.player_health == before_health
    assert runtime.player_vitals.bleeding > 0 and runtime.equipment.medkits == 1
    runtime._apply_player(InputState(use_medkit=True))
    for _ in range(119):
        runtime._apply_player(InputState())
    assert runtime.player_vitals.bleeding == 0 and runtime.equipment.medkits == 0
    assert runtime.core.player_health == runtime.player_vitals.health


def test_obscured_evidence_stays_unknown_until_visible():
    grid = np.zeros((10, 10), dtype=np.int32)
    marks = [BloodMark(1, np.array([4., 4.]))]
    pos = np.array([2., 4.])
    assert observe_blood(marks, grid, pos, 0, 'G1', obscured=lambda a, b: True) is None
    assert not marks[0].discovered_by
    assert observe_blood(marks, grid, pos, 0, 'G1').mark_id == 1


def test_untreated_wound_exhausts_vigour_and_can_be_fatal():
    vitals = PlayableVitals(health=.04, stamina=.01, bleeding=1.)
    vitals.step_bleeding(1., 8.)
    assert vitals.exhausted and vitals.stamina == 0
    vitals.step_bleeding(1., 8.)
    assert vitals.health == 0
    assert vitals.step_bleeding(1., 8.) == 0


def test_bandage_takes_one_second_and_never_restores_health():
    vitals = PlayableVitals(health=.5, bleeding=.3)
    assert vitals.start_treatment('BANDAGE')
    vitals.step_treatment(.9)
    assert vitals.bleeding == .3 and vitals.health == .5
    vitals.step_treatment(.1)
    assert vitals.bleeding == 0 and vitals.health == .5
    vitals.step_treatment(120)
    assert vitals.health == .5 and vitals.regen_remaining == 0


def test_medkit_heals_ten_percent_in_four_seconds_and_twenty_five_in_two_minutes():
    vitals = PlayableVitals(health=.4, bleeding=.3)
    assert vitals.start_treatment('MEDKIT')
    vitals.step_treatment(2.)
    assert vitals.health == pytest.approx(.45)
    assert vitals.bleeding > 0 and vitals.treatment_remaining == 2
    vitals.step_treatment(2.)
    assert vitals.health == pytest.approx(.5)
    assert vitals.bleeding == 0 and vitals.regen_remaining == .25
    vitals.step_treatment(60)
    assert vitals.health == pytest.approx(.625)
    vitals.step_treatment(60)
    assert vitals.health == pytest.approx(.75)
    vitals.step_treatment(60)
    assert vitals.health == pytest.approx(.75)


def test_medkit_health_is_capped_and_damage_interrupts_application():
    vitals = PlayableVitals(health=.95)
    assert vitals.start_treatment('MEDKIT')
    vitals.step_treatment(4)
    vitals.step_treatment(120)
    assert vitals.health == 1
    wounded = PlayableVitals(health=.5, bleeding=.3)
    assert wounded.start_treatment('MEDKIT')
    wounded.step_treatment(1)
    wounded.take_hit(.12)
    assert wounded.treatment is None and wounded.bleeding > 0
    assert wounded.regen_remaining == 0


def test_bandage_collection_and_application_consumes_only_one_item(game):
    game.player_vitals.take_hit(.12)
    game.pickups = [dict(id='bandage', kind='BANDAGE', amount=2, source='MAP', pos=game.env.player_pos.copy())]
    game._collect_pickups()
    game._apply_player(InputState(use_bandage=True))
    assert game.equipment.bandages == 1
    for _ in range(29):
        game._apply_player(InputState(use_bandage=True))
    assert game.player_vitals.bleeding == 0 and game.equipment.bandages == 1
    assert game.player_vitals.health < .88
    assert game.player_vitals.regen_remaining == 0


def test_movement_interrupts_application_and_empty_inventory_cannot_treat(game):
    game.player_vitals.take_hit(.12)
    game._apply_player(InputState(use_bandage=True))
    assert game.player_vitals.treatment is None
    game.equipment.medkits = 1
    game._apply_player(InputState(use_medkit=True))
    assert game.player_vitals.treatment == 'MEDKIT'
    game._apply_player(InputState(right=True))
    assert game.player_vitals.treatment is None and game.equipment.medkits == 0
    assert game.player_vitals.bleeding > 0 and game.player_vitals.regen_remaining == 0


def test_medical_supplies_exist_only_on_map_and_reset_empty_inventory(game):
    assert any(i['kind'] == 'BANDAGE' for i in game.pickups)
    assert any(i['kind'] == 'MEDKIT' for i in game.pickups)
    guard = game.aux_guards[0]
    game._down_guard(guard.guard_id, True)
    assert not any(i['kind'] == 'MEDKIT' and i['source'] != 'MAP' for i in game.pickups)
    game.equipment.bandages = game.equipment.medkits = 2
    game.player_vitals.start_treatment('MEDKIT')
    game.reset('routine')
    assert game.equipment.bandages == game.equipment.medkits == 0
    assert game.player_vitals.treatment is None and game.player_vitals.regen_remaining == 0


def test_medical_input_pulses_survive_merging_and_are_cleared():
    previous = InputState.from_dict(dict(use_bandage=True, use_medkit=True))
    current = InputState()
    current.merge_transients(previous)
    assert current.use_bandage and current.use_medkit
    current.clear_transients()
    assert not current.use_bandage and not current.use_medkit
