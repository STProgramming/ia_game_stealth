from pathlib import Path
from unittest.mock import patch
import math
import numpy as np
import pytest

from shadow_ai.interactive_runtime import InputState, InteractiveGuardRuntime, InteractiveSwatRuntime
from shadow_ai.playable_equipment import Equipment
from shadow_ai.playable_firearms import Firearm, acoustic_profile
from shadow_ai.soliton_security import SecurityGuard
from test_interactive_tactics import game, HoldPolicy


@pytest.fixture(params=['guard', 'swat'])
def runtime(request):
    with patch('shadow_ai.interactive_runtime.PPO.load', return_value=HoldPolicy()):
        r = (InteractiveGuardRuntime if request.param == 'guard' else InteractiveSwatRuntime)(Path('fake.zip'))
    r.spawn_safe, r.spawn_grace = False, 0.
    world, grid = r._combat_world()
    grid[:] = 0
    world.player_pos = np.array([10., 10.], dtype=np.float32)
    if hasattr(r, 'env'):
        r.lead_alive = False
        r.aux_guards = []
        r.cameras = []
    else:
        for unit in r.core.units.values():
            unit.alive = False
        r.core.blackboard.contact_pos = None
    r.equipment = Equipment(weapon='PISTOL_SUPPRESSED', ammo=7, selected='PISTOL_SUPPRESSED', weapons={'PISTOL_SUPPRESSED'})
    r.equipment.gun().spares = [15, 15]
    return r


def action(r, **kwargs):
    call = r._player_equipment_action if hasattr(r, 'env') else r._swat_player_equipment
    call(InputState(aim_x=20, aim_y=10, **kwargs))


def wait_reload(r):
    for _ in range(100):
        action(r)
        if not r.equipment.gun().reload_mode:
            break


def test_chamber_is_independent_and_feeds_one_round_at_a_time():
    gun = Firearm.loaded('PISTOL', 16)
    assert gun.magazine == 15 and gun.chamber
    for remaining in range(15, -1, -1):
        assert gun.fire()
        assert gun.rounds == remaining
    assert not gun.fire()
    assert gun.magazine == 0 and not gun.chamber


def test_tactical_reload_keeps_the_partial_magazine_and_chamber(runtime):
    action(runtime, reload=True)
    gun = runtime.equipment.gun()
    assert gun.reload_mode == 'TACTICAL' and gun.chamber and gun.magazine == 0
    assert not [i for i in runtime.pickups if i['id'].startswith('discarded-mag')]
    before = runtime.equipment.ammo
    action(runtime, fire=True)
    assert runtime.equipment.ammo == before and not runtime.shots
    wait_reload(runtime)
    assert gun.magazine == 15 and gun.chamber
    assert sorted(gun.spares) == [6, 15]
    assert gun.rounds + sum(gun.spares) == 37


def test_double_reload_upgrades_once_and_discards_remaining_ammo(runtime):
    action(runtime, reload=True)
    action(runtime, reload=True, fast_reload=True)
    action(runtime, fast_reload=True)
    drops = [i for i in runtime.pickups if i['id'].startswith('discarded-mag')]
    assert len(drops) == 1 and drops[0]['lost_rounds'] == 6
    assert any(e.source == 'MAGAZINE DROP' and e.radius == 3 for e in runtime.noise_events)
    assert runtime.equipment.gun().reload_remaining < 1.05
    wait_reload(runtime)
    gun = runtime.equipment.gun()
    assert gun.rounds + sum(gun.spares) == 31
    assert 6 not in gun.spares


def test_empty_magazine_is_saved_as_a_diversion_and_empty_chamber_is_racked(runtime):
    runtime.equipment.ammo = 0
    action(runtime, reload=True)
    wait_reload(runtime)
    assert runtime.equipment.magazines == 1
    assert runtime.equipment.gun().magazine == 14
    assert runtime.equipment.gun().chamber
    assert runtime.equipment.select('MAGAZINE')
    assert runtime.equipment.take_diversion()
    assert runtime.equipment.magazines == 0


def test_discovered_player_only_performs_fast_reloads(runtime):
    if hasattr(runtime, 'env'):
        runtime.facility_alert = 'CONTACT'
    else:
        runtime._equipment_alert = 'CONTACT'
    action(runtime, reload=True)
    assert runtime.equipment.gun().reload_mode == 'FAST'
    assert len([i for i in runtime.pickups if i['id'].startswith('discarded-mag')]) == 1


def test_late_contact_upgrades_an_existing_tactical_reload(runtime):
    action(runtime, reload=True)
    if hasattr(runtime, 'env'):
        runtime.facility_alert = 'CONTACT'
    else:
        runtime._equipment_alert = 'CONTACT'
    action(runtime)
    assert runtime.equipment.gun().reload_mode == 'FAST'


def test_reload_without_spares_never_creates_ammunition(runtime):
    runtime.equipment.gun().spares.clear()
    action(runtime, reload=True, fast_reload=True)
    assert runtime.equipment.ammo == 7
    assert not runtime.equipment.gun().reload_mode
    assert not [i for i in runtime.pickups if i['id'].startswith('discarded-mag')]


def test_weapon_switch_during_reload_preserves_rounds_without_finishing_it(runtime):
    runtime.equipment.collect('RIFLE', 10)
    runtime.equipment.select('PISTOL_SUPPRESSED')
    action(runtime, reload=True)
    runtime.equipment.select('RIFLE')
    assert runtime.equipment.ammo == 10
    runtime.equipment.select('PISTOL_SUPPRESSED')
    gun = runtime.equipment.gun()
    assert not gun.reload_mode and gun.rounds + sum(gun.spares) == 37


@pytest.mark.parametrize('kind', ['PISTOL', 'PISTOL_SUPPRESSED', 'PISTOL_SUBSONIC'])
def test_every_shot_ejects_a_case_but_only_suppressed_secondary_sounds_are_audible(runtime, kind):
    runtime.equipment = Equipment(weapon=kind, ammo=2, weapons={kind}, selected=kind)
    _, grid = runtime._combat_world()
    grid[:, 15] = 1
    action(runtime, fire=True)
    assert runtime.equipment.ammo == 1 and len(runtime.spent_cases) == 1
    assert not runtime.spent_cases[0]['landed']
    for _ in range(7):
        action(runtime)
    assert runtime.spent_cases[0]['landed']
    secondary = [s for s in runtime.noise_events if s.source in {'CASING DROP', 'BULLET IMPACT'}]
    assert {s.source for s in secondary} == {'CASING DROP', 'BULLET IMPACT'}
    assert all(s.masked == (kind == 'PISTOL') for s in secondary)
    if hasattr(runtime, 'env'):
        assert (runtime.team_contact is not None) == (kind == 'PISTOL')


def test_subsonic_gunshot_has_shorter_hearing_and_recognition_range():
    loud, quiet, sub = [acoustic_profile(k) for k in ['PISTOL', 'PISTOL_SUPPRESSED', 'PISTOL_SUBSONIC']]
    assert loud['radius'] > quiet['radius'] > sub['radius']
    assert quiet['recognition'] > sub['recognition'] > 0


def test_suppressed_gunfire_is_recognized_only_by_nearby_observers(runtime):
    action(runtime, fire=True)
    assert runtime._heard_firearm('close', np.array([11., 10.])) is not None
    assert runtime._heard_firearm('far', np.array([20., 10.])) is None
    if hasattr(runtime, 'env'):
        assert runtime.security_compromised and runtime.threat_level == 2
        assert runtime.team_contact is None
    else:
        assert runtime._equipment_alert == 'SUSPICIOUS'
        assert runtime.core.blackboard.contact_pos is None


def test_case_discovery_requires_sight_and_reports_evidence_position_without_telepathy(runtime):
    action(runtime, fire=True)
    for _ in range(7):
        action(runtime)
    case = runtime.spent_cases[0]
    if hasattr(runtime, 'env'):
        actor = SecurityGuard('G9', np.array([12., 10.], dtype=np.float32), 0., ((12,10),))
    else:
        actor = next(iter(runtime.core.units.values()))
        actor.pos, actor.angle, actor.blinded_timer = np.array([12.,10.]), 0., 0.
    runtime._observe_spent_cases('observer', actor)
    assert not case['discovered']  # Facing away.
    actor.angle = math.pi
    _, grid = runtime._combat_world()
    grid[:,11] = 1
    runtime._observe_spent_cases('observer', actor)
    assert not case['discovered']  # Wall blocks the view.
    grid[:,11] = 0
    world, _ = runtime._combat_world()
    world.player_pos = np.array([25., 15.], dtype=np.float32)
    runtime._observe_spent_cases('observer', actor)
    assert case['discovered']
    if hasattr(runtime, 'env'):
        assert runtime.facility_alert == 'SUSPICIOUS' and runtime.team_contact is None
    else:
        assert runtime._equipment_alert == 'SUSPICIOUS' and runtime.core.blackboard.contact_pos is None
        assert np.array_equal(runtime._search_anchor, case['pos'])


def test_reload_inputs_are_latched_between_network_ticks():
    pending = InputState.from_dict({'reload': True, 'fast_reload': True})
    latest = InputState.from_dict({})
    latest.merge_transients(pending)
    assert latest.reload and latest.fast_reload
    latest.clear_transients()
    assert not latest.reload and not latest.fast_reload


def test_enemy_empty_weapon_starts_a_timed_reload_instead_of_firing(runtime):
    if hasattr(runtime, 'env'):
        actor = SecurityGuard('G9', np.array([12., 10.], dtype=np.float32), math.pi, ((12,10),))
        vitals = actor.vitals
    else:
        actor = next(iter(runtime.core.units.values()))
        actor.pos = np.array([12.,10.]); actor.alive = True
        vitals = runtime._unit_vitals[actor.uid]
    actor.ammo = 0
    assert not runtime._enemy_shot('enemy', actor, vitals)
    gun = runtime._enemy_guns['enemy']
    assert gun.reload_mode == ('TACTICAL' if hasattr(runtime, 'env') else 'FAST') and gun.reload_remaining > 0
    assert runtime._combat_world()[0].player_health == 1 if not hasattr(runtime, 'env') else runtime.player_vitals.health == 1
