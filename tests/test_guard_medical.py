import numpy as np
import pytest

from shadow_ai.interactive_runtime import InputState
from shadow_ai.playable_threats import MedicalSurveillance
from shadow_ai.soliton_security import SecurityGuard
from test_interactive_tactics import game


def prepare(game, with_medic=True):
    game.env.grid[:] = 0
    game.lead_alive = False
    game.env.player_pos = np.array([55., 35.], dtype=np.float32)
    patient = SecurityGuard('G8', np.array([8., 8.], dtype=np.float32), 0., ((8., 8.),))
    patient.vitals.take_hit(.3)
    medic = SecurityGuard('G9', np.array([12., 8.], dtype=np.float32), 0., ((12., 8.),),
                          is_medic=True, medkits=2)
    game.aux_guards = [patient, medic] if with_medic else [patient]
    station = dict(id='shared-kit', kind='MEDKIT', amount=1, fixed=True,
                   pos=np.array([8., 8.], dtype=np.float32), source='MAP')
    game.pickups = [station]
    game.medical_watch = MedicalSurveillance(game.pickups)
    return patient, medic, station


@pytest.mark.parametrize('level', [1, 2, 3])
def test_every_mission_has_a_real_medic_with_finite_kits(game, level):
    game.level = level
    game.reset('routine')
    medics = [guard for guard in game.aux_guards if guard.is_medic]
    assert len(medics) == 1 and medics[0].medkits == 2
    snapshot = next(guard for guard in game.state()['guards'] if guard['id'] == medics[0].guard_id)
    assert snapshot['role'] == 'MEDIC' and snapshot['medkits'] == 2


def test_wounded_guard_requests_medic_who_physically_arrives_and_consumes_one_kit(game):
    patient, medic, station = prepare(game)
    assert game._step_guard_medical(patient, False)
    assert patient.state == 'WAIT MEDIC'
    assert patient.vitals.treatment is None and medic.medkits == 2
    assert any('assistenza medica' in entry['text'] for entry in game.radio_log.entries)
    start = medic.pos.copy()
    for _ in range(600):
        game._step_guard_medical(patient, False)
        game._step_guard_medical(medic, False)
        game._step_guard_recovery()
        if not game._medical_jobs:
            break
    assert np.linalg.norm(medic.pos - start) > 1.
    assert np.linalg.norm(medic.pos - patient.pos) <= 1.2
    assert patient.vitals.health > .79 and patient.vitals.bleeding == 0
    assert patient.vitals.regen_remaining > 0
    assert medic.medkits == 1 and station in game.pickups
    for _ in range(5):
        game._step_guard_medical(patient, False)
    assert medic.medkits == 1


def test_unreachable_medic_cannot_treat_through_a_wall(game):
    patient, medic, _ = prepare(game)
    medic.pos = np.array([10., 8.], dtype=np.float32)
    game.env.grid[:, 9] = 1
    game.pickups = []
    assert not game._step_guard_medical(patient, False)
    assert patient.vitals.health == pytest.approx(.7)
    assert patient.vitals.treatment is None and medic.medkits == 2


def test_shared_station_is_consumed_once_and_removed_from_surveillance(game):
    patient, _, station = prepare(game, with_medic=False)
    game.medical_watch.hit('G8')
    assert game.medical_watch.goal('G8', patient.pos, False, game.navigator.path_length) is not None
    assert game._step_guard_medical(patient, False)
    assert patient.vitals.treatment == 'MEDKIT'
    assert not game.pickups and station['id'] in game.medical_watch.consumed
    assert game.medical_watch.goal('G8', patient.pos, False, game.navigator.path_length) is None
    game.env.player_pos = station['pos'].copy()
    game.player_vitals.health = .5
    for _ in range(150):
        game._apply_player(InputState(interact_held=True))
    assert game.player_vitals.treatment is None and game.player_vitals.health == .5
    for _ in range(125):
        game._step_guard_medical(patient, False)
    assert patient.vitals.health >= .8 - 1e-6
    assert patient.vitals.bleeding == 0


def test_two_guards_cannot_consume_the_same_station(game):
    first, second, _ = prepare(game, with_medic=False)
    second.is_medic = False
    second.medkits = 0
    second.pos = first.pos.copy()
    second.vitals.take_hit(.3)
    game.aux_guards.append(second)
    assert game._step_guard_medical(first, False)
    assert not game._step_guard_medical(second, False)
    assert second.vitals.treatment is None


def test_player_station_reservation_excludes_guards_until_cancelled(game):
    patient, _, station = prepare(game, with_medic=False)
    game.env.player_pos = station['pos'].copy()
    game.player_vitals.health = .5
    for _ in range(12):
        game._apply_player(InputState(interact_held=True))
    assert station.get('reserved_by') == 'PLAYER'
    assert not game._step_guard_medical(patient, False)
    game._apply_player(InputState())
    assert not station.get('reserved_by')
    game._medical_retry.clear()
    assert game._step_guard_medical(patient, False)


def test_completed_player_station_cannot_be_watched_or_used_again(game):
    patient, _, station = prepare(game, with_medic=False)
    game.env.player_pos = station['pos'].copy()
    game.player_vitals.health = .5
    game.medical_watch.hit(patient.guard_id)
    for _ in range(140):
        game._apply_player(InputState(interact_held=True))
    assert station['id'] in game.medical_watch.consumed and not game.pickups
    assert game.medical_watch.goal(patient.guard_id, patient.pos, False, game.navigator.path_length) is None
    assert not game._step_guard_medical(patient, False)


@pytest.mark.parametrize('lethal', [False, True])
def test_medic_drops_remaining_kits_once_on_stun_or_death(game, lethal):
    _, medic, _ = prepare(game)
    game._down_guard(medic.guard_id, lethal)
    kits = [item for item in game.pickups if item['kind'] == 'MEDKIT' and item['source'] == medic.guard_id]
    assert len(kits) == 1 and kits[0]['amount'] == 2 and not kits[0].get('fixed')
    game._drop_guard_equipment(medic.guard_id, medic.pos)
    assert len([item for item in game.pickups if item['source'] == medic.guard_id and item['kind'] == 'MEDKIT']) == 1
    game.env.player_pos = medic.pos.copy()
    game._collect_pickups()
    assert game.equipment.medkits == 2 and medic.medkits == 0


def test_spent_medic_kit_is_not_dropped_or_reused(game):
    patient, medic, _ = prepare(game)
    medic.medkits = 1
    medic.pos = patient.pos.copy()
    game._step_guard_medical(patient, False)
    assert medic.medkits == 0
    game._down_guard(medic.guard_id, True)
    assert not any(item['kind'] == 'MEDKIT' and item['source'] == medic.guard_id for item in game.pickups)
    assert patient.vitals.treatment is None and not game._medical_jobs


def test_combat_interrupts_medical_assistance_without_refunding_spent_kit(game):
    patient, medic, _ = prepare(game)
    medic.pos = patient.pos.copy()
    game._step_guard_medical(patient, False)
    assert medic.medkits == 1 and patient.vitals.treatment == 'MEDKIT'
    assert not game._step_guard_medical(medic, True)
    assert not game._medical_jobs and patient.vitals.treatment is None
    assert medic.medkits == 1


def test_runtime_guard_can_self_treat_instead_of_only_watching_station(game):
    patient, _, _ = prepare(game, with_medic=False)
    game._step_guard(patient)
    assert patient.state == 'TREATMENT' and patient.vitals.treatment == 'MEDKIT'
    game._step_guard_weapon(patient, True)
    assert patient.aim_timer == 0
