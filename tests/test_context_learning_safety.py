import numpy as np
import pytest

from shadow_ai.tactical_context import ContextAction, apply_context, observe_context
from test_playable_actions import swat


def start_standalone_treatment(runtime):
    uid, patient = next(iter(runtime.core.units.items()))
    station = next(item for item in runtime.pickups if item.get('fixed'))
    patient.pos = station['pos'].copy()
    patient.local_visual = False
    patient.health = runtime._unit_vitals[uid].health = .5
    runtime._unit_vitals[uid].bleeding = .5
    assert apply_context(runtime, uid, patient, ContextAction.SEEK_CARE)
    assert uid in runtime._context_medical_jobs
    assert runtime._unit_vitals[uid].treatment == 'MEDKIT'
    return uid, patient, station


@pytest.mark.parametrize('lethal', [False, True])
def test_downed_standalone_patient_no_longer_requests_ally_cover(swat, lethal):
    uid, patient, station = start_standalone_treatment(swat)
    ally_uid, ally = next((key, unit) for key, unit in swat.core.units.items() if key != uid)
    assert observe_context(swat, ally_uid, ally)[18] == 1.
    swat._down_swat_player_target(patient, lethal)
    assert uid not in swat._context_medical_jobs
    assert swat._unit_vitals[uid].treatment is None
    assert observe_context(swat, ally_uid, ally)[18] == 0.
    assert not apply_context(swat, ally_uid, ally, ContextAction.COVER_ALLY)
    assert station['id'] in swat.medical_watch.consumed
    assert not any(item['id'] == station['id'] for item in swat.pickups)


def test_incoming_fire_cancels_standalone_medical_job_before_cover(swat):
    uid, patient, station = start_standalone_treatment(swat)
    source = patient.pos - np.array([4., 0.], dtype=np.float32)
    swat._react_to_incoming_hit(uid, patient, swat._unit_vitals[uid], source, eligible=True)
    assert uid not in swat._context_medical_jobs
    assert swat._unit_vitals[uid].treatment is None
    assert patient._incoming_cover is not None
    assert any(entry['text'] == 'SHOOT FIRE, SHOOT FIRE' for entry in swat.radio_log.entries)
    assert station['id'] in swat.medical_watch.consumed
    assert not any(item['id'] == station['id'] for item in swat.pickups)
