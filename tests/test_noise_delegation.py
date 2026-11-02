import numpy as np
import pytest
from shadow_ai import TacticalAction
from shadow_ai.soliton_security import SecurityGuard, NoiseEvent
from test_interactive_tactics import game


def caution_pair(game, lead=False):
    game.env.grid[:]=0
    game.env.player_pos=np.array([55.,35.],dtype=np.float32)
    game.lead_alive=lead
    if lead:
        a=game._lead_patrol
        game.env.guard_pos=np.array([4.,10.],dtype=np.float32)
        a.pos=game.env.guard_pos.copy()
    else:
        a=SecurityGuard('G1',np.array([4.,10.],dtype=np.float32),0,((4,10),(4,12)))
    b=SecurityGuard('G2',np.array([9.,10.],dtype=np.float32),0,((9,10),(9,12)))
    game.aux_guards=[b] if lead else [a,b]
    game.facility_alert='SUSPICIOUS'; game.security_compromised=True
    game.team_contact=None; game.cameras=[]
    origin=np.array([11.,10.],dtype=np.float32)
    return a,b,origin


@pytest.mark.parametrize('lead',[False,True])
def test_caution_guard_delegates_to_a_closer_colleague_and_waits_for_a_report(game,lead):
    a,b,origin=caution_pair(game,lead)
    game.noise_events=[NoiseEvent(900,origin,10.,.7,'BULLET IMPACT')]
    game._step_guard(a,TacticalAction.PATROL)
    assert a.state=='WAIT REPORT' and b.state=='INVESTIGATE FOR ALLY'
    task=next(iter(game._noise_tasks.values()))
    assert task['requester']==a.guard_id and task['investigator']==b.guard_id
    assert np.linalg.norm(task['origin']-origin)<1.5
    assert a.pos[0]==4
    assert 'fammi rapporto' in game.guard_dialogue[a.guard_id]


def test_colleague_physically_checks_the_location_then_reports_to_the_requester(game):
    a,b,origin=caution_pair(game)
    assert game._delegate_noise(a,origin,'NOISEMAKER')
    task=next(iter(game._noise_tasks.values()))
    for _ in range(1200):
        game._step_guard(a,TacticalAction.PATROL)
        game._step_guard(b,TacticalAction.PATROL)
        game._step_noise_radio()
        if task['status']=='REPORTED': break
    assert task['status']=='REPORTED'
    assert task['search'].index==len(task['search'].points)
    assert 'nessun intruso' in task['result']
    assert f'Radio {a.guard_id}' in game.guard_dialogue[b.guard_id]
    assert a.state!='WAIT REPORT' and not game._noise_requesters


@pytest.mark.parametrize('missing',['dead','unconscious','radio_disabled','stunned'])
def test_missing_reply_makes_the_requester_investigate_without_learning_the_colleagues_fate(game,missing):
    a,b,origin=caution_pair(game)
    game._delegate_noise(a,origin,'IMPACT')
    task=next(iter(game._noise_tasks.values()))
    task['deadline']=game._noise_clock+1.
    if missing=='dead': b.alive=False
    elif missing=='unconscious': b.unconscious=True
    elif missing=='stunned': b.vitals.hit_stun_remaining=5.
    else:
        b.radio_disabled=True
        task['status'],task['result'],task['report_wait']='REPORTING','nessun intruso',.1
    for _ in range(20): game._step_noise_radio()
    assert task['status'] in {'INVESTIGATING','REPORTING'} and a.state=='WAIT REPORT'
    for _ in range(15): game._step_noise_radio()
    assert task['status']=='OVERDUE'
    assert a.state=='SEARCH' and np.array_equal(a.target,origin)
    assert a.suspicion>=.9
    assert 'non risponde' in game.guard_dialogue[a.guard_id]
    assert game.team_contact is None


def test_lack_of_a_closer_reachable_available_guard_keeps_the_investigation_local(game):
    a,b,origin=caution_pair(game)
    b.pos=np.array([2.,10.],dtype=np.float32)
    assert not game._delegate_noise(a,origin,'IMPACT')
    b.pos=np.array([9.,10.],dtype=np.float32); b.state='CHASE'
    assert not game._delegate_noise(a,origin,'IMPACT')
    b.state='PATROL'; game.env.grid[:,8]=1
    b.pos=np.array([7.,10.],dtype=np.float32)
    assert not game._delegate_noise(a,origin,'IMPACT')


def test_delegation_is_caution_only_and_never_tracks_the_hidden_player(game):
    a,b,origin=caution_pair(game)
    for phase in ['CALM','CONTACT','SEARCH']:
        game.facility_alert=phase
        assert not game._delegate_noise(a,origin,'IMPACT')
    game.facility_alert='SUSPICIOUS'
    assert game._delegate_noise(a,origin,'IMPACT')
    task=next(iter(game._noise_tasks.values()))
    game.env.player_pos=np.array([30.,30.],dtype=np.float32)
    assert np.array_equal(task['origin'],origin)
    assert game.team_contact is None


def test_real_visual_contact_interrupts_waiting_and_reports_an_actual_discovery(game):
    a,b,origin=caution_pair(game)
    game._delegate_noise(a,origin,'IMPACT')
    task=next(iter(game._noise_tasks.values()))
    b.seeing_player=True; b.visual_confidence=1.; b.state='CHASE'
    assert not game._step_noise_order(b)
    assert task['status']=='REPORTED' and 'intruso avvistato' in task['result']
    assert b.state=='CHASE' and a.state!='WAIT REPORT'
