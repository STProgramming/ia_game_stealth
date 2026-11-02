import pytest
from shadow_ai.security_runtime import SecurityCycle, RadioLog, RadioDialogue
from shadow_ai.soliton_security import SecurityGuard
from test_interactive_tactics import game
import numpy as np


def test_security_countdowns_and_loss_of_visual_contact():
    cycle=SecurityCycle()
    cycle.contact()
    cycle.step(5,visible=True)
    assert cycle.remaining==99 and cycle.rate==0
    cycle.step(1)
    assert cycle.remaining==95 and cycle.rate==4
    cycle.step(23.75)
    assert cycle.phase=='SEARCH' and cycle.remaining==99
    cycle.step(99)
    assert cycle.phase=='SUSPICIOUS' and cycle.remaining==180
    cycle.step(179)
    assert cycle.phase=='SUSPICIOUS' and cycle.remaining==1
    cycle.step(1)
    assert cycle.phase=='CALM'


def test_reacquisition_resets_alert_and_new_evidence_extends_caution():
    cycle=SecurityCycle()
    cycle.caution(); cycle.step(150)
    cycle.caution()
    assert cycle.remaining==180
    cycle.contact(); cycle.step(10)
    cycle.contact()
    assert cycle.phase=='CONTACT' and cycle.remaining==99


def test_caution_actually_ends_without_old_flags_restarting_it(game):
    game.security_compromised=True
    game._update_facility_alert()
    assert game.facility_alert=='SUSPICIOUS'
    game.security_cycle.remaining=.1
    game.game_dt=.1
    for guard in game._active_guards(): guard.seeing_player=False
    game._step_security_cycle()
    for _ in range(10): game._update_facility_alert()
    assert game.facility_alert=='CALM' and not game.security_compromised
    assert game.env.personal_suspicion==0


def test_swat_has_red_cones_on_arrival_without_gunfire(game):
    swat=SecurityGuard('S9',np.array([10.,10.]),0.,((10,10),))
    swat.state='RESPOND'
    game.team_contact_source='G1'
    assert game._guard_state(swat)['combat_alert']
    assert not game._guard_state(game.aux_guards[0])['combat_alert']


def test_radio_preserves_every_same_frame_transmission_after_bubbles_expire():
    log=RadioLog(); dialogue=RadioDialogue(log)
    for text in ['Ho sentito un rumore','Ho visto qualcosa','Intruso avvistato','Richiedo SWAT']:
        dialogue['G1']=text
    dialogue.clear()
    assert len(log.entries)==4
    assert [item['text'] for item in log.snapshot()]==['Ho visto qualcosa','Intruso avvistato','Richiedo SWAT']
    log.time=1
    dialogue['G1']='Richiedo SWAT'
    assert len(log.entries)==4
    log.time=10
    dialogue['G1']='Richiedo SWAT'
    assert len(log.entries)==5


def test_radio_reception_and_positive_contact_are_both_in_log(game):
    game._publish_contact(np.array([10.,10.]),'G1')
    assert any('Contatto confermato' in message['text'] for message in game.radio_log.entries)
    assert game.state()['security']['countdown']['remaining']==99
