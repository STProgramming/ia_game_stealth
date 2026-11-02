import math
import numpy as np
import pytest
from shadow_ai.soliton_security import BloodMark, SecurityGuard, BLOOD_FOOTPRINT_SECONDS, BLOOD_WOUND_SECONDS
from shadow_ai.playable_blood import observe_blood, wound_radius
from shadow_ai.interactive_runtime import InputState
from test_interactive_tactics import game
from test_playable_actions import swat


def test_footprints_expire_earlier_than_wound_stains_and_stop_being_detectable(game):
    game.env.grid[:]=0
    footprint=BloodMark(1,np.array([6.,4.]),kind='FOOTPRINT',lifetime=BLOOD_FOOTPRINT_SECONDS)
    wound=BloodMark(2,np.array([7.,4.]),lifetime=BLOOD_WOUND_SECONDS)
    game.blood=[footprint,wound]
    for _ in range(361): game._age_blood()
    assert footprint not in game.blood and wound in game.blood
    assert not footprint.active and wound.active
    assert observe_blood([footprint],game.env.grid,np.array([4.,4.]),0,'S0',include_shared=True) is None
    assert BLOOD_FOOTPRINT_SECONDS < BLOOD_WOUND_SECONDS


def test_wound_stain_area_scales_with_actual_damage(game):
    game.env.grid[:]=0
    game.env.player_pos=np.array([10.,10.],dtype=np.float32)
    game.player_vitals.take_hit(.1)
    game._deposit_player_injury_blood()
    small=game.blood[-1]
    game.player_vitals.take_hit(.4)
    game._deposit_player_injury_blood()
    large=game.blood[-1]
    assert small.damage == pytest.approx(.1) and large.damage == pytest.approx(.4)
    assert large.radius**2 / small.radius**2 == pytest.approx(4.)
    assert small.kind == large.kind == 'WOUND'
    assert large.snapshot()['radius'] == large.radius


def test_damage_marks_are_consumed_once_and_cannot_exceed_health_actually_lost(game):
    game.player_vitals.health=.15
    game.player_vitals.take_hit(.8)
    game._deposit_player_injury_blood()
    assert game.blood[-1].damage == pytest.approx(.15)
    count=len(game.blood)
    game._deposit_player_injury_blood()
    assert len(game.blood)==count


def test_contaminated_shoes_make_short_lived_footprints_without_a_new_wound(game):
    game.env.grid[:]=0
    game.blood=[]
    game.bloody_timer=10.
    game._leave_footprints(np.array([4.,4.]),np.array([5.,4.]))
    mark=game.blood[-1]
    assert mark.kind=='FOOTPRINT' and mark.lifetime==BLOOD_FOOTPRINT_SECONDS
    assert mark.damage==0 and game.player_vitals.bleeding==0
    assert math.isfinite(mark.radius)


def test_deployed_swat_follows_visible_blood_previously_reported_by_a_regular_guard(game):
    game.env.grid[:]=0
    game.lead_alive=False; game.aux_guards=[]
    game.env.player_pos=np.array([55.,35.],dtype=np.float32)
    unit=SecurityGuard('S0',np.array([4.,4.],dtype=np.float32),0,((4,4),))
    unit.state='RESPOND'
    game.swat_units=[('ASSAULT',unit)]
    mark=BloodMark(100,np.array([6.,4.],dtype=np.float32),discovered_by={'G1'})
    game.blood=[mark]
    game.team_contact=np.array([24.,6.],dtype=np.float32); game.team_contact_age=20.
    game._step_guard(unit)
    assert unit.state=='TRACK BLOOD'
    assert unit.evidence_id==100 and unit.pos[0]>4
    assert np.array_equal(unit.target,mark.pos)
    assert 'S0' in mark.discovered_by
    assert np.array_equal(game.team_contact,np.array([24.,6.]))  # No invented visual report.


def test_standalone_swat_prefers_a_fresh_visual_report_then_follows_shared_blood_when_lost(swat):
    swat.core.map.grid[:]=0
    unit=next(iter(swat.core.units.values()))
    unit.pos=np.array([4.,4.],dtype=np.float32); unit.angle=0; unit.local_visual=False
    mark=BloodMark(100,np.array([6.,4.],dtype=np.float32),discovered_by={'G1'})
    swat.blood=[mark]
    swat.core.blackboard.contact_pos=np.array([20.,5.],dtype=np.float32)
    swat.core.blackboard.contact_age=0.
    assert not swat._step_swat_blood(unit.uid)
    swat.core.blackboard.contact_age=5.
    assert swat._step_swat_blood(unit.uid)
    assert swat.actions[unit.uid]=='TRACK BLOOD' and unit.uid in mark.discovered_by


def test_swat_does_not_choose_hidden_newer_blood_beyond_a_wall(swat):
    swat.core.map.grid[:]=0; swat.core.map.grid[:,7]=1
    swat.core.blackboard.contact_pos=None
    unit=next(iter(swat.core.units.values()))
    unit.pos=np.array([4.,4.],dtype=np.float32); unit.angle=0; unit.local_visual=False
    near=BloodMark(100,np.array([6.,4.],dtype=np.float32))
    hidden=BloodMark(101,np.array([9.,4.],dtype=np.float32))
    swat.blood=[near,hidden]
    swat._step_swat_blood(unit.uid)
    assert swat._blood_progress[unit.uid]==100
    assert not hidden.discovered_by
