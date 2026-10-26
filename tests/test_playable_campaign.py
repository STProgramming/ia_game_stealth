import numpy as np
import pytest
from collections import deque
from shadow_ai.interactive_runtime import InputState
from shadow_ai.playable_campaign import MISSIONS,mission_layout
from test_interactive_tactics import game


@pytest.mark.parametrize('level',[1,2,3])
def test_each_mission_has_distinct_reachable_objectives_and_supplies(game,level):
    game.level = level
    game.reset('routine')
    assert len(game.state()['map']['intel']) == len(MISSIONS[level]['indices'])
    assert game.state()['mission']['title'] == MISSIONS[level]['title']
    nav = game.navigator
    seen = {nav.cell(game.env.player_pos)}
    queue = deque(seen)
    while queue:
        x,y = queue.popleft()
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            pt=(x+dx,y+dy)
            if pt not in seen and nav.free(*pt):
                seen.add(pt);queue.append(pt)
    for item in game.pickups:
        assert nav.cell(item['pos']) in seen
    for index in MISSIONS[level]['indices']:
        assert nav.cell(game.env._intel[index]) in seen
    assert nav.cell(game.env._exit) in seen


def test_three_mission_geometries_and_security_increase(game):
    grids,counts=[],[]
    for level in (1,2,3):
        game.level=level;game.reset('routine')
        grids.append(game.env.grid.copy())
        assert all(game.env._free(*game.env._cell(np.array(point))) for point in game._lead_patrol.patrol)
        counts.append((len(game.aux_guards),len(game.cameras)))
    assert not np.array_equal(grids[0],grids[1]) and not np.array_equal(grids[1],grids[2])
    assert counts == [(2,2),(4,4),(7,6)]


def test_security_card_is_acquired_in_office_and_carried_to_rd(game):
    game.level=1;game.reset('routine')
    game.env.player_pos=game.env._intel[0].copy()
    game._apply_player(InputState(interact=True))
    assert game.equipment.security_card and 'SECURITY_CARD' in game.campaign_items
    game.level=2;game.reset('routine')
    assert game.equipment.security_card
    assert game.env.player_has_intel.sum()==0


def test_rd_requires_both_proofs_before_exit(game):
    game.level=2;game.reset('routine')
    game.spawn_safe=False;game.spawn_grace=0;game.lead_alive=False;game.aux_guards=[];game.cameras=[]
    game.env.player_pos=game.env._exit.copy()
    game.env.player_has_intel[0]=1
    game.tick(InputState())
    assert game.status=='RUNNING'
    game.env.player_has_intel[1]=1
    game.tick(InputState())
    assert game.status=='ESCAPED'


def test_unprotected_pathogen_contact_kills_in_quarter_second(game):
    game.level=3;game.reset('routine')
    game.spawn_safe=False;game.spawn_grace=0;game.lead_alive=False;game.aux_guards=[];game.cameras=[]
    game.env.player_pos=np.array([44.,22.],dtype=np.float32)
    for _ in range(8):
        game.tick(InputState())
    assert game.player_vitals.health==0 and game.status=='NEUTRALIZED'
    assert not game.env.player_has_intel[2]


def test_lab_protection_and_specific_vial_are_both_required(game):
    game.level=3;game.reset('routine')
    for kind in ('HAZMAT','SAMPLE_VIAL'):
        item=next(p for p in game.pickups if p['kind']==kind)
        game.env.player_pos=item['pos'].copy()
        game._collect_pickups()
        assert not game.pathogen_exposure and game.player_vitals.health==1
    assert game.equipment.hazmat and game.equipment.sample_vial
    game.env.player_pos=game.env._intel[2].copy()
    game.equipment.sample_vial=False
    game._apply_player(InputState(interact=True))
    assert not game.env.player_has_intel[2] and game.player_vitals.health==1
    game._apply_player(InputState())
    game.equipment.sample_vial=True
    game._apply_player(InputState(interact=True))
    assert game.env.player_has_intel[2] and not game.equipment.sample_vial
    assert game.player_vitals.health==1
