import numpy as np
import pytest
from unittest.mock import patch
from shadow_ai import TacticalAction
from shadow_ai.interactive_runtime import InputState
from shadow_ai.soliton_security import SecurityGuard
from test_interactive_tactics import game


@pytest.fixture(params=['G0','G1','G7'])
def sentry(game,request):
    game.env.grid[:] = 0
    game.cameras = []
    game.env.guard_pos = np.array([4.,4.],dtype=np.float32)
    game.env.guard_angle = 0
    if request.param == 'G0':
        guard = game._lead_patrol
    else:
        game.lead_alive = False
        guard = SecurityGuard(request.param,np.array([4.,4.],dtype=np.float32),0,((4,4),(8,4)))
    guard.pos = np.array([4.,4.],dtype=np.float32)
    guard.angle = 0
    game.aux_guards = [] if request.param == 'G0' else [guard]
    return guard


def test_all_sentries_detect_chase_and_request_reinforcements(game,sentry):
    game.env.player_pos = np.array([8.,4.],dtype=np.float32)
    sentry.visual_confidence = .999
    game._step_guard(sentry,TacticalAction.HOLD)
    assert sentry.state == 'CHASE'
    assert sentry.visual_confidence == 1
    assert sentry.last_seen_age == 0
    assert np.array_equal(sentry.last_seen,game.env.player_pos)
    assert game.team_contact_source == sentry.guard_id
    assert game.radio_timer is not None


def test_all_sentries_search_after_losing_contact(game,sentry):
    game.env.player_pos = np.array([30.,25.],dtype=np.float32)
    sentry.state = 'CHASE'
    sentry.last_seen = np.array([8.,4.],dtype=np.float32)
    sentry.visual_confidence = 1
    game._step_guard(sentry,TacticalAction.HOLD)
    assert sentry.state == 'SEARCH'
    assert sentry.guard_id in game._searches


def test_all_sentries_escape_smoke_with_same_energy_rules(game,sentry):
    game.env.player_pos = np.array([30.,25.],dtype=np.float32)
    game.smoke_clouds = [dict(pos=sentry.pos.copy(),radius=2.,ttl=8.)]
    sentry.vitals.step_smoke(1.,1.)
    game._step_guard(sentry,TacticalAction.PURSUE)
    assert sentry.state == 'ESCAPE SMOKE'
    assert sentry.smoke_goal is not None
    assert sentry.vitals.stamina == pytest.approx(.65)


def test_all_sentries_accept_ai_defence_intent(game,sentry):
    game.env.player_pos = np.array([30.,25.],dtype=np.float32)
    game._step_guard(sentry,TacticalAction.PROTECT_NEAREST_INTEL)
    assert sentry.state == 'PROTECT INTEL'
    assert any(np.array_equal(sentry.target,p) for p in game.env._intel)


def test_all_sentries_cannot_aim_or_fire_exhausted(game,sentry):
    game.env.player_pos = np.array([8.,4.],dtype=np.float32)
    sentry.vitals.stamina = 0
    sentry.vitals.exhausted = True
    ammo = sentry.ammo
    for _ in range(60):
        game._step_guard_weapon(sentry,True)
    assert sentry.ammo == ammo and sentry.aim_timer == 0
    assert game.player_vitals.health == 1


def test_runtime_executes_primary_and_patrol_through_the_same_engine(game):
    with patch.object(game,'_step_guard', wraps=game._step_guard) as step:
        game.tick(InputState())
    ids = [call.args[0].guard_id for call in step.call_args_list]
    assert ids.count('G0') == 1
    assert all(ids.count(guard.guard_id) == 1 for guard in game.aux_guards)


def test_patrol_ai_can_select_the_same_defence_intent(game,sentry):
    game.env.player_pos = np.array([30.,25.],dtype=np.float32)
    sentry.policy_action = int(TacticalAction.PROTECT_NEAREST_INTEL)
    game._step_aux_guard(sentry)
    assert sentry.state == 'PROTECT INTEL'
