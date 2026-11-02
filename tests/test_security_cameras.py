import math
import numpy as np
import pytest
from shadow_ai.soliton_security import SecurityCamera, BodyEvidence, CAMERA_PAN_DEGREES
from test_interactive_tactics import game


def test_camera_pan_is_at_most_125_degrees_even_with_an_oversized_sweep():
    camera = SecurityCamera('C0', np.array([10.,10.]), .7, sweep=math.pi)
    angles=[]
    for _ in range(1000):
        camera.step(.03); angles.append(camera.angle)
    assert max(angles)-min(angles) == pytest.approx(math.radians(125), abs=.0001)
    assert all(abs(angle-camera.center_angle) <= math.radians(CAMERA_PAN_DEGREES/2) for angle in angles)


@pytest.mark.parametrize('angle',[0,math.pi/2,math.pi,-math.pi/2])
def test_camera_cannot_see_directly_beneath_it_at_any_angle(angle):
    camera=SecurityCamera('C0', np.array([10.,10.]), angle, angle=angle)
    grid=np.zeros((25,25),dtype=np.uint8)
    for distance in [0,.2,.8,camera.blind_radius]:
        point=camera.pos + np.array([math.cos(angle),math.sin(angle)])*distance
        assert not camera.sees(grid,point)
    assert camera.sees(grid,camera.pos+np.array([math.cos(angle),math.sin(angle)])*2)


def test_entering_the_blind_spot_immediately_loses_camera_lock_without_reporting(game):
    game.env.grid[:]=0
    camera=SecurityCamera('C0',np.array([10.,10.]),0,confidence=1.,tracking=True)
    game.cameras=[camera]
    game.env.player_pos=np.array([10.,10.],dtype=np.float32)
    game.team_contact=None
    game._step_cameras()
    assert not camera.tracking and camera.confidence == 0
    assert game.team_contact is None
    state=game.state()['cameras'][0]
    assert state['vision_inner_range'] == camera.blind_radius
    assert state['pan_degrees'] == pytest.approx(125)


def evidence_camera(game):
    game.env.grid[:] = 0
    camera = SecurityCamera('C0', np.array([10., 10.], dtype=np.float32), 0, sweep=0)
    game.cameras = [camera]
    # Hiding beneath a camera must not disable its view of the rest of the room.
    game.env.player_pos = camera.pos.copy()
    return camera


def test_camera_reports_missing_intel_and_enters_caution_without_seeing_player(game):
    evidence_camera(game)
    index = game.env.episode_intel_order[0]
    game.env._intel[index] = np.array([14., 10.], dtype=np.float32)
    game.env.player_has_intel[index] = 1
    game._step_cameras()
    assert game._intel_checks[index]['source'] == 'C0'
    assert game.team_contact is None
    for _ in range(36):
        game._step_intel_radio()
    assert game.facility_alert == 'SUSPICIOUS'
    assert not game.env.known_intel_present[index]
    assert "è stato rubato l'intel" in game.guard_dialogue['C0']
    assert game.team_contact is None and game.radio_timer is None
    updates = game.radio_updates
    game._step_cameras()
    game._step_intel_radio()
    assert game.radio_updates == updates


@pytest.mark.parametrize('lethal', [False, True])
def test_camera_reports_unconscious_and_dead_guards_once(game, lethal):
    evidence_camera(game)
    body = BodyEvidence(90, 'G9', np.array([14., 10.], dtype=np.float32), 0, lethal)
    game.bodies = [body]
    game._step_cameras()
    assert body.discovered and game.security_compromised
    assert ('morta' if lethal else 'incosciente') in game.guard_dialogue['C0']
    assert game._guard_dispatch is not None
    assert game.team_contact is None
    updates = game.radio_updates
    game._step_cameras()
    assert game.radio_updates == updates


@pytest.mark.parametrize('obstruction', ['wall', 'smoke', 'blind_spot', 'behind', 'offline', 'far'])
def test_camera_needs_clear_sight_of_body_and_missing_intel(game, obstruction):
    camera = evidence_camera(game)
    point = np.array([14., 10.], dtype=np.float32)
    if obstruction == 'wall':
        game.env.grid[10, 12] = 1
    elif obstruction == 'smoke':
        game.smoke_clouds = [dict(pos=point.copy(), radius=2., ttl=8.)]
    elif obstruction == 'blind_spot':
        point = camera.pos + np.array([.5, 0.], dtype=np.float32)
    elif obstruction == 'behind':
        point = camera.pos - np.array([4., 0.], dtype=np.float32)
    elif obstruction == 'offline':
        camera.online = False
    else:
        point = camera.pos + np.array([15., 0.], dtype=np.float32)
    index = game.env.episode_intel_order[0]
    game.env._intel[index] = point.copy()
    game.env.player_has_intel[index] = 1
    game.bodies = [BodyEvidence(90, 'G9', point.copy(), 0, False)]
    game._step_cameras()
    assert not game._intel_checks
    assert not game.bodies[0].discovered
    assert not game.security_compromised


def test_offline_camera_cannot_complete_missing_intel_report(game):
    camera = evidence_camera(game)
    index = game.env.episode_intel_order[0]
    game.env._intel[index] = np.array([14., 10.], dtype=np.float32)
    game.env.player_has_intel[index] = 1
    game._step_cameras()
    camera.online = False
    game._step_intel_radio()
    assert not game._intel_checks and not game.security_compromised
