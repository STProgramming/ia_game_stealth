import math

import numpy as np
import pytest

from shadow_ai.playable_combat import lane_blocked
from shadow_ai.soliton_security import SecurityGuard, line_clear
from test_interactive_tactics import game
from test_playable_actions import swat


@pytest.fixture(params=['game', 'swat'])
def squad(request):
    runtime = request.getfixturevalue(request.param)
    world, grid = runtime._combat_world()
    grid[:] = 0
    world.player_pos = np.array([14., 10.], dtype=np.float32)
    if hasattr(runtime, 'env'):
        runtime.lead_alive = False
        runtime.aux_guards = []
        runtime.team_contact, runtime.team_contact_age = world.player_pos.copy(), 0.
        runtime.swat_units = []
        for index, point in enumerate([(8., 10.), (6.9, 10.), (6.1, 10.)]):
            actor = SecurityGuard(f'S{index}', np.array(point, dtype=np.float32), 0., (point,))
            actor.has_shield = index == 0
            actor.weapon, actor.ammo = ('PISTOL', 16) if index == 0 else ('SMG', 31)
            actor.seeing_player, actor.visual_confidence = True, 1.
            actor.last_seen, actor.last_seen_age = world.player_pos.copy(), 0.
            actor.grenade_count = 0
            runtime.swat_units.append(('SHIELD' if index == 0 else 'ASSAULT', actor))
    else:
        world.blackboard.contact_pos, world.blackboard.contact_age = world.player_pos.copy(), 0.
        for index, (uid, actor) in enumerate(world.units.items()):
            actor.alive = index < 3
            if index >= 3:
                continue
            actor.pos = np.array([(8., 10.), (6.9, 10.), (6.1, 10.)][index], dtype=np.float32)
            actor.angle = 0.
            actor.local_visual, actor.local_visual_confidence = True, 1.
            actor.last_seen_pos, actor.last_seen_age = world.player_pos.copy(), 0.
            actor.grenade_count = 0
    return runtime


def advance(runtime, frames):
    for _ in range(frames):
        for uid, actor, _ in runtime._formation_members():
            runtime._step_swat_formation(uid, actor)
        if hasattr(runtime, 'env'):
            runtime.env.steps += 1
        else:
            runtime.core.world_ticks += 1


def test_shield_advances_slowly_and_followers_remain_physically_behind(squad):
    shield = squad._formation_shield()
    start = shield.pos.copy()
    advance(squad, 90)
    assert .1 < shield.pos[0] - start[0] < 2.
    for _, actor, _ in squad._formation_members():
        if actor is not shield:
            assert actor.pos[0] < shield.pos[0]
    assert shield.formation == 'SHIELD ADVANCE'


def test_followers_physically_peek_and_alternate_without_shooting_through_allies(squad):
    seen = set()
    for _ in range(420):
        advance(squad, 1)
        peeking = []
        for uid, actor, _ in squad._formation_members():
            if not actor.lean_amount:
                continue
            peeking.append(uid)
            seen.add(uid)
            allies = [other.pos for key, other, _ in squad._combat_actors() if key != uid]
            target = squad._formation_contact(actor)
            assert squad._swat_formation_can_fire(uid, actor)
            assert not lane_blocked(actor.pos, target, allies)
            assert line_clear(squad._combat_world()[1], actor.pos, target)
        assert len(peeking) <= 1
    followers = {uid for uid, actor, _ in squad._formation_members() if not squad._formation_has_shield(actor)}
    assert seen == followers


def test_peek_fire_origin_is_the_physical_actor_position(squad):
    shooter = None
    for _ in range(150):
        advance(squad, 1)
        shooter = next(((uid, actor, vitals) for uid, actor, vitals in squad._formation_members()
                        if actor.lean_amount), None)
        if shooter:
            break
    assert shooter is not None
    uid, actor, vitals = shooter
    origin = actor.pos.copy()
    assert squad._enemy_shot(uid, actor, vitals)
    shot = squad.shots[-1]['from']
    assert np.allclose([shot['x'], shot['y']], origin)


def test_narrow_corridor_keeps_followers_behind_shield_and_holds_fire(squad):
    _, grid = squad._combat_world()
    grid[:] = 1
    grid[10, :] = 0
    advance(squad, 120)
    shield = squad._formation_shield()
    for uid, actor, _ in squad._formation_members():
        assert not grid[int(round(actor.pos[1])), int(round(actor.pos[0]))]
        if actor is not shield:
            assert actor.lean_amount == 0
            assert not squad._swat_formation_can_fire(uid, actor)


def test_reload_holds_shield_still_and_sends_followers_back_behind_it(squad):
    advance(squad, 60)
    members = squad._formation_members()
    shield_uid, shield, _ = next(entry for entry in members if squad._formation_has_shield(entry[1]))
    follower_uid, follower, _ = next(entry for entry in members if entry[1] is not shield)
    follower.pos = shield.pos + np.array([.5, 1.1], dtype=np.float32)
    follower.lean_amount = 1.
    for uid, actor in [(shield_uid, shield), (follower_uid, follower)]:
        gun = squad._enemy_gun(uid, actor)
        gun.magazine, gun.chamber = 0, False
        actor.ammo = 0
        squad._enemy_reload(uid, actor)
    # A non-shield may choose its loaded pistol before any magazine replacement.
    assert squad._enemy_reloading(shield_uid, shield)
    shield_start, follower_start = shield.pos.copy(), follower.pos.copy()
    for _ in range(20):
        assert squad._step_reload_cover(shield_uid, shield)
        assert squad._step_reload_cover(follower_uid, follower)
    assert np.array_equal(shield.pos, shield_start)
    assert follower.pos[0] < follower_start[0]
    assert follower.lean_amount == 0
    assert not squad._swat_formation_can_fire(follower_uid, follower)


def test_down_shield_releases_stale_formation_and_stale_turn_gate(squad):
    advance(squad, 1)
    shield = squad._formation_shield()
    shield.alive = False
    for uid, actor, _ in squad._formation_members():
        assert not squad._step_swat_formation(uid, actor)
        assert not actor.formation and actor.lean_amount == 0
        assert squad._swat_formation_can_fire(uid, actor)


def test_old_report_does_not_track_hidden_player_or_create_a_live_formation(squad):
    for _, actor, _ in squad._formation_members():
        if hasattr(squad, 'env'):
            actor.seeing_player = False
        else:
            actor.local_visual = False
        actor.last_seen_age = 20.
    world, _ = squad._combat_world()
    if hasattr(squad, 'env'):
        squad.team_contact_age = 20.
    else:
        world.blackboard.contact_age = 20.
    starts = {uid: actor.pos.copy() for uid, actor, _ in squad._formation_members()}
    world.player_pos = np.array([2., 2.], dtype=np.float32)
    for uid, actor, _ in squad._formation_members():
        assert squad._formation_contact(actor) is None
        assert not squad._step_swat_formation(uid, actor)
        assert np.array_equal(actor.pos, starts[uid])


def test_regular_guard_uses_cover_during_reload_and_can_fire_after_completion(game):
    game.env.grid[:] = 0
    game.lead_alive = False
    guard = game.aux_guards[0]
    game.aux_guards = [guard]
    guard.pos = np.array([8., 10.], dtype=np.float32)
    guard.last_seen, guard.last_seen_age = np.array([12., 10.], dtype=np.float32), 0.
    guard.visual_confidence = 1.
    gun = game._enemy_gun(guard.guard_id, guard)
    gun.magazine, gun.chamber, guard.ammo = 0, False, 0
    assert game._enemy_reload(guard.guard_id, guard)
    old = guard.pos.copy()
    assert game._step_reload_cover(guard.guard_id, guard)
    assert guard.pos[0] < old[0] and guard.formation == 'RELOAD COVER'
    gun = game._enemy_gun(guard.guard_id, guard)
    gun.step(3.)
    guard.ammo = gun.rounds
    assert game._swat_formation_can_fire(guard.guard_id, guard)
    assert guard.state == 'SEARCH'


def test_lost_report_releases_campaign_formation_to_a_real_local_search(game):
    game.env.grid[:] = 0
    guard = game.aux_guards[0]
    guard.state = 'SHIELD ADVANCE'
    guard.last_seen, guard.last_seen_age = np.array([12., 10.], dtype=np.float32), 2.
    game._clear_formation(guard)
    assert guard.state == 'SEARCH'
    assert guard.guard_id in game._searches


def test_unconfirmed_search_anchor_does_not_restart_a_completed_search(squad):
    if hasattr(squad, 'env'):
        squad.team_contact_age = 100.
    else:
        squad.core.blackboard.contact_age = 100.
    for _, actor, _ in squad._formation_members():
        if hasattr(squad, 'env'):
            actor.seeing_player, actor.visual_confidence = False, 0.
        else:
            actor.local_visual, actor.local_visual_confidence = False, 0.
        actor.last_seen_age = 0.
    for uid, actor, _ in squad._formation_members():
        if hasattr(squad, 'env'):
            squad._start_guard_search(actor, actor.pos)
            search = squad._searches[uid]
            search.index = len(search.points)
        else:
            actor.last_seen_pos = actor.pos.copy()
        assert squad._formation_contact(actor) is None
        assert not squad._step_swat_formation(uid, actor)
        if hasattr(squad, 'env'):
            assert squad._searches[uid] is search
            assert search.index == len(search.points)


def test_shield_keeps_firing_while_stationary_waiting_for_followers(squad):
    shield = squad._formation_shield()
    for _, actor, _ in squad._formation_members():
        if actor is not shield:
            actor.pos[0] -= 5.
    before = shield.pos.copy()
    uid = squad._formation_uid(shield)
    assert squad._step_swat_formation(uid, shield)
    assert shield.formation == 'SHIELD HOLD' and np.array_equal(shield.pos, before)
    assert squad._swat_formation_can_fire(uid, shield)


def test_formation_movement_obeys_pending_friendly_flash_barrier(squad, monkeypatch):
    uid, actor, _ = squad._formation_members()[0]
    old = actor.pos.copy()
    checked = []
    def barrier(unit, start, candidate):
        checked.append((unit, start.copy(), candidate.copy()))
        return start.copy()
    monkeypatch.setattr(squad, '_friendly_flash_move', barrier, raising=False)
    squad._formation_move(uid, actor, old + [2., 0.], 1., 'SHIELD ADVANCE')
    assert checked and checked[0][0] is actor
    assert np.array_equal(actor.pos, old)
