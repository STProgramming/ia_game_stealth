import math
import numpy as np
import pytest
from shadow_ai import TacticalAction
from shadow_ai.room_handover import inside_room
from shadow_ai.soliton_security import SecurityGuard
from test_interactive_tactics import game


def room_scenario(game):
    game.level = 1
    game.reset('routine')
    game.spawn_safe, game.spawn_grace = False, 0.
    game.env.player_pos = np.array([60., 37.], dtype=np.float32)
    game.env.guard_pos = np.array([4., 4.], dtype=np.float32)
    game._lead_patrol.pos = game.env.guard_pos.copy()
    for i, guard in enumerate(game.aux_guards):
        guard.pos = np.array([5. + i, 6.], dtype=np.float32)
        guard.patrol = ((5. + i, 6.),)
    game.swat_units = []
    for i, role in enumerate(['SHIELD', 'ASSAULT', 'LEADER']):
        unit = SecurityGuard(f'S{i}', np.array([12. + i, 5.], dtype=np.float32), math.pi, ((12 + i, 5),))
        unit.has_shield = i == 0
        unit.state = 'RESPOND'
        game.swat_units.append((role, unit))
    game.team_contact = np.array([4., 4.], dtype=np.float32)
    game.team_contact_age = 2.
    game.swat_phase = 'ACTIVE'
    game.facility_alert = 'SEARCH'
    game.security_compromised = True
    room = game._reported_room(game.team_contact)
    game._ensure_room_handover(room)
    return room


def step_regulars(game):
    for guard in [game._lead_patrol] + game.aux_guards:
        game._step_guard(guard, TacticalAction.PATROL)
        if guard.guard_id == 'G0':
            game.env.guard_pos, game.env.guard_angle = guard.pos.copy(), guard.angle


def test_radio_assigns_every_regular_guard_including_the_lead(game):
    room = room_scenario(game)
    assert set(game._room_orders) == {'G0', 'G1', 'G2'}
    assert {o['assignment'] for o in game._room_orders.values()} == {'REAR GUARD', 'PERIMETER PATROL', 'PROTECT INTEL'}
    assert all(not inside_room(o['exit'], room) and not inside_room(o['goal'], room) for o in game._room_orders.values())
    assert all('SWAT: dobbiamo entrare' in game.guard_dialogue[uid] for uid in game._room_orders)
    assert len(game._room_handover_snapshot()['waiting_for']) == 3


def test_swat_holds_outside_and_uses_no_breach_devices_until_every_guard_leaves(game):
    room = room_scenario(game)
    for _ in range(150):
        for _, unit in game.swat_units:
            assert game._step_tactical_entry(unit)
            assert not inside_room(unit.pos, room)
            assert unit.state == 'WAIT GUARDS'
    assert not game.flash_events and not game.explosives
    assert all(entry['phase'] == 'STACK' for entry in game._room_entries.values())


def test_guards_actually_evacuate_then_swat_enters_and_guards_never_reenter(game):
    room = room_scenario(game)
    saw_breach = False
    for _ in range(650):
        step_regulars(game)
        blocked = bool(game._room_blockers(room))
        for _, unit in game.swat_units:
            game._step_tactical_entry(unit)
            if blocked:
                assert not inside_room(unit.pos, room)
                assert not game.flash_events and not game.explosives
            saw_breach |= inside_room(unit.pos, room)
        for uid, order in game._room_orders.items():
            if order['evacuated']:
                guard = next(g for g in game._active_guards() if g.guard_id == uid)
                assert not inside_room(guard.pos, room)
        if saw_breach:
            break
    assert not game._room_blockers(room)
    assert all(o['evacuated'] for o in game._room_orders.values())
    assert saw_breach


def test_actual_visual_contact_does_not_override_the_wait_for_guards(game):
    room = room_scenario(game)
    unit = game.swat_units[0][1]
    unit.state = 'CHASE'
    unit.visual_confidence = 1.
    game._step_tactical_entry(unit)
    assert unit.state == 'WAIT GUARDS' and not inside_room(unit.pos, room)


def test_inactive_guards_do_not_deadlock_the_handover(game):
    room = room_scenario(game)
    game.lead_alive = False
    for guard in game.aux_guards:
        guard.unconscious = True
    assert not game._room_blockers(room)


def test_changed_radio_room_cancels_old_orders_without_using_hidden_player_position(game):
    original = room_scenario(game)
    game.env.player_pos = np.array([4., 4.], dtype=np.float32)
    game.team_contact = np.array([34., 5.], dtype=np.float32)
    new_room = game._reported_room(game.team_contact)
    assert new_room != original
    game._ensure_room_handover(new_room)
    assert game._handover_room == new_room
    assert not game._room_orders


def test_stunned_guard_blocks_the_entry_until_evacuated(game):
    room = room_scenario(game)
    guard = game.aux_guards[0]
    guard.vitals.hit_stun_remaining = 1.
    for _ in range(8):
        game._step_guard(guard)
        game._step_tactical_entry(game.swat_units[0][1])
    assert guard in game._room_blockers(room)
    assert game.swat_units[0][1].state == 'WAIT GUARDS'
