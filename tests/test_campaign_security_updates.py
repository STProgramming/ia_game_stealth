from collections import deque
import math

import numpy as np
import pytest

from shadow_ai.interactive_runtime import InputState
from shadow_ai.playable_campaign import ARCHIVE_DOORS, MAP_WIDTH, MAP_HEIGHT
from shadow_ai.playable_navigation import LocalSearch
from shadow_ai.playable_vitals import PlayableVitals
from shadow_ai.soliton_security import SecurityGuard
from test_interactive_tactics import game, deploy


def reachable(grid, start):
    seen = {tuple(np.round(start).astype(int))}
    queue = deque(seen)
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            p = x+dx, y+dy
            if (0 <= p[0] < grid.shape[1] and 0 <= p[1] < grid.shape[0]
                    and not grid[p[1],p[0]] and p not in seen):
                seen.add(p)
                queue.append(p)
    return seen


@pytest.mark.parametrize('level', (1,2,3))
def test_expanded_wings_have_reachable_supplies_patrols_and_exit(game, level):
    game.level = level
    game.reset('routine')
    assert game.env.grid.shape == (MAP_HEIGHT, MAP_WIDTH)
    seen = reachable(game.env.grid, game.env.player_pos)
    assert tuple(game.env._exit.astype(int)) in seen
    assert game.env._exit[0] > 48 and game.env._exit[1] > 30
    assert any(p['pos'][0] > 48 for p in game.pickups)
    assert any(p['pos'][1] > 30 for p in game.pickups)
    assert any(x > 48 for g in game.aux_guards for x,y in g.patrol)
    for item in game.pickups:
        assert tuple(item['pos'].astype(int)) in seen


def test_all_archive_entrances_block_movement_and_dive_without_card(game):
    game.level = 2
    game.reset('routine')
    game.spawn_safe = False
    game.spawn_grace = 0
    for x,y in ARCHIVE_DOORS:
        above = y == 3
        start = np.array([float(x), y + (-.55 if above else .55)],dtype=np.float32)
        game.env.player_pos = start.copy()
        game._apply_player(InputState(down=above,up=not above))
        assert tuple(np.round(game.env.player_pos).astype(int)) != (x,y)
        game.player_actions.dive_timer = 0
        game.player_actions.dive_cooldown = 0
        game.player_vitals.stamina = 1
        game._apply_player(InputState(dive=True,down=above,up=not above))
        assert not (21 < game.env.player_pos[0] < 26 and 3.5 < game.env.player_pos[1] < 7.5)
    assert (24,6) not in reachable(game._player_grid(), np.array([24,2]))
    game.env.player_pos = np.array([24.,6.],dtype=np.float32)
    game._interact_latch = False
    game._apply_player(InputState(interact=True))
    assert not game.env.player_has_intel[0]


def test_card_carried_from_first_map_unlocks_archive_and_only_then_allows_intel(game):
    game.level = 1
    game.reset('routine')
    game.env.player_pos = game.env._intel[0].copy()
    game._apply_player(InputState(interact=True))
    game.status = 'ESCAPED'
    assert game.advance_level() and game.equipment.security_card
    assert not game.archive_unlocked
    game.env.player_pos = np.array([24.,2.],dtype=np.float32)
    game._apply_player(InputState(interact=True))
    assert game.archive_unlocked
    assert (24,6) in reachable(game._player_grid(),game.env.player_pos)
    assert all(not d['locked'] for d in game.state(False)['map']['doors'])
    game._apply_player(InputState())
    game.env.player_pos = game.env._intel[0].copy()
    game._apply_player(InputState(interact=True))
    assert game.env.player_has_intel[0]


@pytest.mark.parametrize('identified',(False,True))
def test_empty_target_reports_once_then_checks_or_protects_intel(game,identified):
    game.env.player_pos = np.array([60.,38.],dtype=np.float32)
    guard = game.aux_guards[0]
    guard.pos = np.array([18.,12.],dtype=np.float32)
    if identified:
        game._publish_contact(guard.pos,'G0')
    game._start_guard_search(guard,guard.pos)
    game._step_aux_guard(guard)
    message = game.guard_dialogue[guard.guard_id]
    assert 'punto vuoto' in message
    count = game.radio_updates
    game._report_empty_target(guard)
    assert game.radio_updates == count
    game._searches[guard.guard_id] = LocalSearch([],scan_seconds=.01)
    game._step_aux_guard(guard)
    assert guard.state == ('PROTECT INTEL' if identified else 'CHECK INTEL')
    assert game.env.known_intel_present[2]  # Radio inference cannot reveal a theft.


def test_combat_losses_request_one_fresh_squad_after_failed_search(game):
    deploy(game,[20,12])
    game.threat_level = 2
    game._down_guard(game.aux_guards[0].guard_id,True)
    game.team_contact_age = 20.
    for guard in game._active_guards():
        guard.seeing_player = False
        guard.state = 'SEARCH'
    game._step_casualty_reinforcements()
    assert game.radio_timer is None
    for guard in game._active_guards():
        guard.state = 'WITHDRAW' if guard.guard_id.startswith('S') else 'PROTECT INTEL'
    game._step_casualty_reinforcements()
    assert game.radio_timer == 1. and 'rinforzi' in game.guard_dialogue['HQ']
    count = game.radio_updates
    game._step_casualty_reinforcements()
    assert game.radio_updates == count
    old_ids = {u.guard_id for _,u in game.swat_units}
    for _ in range(30):
        game._step_radio()
    assert len(game.swat_units) == 6
    assert len({u.guard_id for _,u in game.swat_units}) == 6
    assert old_ids < {u.guard_id for _,u in game.swat_units}


def test_unconscious_guard_does_not_trigger_casualty_reinforcements(game):
    game._publish_contact(np.array([20.,12.],dtype=np.float32),'C1')
    game._down_guard(game.aux_guards[0].guard_id,False)
    assert not game._combat_casualties


def test_old_indoor_report_can_clear_through_side_door(game):
    deploy(game,[4,4])
    game.team_contact_age = 30.
    # Leave only the right-side entrance open.
    game.env.grid[10,6] = 1
    for _,guard in game.swat_units:
        assert game._step_tactical_entry(guard)
        plan = game._room_entries[guard.guard_id]
        assert plan['door'][0] == 9 and np.array_equal(plan['inward'],[-1,0])
        assert plan['stage'][0] > 9
    for _,guard in game.swat_units:
        guard.pos = game._room_entries[guard.guard_id]['stage'].copy()
    for _ in range(32):
        for _,guard in game.swat_units:
            game._step_tactical_entry(guard)
    assert len(game.flash_events) == 1
    assert not game.smoke_clouds  # The entry team must not smoke its own stack.
    assert all(p['phase']=='CLEAR ROOM' for p in game._room_entries.values())


def test_entry_uses_grenade_if_flashes_are_exhausted_and_waits_for_blast(game):
    deploy(game,[24,6])
    for _,guard in game.swat_units:
        guard.flash_count = 0
        game._step_tactical_entry(guard)
    for _,guard in game.swat_units:
        guard.pos = game._room_entries[guard.guard_id]['stage'].copy()
    for _ in range(20):
        for _,guard in game.swat_units:
            game._step_tactical_entry(guard)
    assert len(game.explosives) == 1
    assert all(p['phase']=='BREACH WAIT' for p in game._room_entries.values())
    for _ in range(50):
        game._step_explosives()
        for _,guard in game.swat_units:
            game._step_tactical_entry(guard)
    assert not game.explosives and game.blasts
    assert all(p['phase']=='CLEAR ROOM' for p in game._room_entries.values())


def test_light_wound_bleeds_less_than_severe_wound_and_still_leaves_blood():
    light = PlayableVitals(health=.8,bleeding=.5)
    severe = PlayableVitals(health=.54,bleeding=.5)
    assert 0 < light.step_bleeding(.1,2.) < severe.step_bleeding(.1,2.)
    light.treat_wound()
    assert light.step_bleeding(1.,2.) == 0


def test_walking_through_remote_kill_blood_stains_shoes_at_players_feet(game):
    victim = game.aux_guards[0]
    victim.pos = np.array([12.,12.],dtype=np.float32)
    game.env.player_pos = np.array([20.,12.],dtype=np.float32)
    game._down_guard(victim.guard_id,True)
    assert game.bloody_timer == 0
    game.env.player_pos = np.array([12.2,12.],dtype=np.float32)
    game._leave_blood_trail(np.array([11.,12.],dtype=np.float32))
    assert game.bloody_timer > 0
    assert np.array_equal(game.blood[-1].pos,game.env.player_pos)
    assert game.player_vitals.bleeding == 0
