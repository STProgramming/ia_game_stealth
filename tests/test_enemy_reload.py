import numpy as np
import pytest

from shadow_ai.interactive_runtime import InputState
from shadow_ai.soliton_security import SecurityGuard
from test_firearms_runtime import runtime
from test_interactive_tactics import game


def actor_for(runtime, weapon='PISTOL', rounds=0, contact=False):
    world, grid = runtime._combat_world()
    grid[:] = 0
    world.player_pos = np.array([14., 10.], dtype=np.float32)
    if hasattr(runtime, 'env'):
        runtime.lead_alive = False
        actor = SecurityGuard('S9', np.array([8., 10.], dtype=np.float32), 0., ((8, 10),),
                              weapon=weapon, ammo=rounds)
        actor.seeing_player = contact
        actor.visual_confidence = float(contact)
        runtime.aux_guards = []
        runtime.swat_units = [('ASSAULT', actor)]
        runtime.team_contact = None
        vitals = actor.vitals
    else:
        for unit in world.units.values():
            unit.alive = False
        actor = next(iter(world.units.values()))
        actor.alive = True
        actor.weapon, actor.ammo = weapon, rounds
        actor.pos, actor.angle = np.array([8., 10.], dtype=np.float32), 0.
        actor.local_visual = contact
        world.blackboard.contact_pos = None
        vitals = runtime._unit_vitals[actor.uid]
    uid = getattr(actor, 'guard_id', getattr(actor, 'uid', None))
    return uid, actor, vitals


def drain(runtime, uid, actor):
    gun = runtime._enemy_gun(uid, actor)
    fired = 0
    while gun.fire():
        fired += 1
    runtime._sync_enemy_ammo(uid, actor, gun)
    return fired


def advance(runtime, uid, actor, seconds):
    for _ in range(int(np.ceil(seconds / runtime.game_dt))):
        runtime._step_enemy_firearms(uid, actor)


def stock(runtime, uid, actor):
    runtime._enemy_gun(uid, actor)
    return sum(gun.rounds + sum(gun.spares) + gun.incoming +
               (gun.outgoing if gun.reload_mode and not gun.discarded else 0)
               for gun in runtime._enemy_arsenals[uid].values())


def test_ordinary_guard_uses_normal_tactical_reload_and_preserves_magazine(game):
    game.lead_alive = False
    actor = SecurityGuard('G9', np.array([8., 8.], dtype=np.float32), 0., ((8, 8),), ammo=0)
    game.aux_guards = [actor]
    assert game._enemy_reload('G9', actor)
    gun = game._enemy_guns['G9']
    assert gun.reload_mode == 'TACTICAL' and gun.reload_duration == 2.8
    advance(game, 'G9', actor, 1.1)
    assert gun.reload_mode == 'TACTICAL' and actor.ammo == 0
    assert not any(item.get('owner') == 'G9' for item in game.pickups)
    advance(game, 'G9', actor, 1.8)
    assert not gun.reload_mode and actor.ammo == 15


def test_swat_fast_reload_is_timed_and_drops_one_magazine(runtime):
    uid, actor, _ = actor_for(runtime)
    assert runtime._enemy_reload(uid, actor)
    gun = runtime._enemy_guns[uid]
    assert gun.reload_mode == 'FAST' and gun.reload_duration == 1.05
    assert len([item for item in runtime.pickups if item.get('owner') == uid]) == 1
    runtime._enemy_reload(uid, actor)
    assert len([item for item in runtime.pickups if item.get('owner') == uid]) == 1
    advance(runtime, uid, actor, 1.)
    assert gun.reload_mode and actor.ammo == 0
    advance(runtime, uid, actor, .1)
    assert not gun.reload_mode and actor.ammo == 15


def test_empty_enemy_reloads_automatically_without_visual_contact(runtime):
    uid, actor, _ = actor_for(runtime)
    runtime._step_player_firearm(InputState())
    assert runtime._enemy_reloading(uid, actor)
    for _ in range(34):
        runtime._step_player_firearm(InputState())
    assert actor.ammo == 15 and not runtime._enemy_reloading(uid, actor)


def test_all_loaded_and_reserve_rounds_are_finite_across_repeated_enemy_reloads(runtime):
    uid, actor, _ = actor_for(runtime, rounds=16)
    initial = stock(runtime, uid, actor)
    fired = 0
    for _ in range(8):
        fired += drain(runtime, uid, actor)
        runtime._enemy_reload(uid, actor)
        advance(runtime, uid, actor, 1.1)
    assert initial == fired == 46
    assert stock(runtime, uid, actor) == actor.ammo == 0
    assert not runtime._enemy_reload(uid, actor)
    for _ in range(5):
        runtime._actor_firearm_snapshot(uid, actor)
    assert stock(runtime, uid, actor) == 0


@pytest.mark.parametrize('weapon', ['SMG', 'RIFLE', 'SHOTGUN'])
def test_swat_switches_to_actual_loaded_pistol_faster_than_reloading(runtime, weapon):
    uid, actor, vitals = actor_for(runtime, weapon=weapon, contact=True)
    assert runtime._enemy_reload(uid, actor)
    assert actor.weapon == 'PISTOL' and actor.ammo == 16
    assert runtime._enemy_drawing(uid, actor)
    assert actor.enemy_draw_duration == pytest.approx(.42)
    assert actor.enemy_draw_duration < 1.05
    assert not runtime._enemy_shot(uid, actor, vitals)
    assert not any(item.get('owner') == uid for item in runtime.pickups)
    advance(runtime, uid, actor, .45)
    assert not runtime._enemy_drawing(uid, actor)
    assert runtime._enemy_arsenals[uid][weapon].rounds == 0


def test_firing_live_gun_then_syncing_never_recreates_spent_rounds(runtime):
    uid, actor, _ = actor_for(runtime, rounds=3)
    gun = runtime._enemy_gun(uid, actor)
    assert gun.fire()
    # A direct chamber mutation is authoritative even before the public mirror
    # has been refreshed by the runtime's shot executor.
    assert runtime._enemy_gun(uid, actor) is gun
    assert actor.ammo == 2


def test_both_empty_swat_weapons_reload_from_their_own_reserves(runtime):
    uid, actor, _ = actor_for(runtime, weapon='SMG', contact=True)
    initial = stock(runtime, uid, actor)
    assert runtime._enemy_reload(uid, actor) and actor.weapon == 'PISTOL'
    advance(runtime, uid, actor, .45)
    assert drain(runtime, uid, actor) == 16
    # Both magazines are empty. First reload the pistol instead of inventing
    # another weapon; after its spare is spent the primary must be reloaded.
    assert runtime._enemy_reload(uid, actor)
    assert actor.weapon == 'PISTOL' and runtime._enemy_reloading(uid, actor)
    advance(runtime, uid, actor, 1.1)
    assert drain(runtime, uid, actor) == 15
    assert runtime._enemy_reload(uid, actor) and actor.weapon == 'SMG'
    advance(runtime, uid, actor, 2.5)
    assert actor.ammo == 30 and runtime._enemy_arsenals[uid]['PISTOL'].rounds == 0
    assert stock(runtime, uid, actor) == initial - 31


def test_safe_swat_services_both_empty_weapons_then_returns_to_primary(runtime):
    uid, actor, _ = actor_for(runtime, weapon='SMG', contact=False)
    runtime._enemy_gun(uid, actor)
    pistol = runtime._enemy_arsenals[uid]['PISTOL']
    pistol.magazine, pistol.chamber = 0, False
    advance(runtime, uid, actor, 6.)
    assert actor.weapon == 'SMG'
    assert runtime._enemy_arsenals[uid]['SMG'].rounds == 30
    assert pistol.rounds == 15 and pistol.spares == []
    assert not runtime._enemy_reloading(uid, actor) and not runtime._enemy_drawing(uid, actor)


def test_complete_long_gun_and_sidearm_cycle_exhausts_every_round_without_regeneration(runtime):
    uid, actor, _ = actor_for(runtime, weapon='SMG', rounds=31, contact=True)
    initial = stock(runtime, uid, actor)
    fired = 0
    used_weapons = set()
    for _ in range(12):
        used_weapons.add(actor.weapon)
        fired += drain(runtime, uid, actor)
        runtime._enemy_reload(uid, actor)
        advance(runtime, uid, actor, 2.5)
    assert used_weapons == {'SMG', 'PISTOL'}
    assert initial == fired == 122
    assert stock(runtime, uid, actor) == 0
    assert not runtime._enemy_reload(uid, actor)


def test_switching_never_grants_a_weapon_not_in_actual_arsenal(runtime):
    uid, actor, _ = actor_for(runtime, rounds=3)
    runtime._enemy_gun(uid, actor)
    assert not runtime._enemy_switch_weapon(uid, actor, 'RIFLE')
    assert actor.weapon == 'PISTOL'


def test_swat_down_loot_contains_primary_and_sidearm_with_all_remaining_rounds(runtime):
    uid, actor, _ = actor_for(runtime, weapon='SMG', rounds=7, contact=True)
    before = stock(runtime, uid, actor)
    if hasattr(runtime, 'env'):
        runtime._down_guard(uid, True)
    else:
        runtime._down_swat_player_target(actor, True)
    weapons = [item for item in runtime.pickups if item['source'] == uid and 'firearm' in item]
    assert {item['kind'] for item in weapons} == {'SMG', 'PISTOL'}
    total = sum(item['firearm']['magazine'] + item['firearm']['chamber'] +
                sum(item['firearm']['spare_magazines']) for item in weapons)
    assert total == before
    assert stock(runtime, uid, actor) == 0
    assert not runtime._enemy_inventory_loot(uid, actor)


def test_disarmed_primary_does_not_drop_a_second_copy_with_remaining_sidearm(runtime):
    uid, actor, _ = actor_for(runtime, weapon='SMG', rounds=7)
    original = stock(runtime, uid, actor)
    disarmed = runtime._enemy_weapon_loot(uid, actor)
    primary_rounds = disarmed['magazine'] + disarmed['chamber'] + sum(disarmed['spare_magazines'])
    actor.weapon, actor.ammo, actor.disarmed = None, 0, True
    remaining = runtime._enemy_inventory_loot(uid, actor)
    assert len(remaining) == 1 and remaining[0][0] == 'PISTOL'
    assert primary_rounds + remaining[0][1] + sum(remaining[0][2]['spare_magazines']) == original


def test_disarm_during_normal_reload_cannot_restore_retained_rounds_to_the_guard(game):
    actor = SecurityGuard('G9', np.array([8., 8.], dtype=np.float32), 0., ((8, 8),), ammo=7)
    assert game._enemy_reload('G9', actor)
    assert actor.ammo == 1
    loot = game._enemy_weapon_loot('G9', actor)
    assert loot['magazine'] + loot['chamber'] == 7
    assert game._enemy_gun('G9', actor).rounds == 0
    assert not game._enemy_gun('G9', actor).spares
