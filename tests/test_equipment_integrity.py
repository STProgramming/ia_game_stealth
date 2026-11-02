import pytest
from shadow_ai.playable_equipment import Equipment
from shadow_ai.playable_firearms import Firearm
from shadow_ai.playable_firearms import weapon_mass, weapon_draw_time, weapon_movement_factor
from shadow_ai.interactive_runtime import InputState
from test_firearms_runtime import runtime, action, wait_reload


def total_rounds(equipment):
    return equipment.loose_ammo + sum(g.rounds + sum(g.spares) + g.incoming +
        (g.outgoing if g.reload_mode and not g.discarded else 0) for g in equipment.firearms.values())


def test_duplicate_weapon_keeps_loaded_rounds_and_every_reserve_magazine():
    loot=Firearm.loaded('PISTOL',16,[15,15])
    equipment=Equipment()
    equipment.collect('PISTOL',16,loot.snapshot())
    equipment.collect('PISTOL',16,loot.snapshot())
    assert total_rounds(equipment)==92


def test_generic_ammo_is_not_consumed_or_assigned_by_inventory_reads():
    equipment=Equipment(weapon='PISTOL',ammo=0,weapons={'PISTOL'},selected='PISTOL')
    equipment.collect('AMMO',30)
    for _ in range(10): equipment.snapshot()
    assert equipment.loose_ammo==30
    assert equipment.gun().spares==[]


def test_snapshot_never_rebuilds_a_reload_from_a_stale_display_counter():
    equipment=Equipment(weapon='PISTOL',ammo=7,weapons={'PISTOL'},selected='PISTOL')
    gun=equipment.gun(); gun.spares=[15]
    gun.start_reload()
    state=equipment.snapshot()
    assert equipment.gun() is gun
    assert state['firearm']['reload_mode']=='TACTICAL'
    assert total_rounds(equipment)==22


def test_duplicate_pickup_does_not_cancel_the_current_reload():
    equipment=Equipment(weapon='PISTOL',ammo=7,weapons={'PISTOL'},selected='PISTOL')
    gun=equipment.gun(); gun.spares=[15]; gun.start_reload(); equipment.sync_ammo()
    equipment.collect('PISTOL',5)
    assert gun.reload_mode=='TACTICAL'
    assert total_rounds(equipment)==27


def test_shared_ammo_can_reload_an_empty_second_weapon(runtime):
    equipment=runtime.equipment
    equipment.collect('RIFLE',0)
    equipment.collect('AMMO',45)
    equipment.select('RIFLE')
    equipment.step_draw(equipment.draw_duration)
    before=total_rounds(equipment)
    action(runtime,reload=True)
    wait_reload(runtime)
    assert equipment.gun().rounds==30
    assert equipment.loose_ammo==15
    assert total_rounds(equipment)==before


def test_switching_weapons_blocks_shooting_and_reload_until_drawn(runtime):
    equipment=runtime.equipment
    equipment.collect('RIFLE',12)
    equipment.collect('AMMO',30)
    equipment.select('RIFLE')
    duration=equipment.draw_remaining
    for _ in range(10):
        action(runtime,fire=True,reload=True)
        equipment.select('RIFLE') # Repeated network messages must not reset extraction.
    assert 0<equipment.draw_remaining<duration
    assert equipment.ammo==12 and not equipment.gun().reload_mode and not runtime.shots
    equipment.step_draw(duration)
    action(runtime,fire=True)
    assert equipment.ammo==11 and len(runtime.shots)==1


@pytest.mark.parametrize('stance',[{},dict(crouch=True),dict(sprint=True)])
def test_weapon_weight_scales_actual_movement_in_both_modes(runtime,stance):
    import numpy as np
    world,_=runtime._combat_world()
    distances=[]
    for kind in ['PISTOL','RIFLE','SHOTGUN']:
        world.player_pos=np.array([10.,10.],dtype=np.float32)
        runtime.equipment=Equipment(weapon=kind,ammo=3,weapons={kind},selected=kind)
        runtime._apply_player(InputState(right=True,**stance))
        distances.append(float(world.player_pos[0]-10))
    assert distances[0]>distances[1]>distances[2]>0
    assert distances[0]/distances[1]==pytest.approx(weapon_movement_factor('PISTOL')/weapon_movement_factor('RIFLE'),rel=.001)
    assert weapon_draw_time('RIFLE')/weapon_draw_time('PISTOL')==pytest.approx(weapon_mass('RIFLE')/weapon_mass('PISTOL'))


def test_selection_is_latched_once_between_input_ticks():
    selection=InputState(selected_item='RIFLE')
    later=InputState()
    later.merge_transients(selection)
    assert later.selected_item=='RIFLE'
    later.clear_transients()
    assert later.selected_item==''


def test_switching_and_tactical_cancellation_conserve_all_rounds():
    equipment=Equipment()
    equipment.collect('PISTOL',7)
    equipment.collect('RIFLE',25)
    equipment.collect('AMMO',40)
    before=total_rounds(equipment)
    for kind in ['RIFLE','PISTOL']*6:
        equipment.select(kind)
        equipment.step_draw(3)
        equipment.start_reload()
        equipment.snapshot()
    equipment.gun().cancel_reload()
    assert total_rounds(equipment)==before


def test_retained_weapon_survives_sector_transition_during_reload():
    equipment=Equipment()
    equipment.collect('RIFLE',12)
    equipment.collect('AMMO',40)
    equipment.step_draw(3)
    equipment.start_reload()
    before=total_rounds(equipment)
    next_sector=equipment.next_level_loadout()
    assert total_rounds(next_sector)==before
    assert not next_sector.gun().reload_mode
    assert next_sector.gun() is not equipment.gun()
