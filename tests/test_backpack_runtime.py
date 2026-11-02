import numpy as np
import pytest

from shadow_ai.interactive_runtime import InputState
from shadow_ai.playable_equipment import Equipment, BACKPACK_CAPACITY
from shadow_ai.playable_firearms import Firearm
from shadow_ai.playable_actions import PlayerActions
from shadow_ai.playable_vitals import PlayableVitals
from test_firearms_runtime import runtime
from test_interactive_tactics import game


def full_backpack():
    equipment = Equipment(knife=True)
    for kind in ('PISTOL', 'AMMO', 'BANDAGE', 'MEDKIT', 'CASING', 'MAGAZINE', 'FLASH', 'SMOKE', 'SNACK'):
        assert equipment.collect(kind, 3)
    assert equipment.slot_count == BACKPACK_CAPACITY
    return equipment


def rounds(equipment):
    return equipment.loose_ammo + sum(g.rounds + sum(g.spares) + g.incoming +
        (g.outgoing if g.reload_mode and not g.discarded else 0) for g in equipment.firearms.values())


def test_full_backpack_rejects_new_types_but_stacks_items_and_duplicate_weapon_ammo():
    equipment = full_backpack()
    assert not equipment.collect('RIFLE', 30)
    assert not equipment.collect('GRENADE', 1)
    assert equipment.collect('SNACK', 2) and equipment.snacks == 5
    before = rounds(equipment)
    incoming = Firearm.loaded('PISTOL', 16, [15, 8])
    assert equipment.collect('PISTOL', 16, incoming.snapshot())
    assert rounds(equipment) == before + 39
    assert equipment.slot_count == 10


@pytest.mark.parametrize('kind,mass', [('KNIFE', .3), ('HAZMAT', 1.2), ('SAMPLE_VIAL', .1), ('SECURITY_CARD', .02)])
def test_duplicate_equipment_stacks_in_one_slot_counts_weight_and_drops_all_copies(kind, mass):
    equipment = Equipment()
    assert equipment.collect(kind, 1)
    before = equipment.total_weight
    assert equipment.collect(kind, 2)
    assert equipment.item_quantity(kind) == equipment.snapshot()['item_quantities'][kind] == 3
    assert equipment.slot_count == 1
    assert equipment.total_weight == pytest.approx(before + 2 * mass)
    assert equipment.select(kind)
    pickups = []
    equipment.drop_selected(np.zeros(2), pickups, 20)
    assert pickups[0]['amount'] == 3 and equipment.item_quantity(kind) == 0
    assert equipment.slot_count == 0 and equipment.total_weight == 0
    assert equipment.collect(kind, pickups[0]['amount'])
    assert equipment.item_quantity(kind) == 3


def test_spare_equipment_can_stack_when_backpack_is_full_and_fall_drops_only_one():
    from types import SimpleNamespace
    equipment = full_backpack()
    assert equipment.collect('KNIFE', 2)
    assert equipment.slot_count == 10 and equipment.item_quantity('KNIFE') == 3
    equipment.select('KNIFE')
    pickups = []
    equipment.drop_on_fall(SimpleNamespace(random=lambda: 0), np.zeros(2), pickups, 21)
    assert pickups[0]['amount'] == 1
    assert equipment.knife and equipment.item_quantity('KNIFE') == 2 and equipment.slot_count == 10


def test_next_sector_keeps_independent_unique_item_piles():
    equipment = Equipment()
    equipment.collect('SAMPLE_VIAL', 3)
    next_sector = equipment.next_level_loadout()
    assert next_sector.consume_item('SAMPLE_VIAL')
    assert next_sector.sample_vial and next_sector.item_quantity('SAMPLE_VIAL') == 2
    assert equipment.item_quantity('SAMPLE_VIAL') == 3


def test_dropping_whole_stack_frees_one_slot_and_can_be_recovered():
    equipment = full_backpack()
    equipment.select('SNACK')
    pickups = []
    assert equipment.drop_selected(np.zeros(2), pickups, 1) == 'SNACK'
    assert equipment.slot_count == 9 and equipment.snacks == 0
    assert pickups[0]['amount'] == 3
    assert equipment.collect('RIFLE', 8)
    assert not equipment.collect('SNACK', 3)


def test_dropping_during_reload_preserves_every_round_in_the_weapon():
    equipment = Equipment()
    equipment.collect('PISTOL', 7)
    gun = equipment.gun()
    gun.spares = [15, 4]
    equipment.step_draw(3)
    equipment.start_reload()
    before = rounds(equipment)
    pickups = []
    assert equipment.drop_selected(np.zeros(2), pickups, 4) == 'PISTOL'
    assert equipment.slot_count == 0 and rounds(equipment) == 0
    assert equipment.collect('PISTOL', pickups[0]['amount'], pickups[0]['firearm'])
    assert rounds(equipment) == before


def test_backpack_weight_includes_stacks_and_stowed_weapons_and_drop_restores_mobility():
    equipment = Equipment(knife=True)
    initial = equipment.total_weight
    equipment.collect('RIFLE', 30)
    equipment.select('KNIFE')
    initial_factor = equipment.backpack_movement_factor
    equipment.collect('MEDKIT', 6)
    assert equipment.total_weight > initial + 8
    assert equipment.backpack_movement_factor < initial_factor
    equipment.select('MEDKIT')
    equipment.drop_selected(np.zeros(2), [], 2)
    equipment.select('KNIFE')
    assert equipment.backpack_movement_factor == pytest.approx(initial_factor)


def test_e_collects_existing_item_at_capacity_and_leaves_unfittable_loot(runtime):
    runtime.equipment = full_backpack()
    world, _ = runtime._combat_world()
    runtime.pickups = [dict(id=kind, kind=kind, amount=5, pos=world.player_pos.copy(), source='MAP')
                       for kind in ('PISTOL', 'SNACK', 'GRENADE')]
    before = rounds(runtime.equipment)
    runtime._apply_player(InputState(interact=True))
    assert rounds(runtime.equipment) == before + 5
    assert runtime.equipment.snacks == 8
    assert [p['kind'] for p in runtime.pickups] == ['GRENADE']
    assert runtime.equipment.slot_count == 10


def test_manual_drop_input_works_in_both_playable_modes(runtime):
    runtime.equipment = Equipment(knife=True, smoke=3, selected='SMOKE')
    before = runtime.equipment.total_weight
    runtime._apply_player(InputState(drop_item=True))
    drop = next(p for p in runtime.pickups if p['source'] == 'PLAYER')
    assert drop['kind'] == 'SMOKE' and drop['amount'] == 3
    assert runtime.equipment.smoke == 0 and runtime.equipment.total_weight < before


def test_stowed_load_reduces_actual_player_movement(runtime):
    world, _ = runtime._combat_world()
    runtime.equipment = Equipment(knife=True)
    origin = world.player_pos.copy()
    runtime._apply_player(InputState(right=True))
    light_distance = float(world.player_pos[0] - origin[0])
    world.player_pos = origin.copy()
    runtime.equipment.collect('RIFLE', 30)
    runtime.equipment.collect('MEDKIT', 5)
    runtime.equipment.select('KNIFE')
    runtime._apply_player(InputState(right=True))
    assert 0 < float(world.player_pos[0] - origin[0]) < light_distance


@pytest.mark.parametrize('lethal', [False, True])
def test_lab_intel_is_recovered_before_corpse_loot_even_with_full_backpack(game, lethal):
    game.level = 3
    game.reset('routine')
    point = game.env._intel[2].copy()
    guard = game.aux_guards[0]
    guard.pos = point.copy()
    game._down_guard(guard.guard_id, lethal)
    game.equipment = full_backpack()
    game.equipment.hazmat = game.equipment.sample_vial = True
    game.equipment.casings = game.equipment.magazines = 0  # Ten occupied types including mission tools.
    assert game.equipment.slot_count == 10
    game.env.player_pos = point.copy()
    game._apply_player(InputState(interact=True))
    assert game.env.player_has_intel[2] and not game.equipment.sample_vial
    assert game.player_vitals.health > 0
    assert game.message == 'OBIETTIVO RECUPERATO'
    assert any(p['source'] == guard.guard_id for p in game.pickups)


def test_lab_uses_one_vial_and_preserves_a_spare(game):
    game.level = 3
    game.reset('routine')
    game.equipment.collect('HAZMAT', 1)
    game.equipment.collect('SAMPLE_VIAL', 2)
    game.env.player_pos = game.env._intel[2].copy()
    game._apply_player(InputState(interact=True))
    assert game.env.player_has_intel[2]
    assert game.equipment.sample_vial and game.equipment.item_quantity('SAMPLE_VIAL') == 1


def test_drop_input_survives_transport_and_is_cleared_after_one_tick():
    latest = InputState.from_dict({})
    latest.merge_transients(InputState.from_dict({'drop_item': True}))
    assert latest.drop_item
    latest.clear_transients()
    assert not latest.drop_item


def test_player_cannot_claim_a_guard_reserved_station_and_releases_cancelled_reservation():
    actions = PlayerActions()
    equipment = Equipment()
    vitals = PlayableVitals(health=.5)
    pos = np.zeros(2)
    item = dict(id='station', kind='MEDKIT', fixed=True, pos=pos, reserved_by='G1')
    inp = InputState(interact_held=True)
    actions.interaction(inp, pos, .3, equipment, vitals, [item], lambda a, b: True)
    assert actions.station_id is None and vitals.treatment is None
    item.pop('reserved_by')
    actions.interaction(inp, pos, .3, equipment, vitals, [item], lambda a, b: True)
    assert item['reserved_by'] == 'PLAYER' and actions.station_id == 'station'
    actions.interaction(InputState(), pos, .1, equipment, vitals, [item], lambda a, b: True)
    assert 'reserved_by' not in item and actions.station_id is None and vitals.treatment is None
