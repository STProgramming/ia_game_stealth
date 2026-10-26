import math
from types import SimpleNamespace

import numpy as np
import pytest

from shadow_ai.interactive_runtime import InputState
from shadow_ai.playable_actions import PlayerActions
from shadow_ai.playable_combat import EnemyMelee, MELEE_PROFILES, WEAPONS, player_shot, swat_loadout
from shadow_ai.playable_equipment import Equipment
from shadow_ai.playable_threats import MedicalSurveillance, blast_damage
from shadow_ai.playable_vitals import PlayableVitals
from shadow_ai.soliton_security import SecurityGuard
from test_interactive_tactics import game
from test_playable_actions import swat


def rng(roll=.99):
    return SimpleNamespace(random=lambda: roll,normal=lambda *args,**kwargs: np.zeros(kwargs['size']) if 'size' in kwargs else 0.)


@pytest.mark.parametrize('kind', MELEE_PROFILES)
def test_melee_damage_energy_and_probability_boundaries(kind):
    p = MELEE_PROFILES[kind]
    v = PlayableVitals()
    v.melee_trauma(p.damage,p.stamina_loss,p.injury_chance,p.fear_chance,rng(p.injury_chance))
    assert v.health == pytest.approx(1-p.damage)
    assert v.stamina == pytest.approx(1-p.stamina_loss)
    assert not v.injured
    v = PlayableVitals()
    v.melee_trauma(p.damage,p.stamina_loss,p.injury_chance,p.fear_chance,rng(0))
    assert v.injured
    assert v.fear == (p.fear_chance > 0)


def test_shield_has_longer_windup_and_heavier_effects_than_swat_and_guard():
    guard, swat, shield = [MELEE_PROFILES[k] for k in ['GUARD','SWAT','SHIELD']]
    for field in ['damage','stamina_loss','injury_chance','drop_chance','windup','energy']:
        assert getattr(guard,field) <= getattr(swat,field) < getattr(shield,field)
    assert shield.fear_chance > swat.fear_chance == guard.fear_chance == 0.


def test_injury_does_not_expire_bandages_do_not_cure_it_and_medkit_does():
    v = PlayableVitals(injured=True,bleeding=.2,health=.7)
    for _ in range(600):
        v.step(False,1.)
    assert v.injured and not v.can_sprint and v.speed_multiplier <= .45
    assert v.start_treatment('BANDAGE')
    v.step_treatment(1.)
    assert v.injured and not v.bleeding
    assert v.start_treatment('MEDKIT')
    v.step_treatment(3.9)
    assert v.injured
    v.step_treatment(.1)
    assert not v.injured and v.can_sprint and v.crouch_remaining == 60.


def test_injury_prevents_dive_and_only_allows_one_minute_crouched_in_total():
    v, actions = PlayableVitals(injured=True), PlayerActions()
    pos = np.array([4.,4.])
    assert np.array_equal(actions.dive(InputState(dive=True),pos,np.zeros((10,10)),v),pos)
    for _ in range(30):
        actions.condition_input(InputState(crouch=True),v,1.,rng(),pos)
    for _ in range(30):
        actions.condition_input(InputState(),v,1.,rng(),pos)
    assert v.crouch_remaining == 30.
    for _ in range(30):
        inp = InputState(crouch=True,sprint=True,dive=True)
        actions.condition_input(inp,v,1.,rng(),pos)
    assert not inp.crouch and not inp.sprint and not inp.dive and not v.can_crouch


@pytest.mark.parametrize('mode',['game','swat'])
def test_injury_limits_actual_runtime_movement_and_posture(request,mode):
    runtime = request.getfixturevalue(mode)
    world,grid = runtime._combat_world()
    grid[:] = 0
    world.player_pos = np.array([4.,4.],dtype=np.float32)
    runtime.player_vitals.injured = True
    runtime.player_vitals.crouch_remaining = 0.
    before = world.player_pos.copy()
    runtime._apply_player(InputState(right=True,sprint=True,dive=True,crouch=True))
    base_speed = 2.6*.94 if mode == 'game' else 1.7*.94
    assert world.player_pos[0]-before[0] == pytest.approx(base_speed*.45*runtime.game_dt,abs=1e-6)
    assert not runtime.player_actions.crouched and not runtime.player_actions.prone


def test_fear_inverts_controls_and_can_cause_uncommanded_action_or_surrender():
    v, actions = PlayableVitals(fear=True), PlayerActions()
    pos = np.array([4.,4.])
    inp = InputState(right=True,up=True)
    actions.condition_input(inp,v,.1,rng(),pos)
    assert inp.left and inp.down and not inp.up and not inp.right
    actions.panic_timer = 0
    inp = InputState()
    actions.condition_input(inp,v,.1,rng(.04),pos)
    assert inp.fire
    actions.panic_timer = 0
    inp = InputState(left=True)
    actions.condition_input(inp,v,.1,rng(0),pos,[np.array([6.,4.])])
    assert v.surrendering and inp.right and not inp.left
    actions.condition_input(inp,v,.1,rng(),np.array([5.5,4.]),[np.array([6.,4.])])
    assert inp.compliant and not np.linalg.norm(inp.direction())


@pytest.mark.parametrize('health,stamina,snack',[(.9,1.,True),(1.,.9,True),(1.,1.,False)])
def test_fear_requires_all_three_recovery_conditions(health,stamina,snack):
    v = PlayableVitals(fear=True,health=health,stamina=stamina,snack_eaten=snack)
    v.step_condition(False,10.)
    assert v.fear
    v.health = v.stamina = 1.
    v.eat_snack()
    v.step_condition(False,2.9)
    assert v.fear
    v.stamina = .9
    v.step_condition(False,.1)
    assert v.calm_seconds == 0.
    v.stamina = 1.
    v.step_condition(False,3.)
    assert not v.fear and not v.surrendering


@pytest.mark.parametrize('mode',['game','swat'])
def test_snack_can_be_collected_and_eaten_with_e(request,mode):
    runtime = request.getfixturevalue(mode)
    world,_ = runtime._combat_world()
    snack = next(i for i in runtime.pickups if i['kind']=='SNACK')
    world.player_pos = snack['pos'].copy()
    runtime._apply_player(InputState(interact=True))
    assert runtime.equipment.snacks == 2
    runtime._apply_player(InputState())
    runtime.equipment.select('SNACK')
    runtime.player_vitals.stamina = .5
    runtime._apply_player(InputState(interact=True))
    assert runtime.equipment.snacks == 1 and runtime.player_vitals.snack_eaten
    assert runtime.player_vitals.stamina >= .7


@pytest.mark.parametrize('shield',[False,True])
@pytest.mark.parametrize('mode',['game','swat'])
def test_fresh_knife_press_counters_on_impact_frame_except_shield(request,mode,shield):
    runtime = request.getfixturevalue(mode)
    world,grid = runtime._combat_world()
    grid[:] = 0
    world.player_pos = np.array([4.,4.],dtype=np.float32)
    if mode == 'game':
        runtime.lead_alive = False
        enemy = SecurityGuard('S1' if shield else 'G1',np.array([4.5,4.],dtype=np.float32),math.pi,((4,4),),has_shield=shield)
        runtime.aux_guards = [enemy]
        runtime.swat_units = []
        uid = enemy.guard_id
        combat = lambda: runtime._step_guard_weapon(enemy,True)
    else:
        enemy = next(u for u in world.units.values() if u.profile.has_shield == shield)
        for u in world.units.values():
            u.alive = u is enemy
        enemy.pos,enemy.angle = np.array([4.5,4.],dtype=np.float32),math.pi
        uid = enemy.uid
        combat = runtime._world_after_ai
    runtime.rng = rng()
    runtime._melee_attacks[uid] = EnemyMelee(windup=runtime.game_dt)
    ammo = enemy.ammo
    inp = InputState(attack_held=True)
    runtime._apply_player(inp)
    if mode == 'game':
        runtime._process_takedown(inp)
    if shield:
        assert not enemy.disarmed and enemy.ammo == ammo
        assert runtime._melee_attacks[uid].windup > 0
        combat()
        assert runtime.player_actions.knockdown_remaining > 0
    else:
        assert enemy.disarmed and enemy.ammo == 0 and enemy.weapon is None
        assert runtime._melee_attacks[uid].windup == 0
        combat()
        assert not runtime.player_actions.prone
        weapon = next(i for i in runtime.pickups if i['id'].startswith('disarm-'))
        assert weapon['amount']==ammo


@pytest.mark.parametrize('mode',['game','swat'])
def test_diving_evades_the_impact_even_if_it_is_due_this_frame(request,mode):
    runtime = request.getfixturevalue(mode)
    world,grid = runtime._combat_world()
    grid[:] = 0
    world.player_pos = np.array([4.,4.],dtype=np.float32)
    if mode == 'game':
        runtime.lead_alive = False
        enemy = SecurityGuard('G1',np.array([4.5,4.],dtype=np.float32),math.pi,((4,4),))
        runtime.aux_guards = [enemy]
        uid = enemy.guard_id
        combat = lambda: runtime._step_guard_weapon(enemy,True)
    else:
        enemy = next(iter(world.units.values()))
        for u in world.units.values():
            u.alive = u is enemy
        enemy.pos,enemy.angle = np.array([4.5,4.],dtype=np.float32),math.pi
        uid = enemy.uid
        combat = runtime._world_after_ai
    enemy.ammo = 0
    runtime._melee_attacks[uid] = EnemyMelee(windup=runtime.game_dt)
    runtime._apply_player(InputState(dive=True,left=True))
    combat()
    assert runtime.player_actions.dive_timer > 0 and runtime.player_actions.knockdown_remaining == 0
    assert runtime.player_vitals.health == 1.


def test_injury_and_fear_survive_mission_transition(game):
    game.level,game.status = 1,'ESCAPED'
    game.player_vitals.injured = game.player_vitals.fear = True
    game.player_vitals.crouch_remaining = 12.
    game.player_vitals.health = .6
    assert game.advance_level()
    assert game.player_vitals.injured and game.player_vitals.fear
    assert game.player_vitals.crouch_remaining == 12. and game.player_vitals.health == .6


def test_medical_evidence_requires_a_wounded_player_lost_sight_and_local_inspection():
    station = dict(id='med',kind='MEDKIT',fixed=True,pos=np.array([8.,4.]))
    watch = MedicalSurveillance([station])
    distance = lambda a,b: np.linalg.norm(b-a)
    pos = np.array([4.,4.])
    assert watch.goal('G1',pos,False,distance) is None
    watch.hit('G1')
    assert watch.goal('G1',pos,True,distance) is None
    assert np.array_equal(watch.goal('G1',pos,False,distance),station['pos'])
    watch.used.add('med')
    assert watch.inspect('G1',lambda p: False) is None
    assert np.array_equal(watch.inspect('G1',lambda p: True),station['pos'])
    assert watch.inspect('G1',lambda p: True) is None


@pytest.mark.parametrize('mode',['game','swat'])
def test_runtime_medkit_watch_travels_to_the_station_without_live_player_knowledge(request,mode):
    runtime = request.getfixturevalue(mode)
    world,grid = runtime._combat_world()
    grid[:] = 0
    world.player_pos = np.array([25.,18.],dtype=np.float32)
    station = dict(id='medical-test',kind='MEDKIT',fixed=True,pos=np.array([8.,4.],dtype=np.float32))
    runtime.medical_watch = MedicalSurveillance([station])
    if mode == 'game':
        runtime.lead_alive = False
        enemy = SecurityGuard('G1',np.array([4.,4.],dtype=np.float32),0.,((4,4),))
        runtime.aux_guards = [enemy]
        runtime.medical_watch.hit(enemy.guard_id)
        runtime._step_guard(enemy)
        assert enemy.state == 'WATCH MEDKIT'
        assert np.array_equal(enemy.target,station['pos'])
    else:
        enemy = next(iter(world.units.values()))
        enemy.pos,enemy.angle,enemy.local_visual = np.array([4.,4.],dtype=np.float32),0.,False
        runtime.medical_watch.hit(enemy.uid)
        assert runtime._step_swat_medical_watch(enemy.uid)
        assert runtime.actions[enemy.uid] == 'WATCH MEDKIT'
    assert enemy.pos[0] > 4.
    runtime.medical_watch.used.add(station['id'])
    if mode == 'game':
        runtime._step_guard(enemy)
        assert 'passato' in runtime.guard_dialogue[enemy.guard_id]
        assert runtime.team_contact is None
    else:
        runtime._step_swat_medical_watch(enemy.uid)
        assert 'TRACCE' in runtime.message and world.blackboard.contact_pos is None


@pytest.mark.parametrize('mode',['game','swat'])
def test_limping_is_audible_while_crouched(request,mode):
    runtime = request.getfixturevalue(mode)
    world,grid = runtime._combat_world()
    grid[:] = 0
    runtime.player_vitals.injured = True
    world.player_pos = np.array([4.,4.],dtype=np.float32)
    world.player_velocity = np.array([.3,0.],dtype=np.float32)
    if mode == 'game':
        world.player_current_speed = .3
        from shadow_ai import PlayerStance
        world.player_stance = PlayerStance.CROUCHED
        cue = runtime._audible_cue(np.array([7.,4.]))
        assert cue[1]=='LIMPING'
        assert runtime._audible_cue(np.array([10.,4.])) is None
    else:
        enemy = next(iter(world.units.values()))
        enemy.pos,enemy.angle = np.array([7.,4.],dtype=np.float32),0.
        runtime._sense_swat_units()
        assert not enemy.local_visual and enemy.local_sound


def test_swat_loadouts_include_shotgun_smg_and_both_shield_options(swat):
    assert {u.weapon for u in swat.core.units.values()} >= {'SHOTGUN','SMG'}
    assert {swat_loadout('SHIELD',i) for i in range(2)} == {'PISTOL','ONE_HAND_SMG'}
    assert WEAPONS['ONE_HAND_SMG'].spread > WEAPONS['SMG'].spread
    assert WEAPONS['SHOTGUN'].damage > WEAPONS['SMG'].damage
    assert WEAPONS['SHOTGUN'].cooldown > WEAPONS['SMG'].cooldown


@pytest.mark.parametrize('weapon',['SHOTGUN','SMG','ONE_HAND_SMG'])
def test_new_weapons_can_be_looted_equipped_and_fire_with_limited_reach(weapon):
    eq = Equipment()
    assert eq.collect(weapon,5) and eq.armed and eq.select(weapon)
    grid = np.zeros((30,30),dtype=np.uint8)
    end,hit = player_shot(grid,np.array([4.,4.]),np.array([28.,4.]),[],weapon,rng())
    assert np.linalg.norm(end-np.array([4.,4.])) <= WEAPONS[weapon].reach+1e-6
    assert hit is None


@pytest.mark.parametrize('mode',['game','swat'])
def test_grenade_fuse_and_walls_limit_real_damage(request,mode):
    runtime = request.getfixturevalue(mode)
    world,grid = runtime._combat_world()
    grid[:] = 0
    world.player_pos = np.array([6.,4.],dtype=np.float32)
    runtime._throw_grenade(np.array([2.,4.]),np.array([5.,4.]),'G9')
    for _ in range(44):
        runtime._step_explosives()
    assert runtime.player_vitals.health == 1 and len(runtime.explosives)==1
    grid[:,5]=1
    runtime._step_explosives()
    assert runtime.player_vitals.health == 1 and not runtime.explosives
    grid[:] = 0
    runtime._throw_grenade(np.array([2.,4.]),np.array([5.,4.]),'G9')
    for _ in range(45):
        runtime._step_explosives()
    assert runtime.player_vitals.health == pytest.approx(.7)
    assert 'G9' in runtime.medical_watch.wounded_by


def test_blast_damage_falls_with_distance_and_never_crosses_occlusion():
    center = np.array([4.,4.])
    assert blast_damage(center,center,lambda a,b: True)==.45
    assert blast_damage(center,np.array([7.,4.]),lambda a,b: True)==0
    assert blast_damage(center,np.array([5.,4.]),lambda a,b: False)==0


@pytest.mark.parametrize('mode',['game','swat'])
def test_partial_medical_station_use_leaves_evidence_after_interruption(request,mode):
    runtime = request.getfixturevalue(mode)
    world,_ = runtime._combat_world()
    station = next(i for i in runtime.pickups if i.get('fixed'))
    world.player_pos = station['pos'].copy()
    runtime.player_vitals.health = .6
    if mode == 'swat':
        world.player_health = .6
    for _ in range(10):
        runtime._apply_player(InputState(interact_held=True))
    assert runtime.player_vitals.treatment == 'MEDKIT'
    assert station['id'] in runtime.medical_watch.used
    runtime._apply_player(InputState(right=True))
    assert runtime.player_vitals.treatment is None
    assert station in runtime.pickups
    assert station['id'] in runtime.medical_watch.used


@pytest.mark.parametrize('mode',['game','swat'])
def test_enemy_grenade_requires_distance_and_keeps_teammates_out_of_the_blast(request,mode):
    runtime = request.getfixturevalue(mode)
    world,grid = runtime._combat_world()
    grid[:] = 0
    world.player_pos = np.array([10.,4.],dtype=np.float32)
    if mode == 'game':
        runtime.lead_alive = False
        shooter = SecurityGuard('S1',np.array([4.,4.],dtype=np.float32),0.,((4,4),),grenade_count=1)
        helper = SecurityGuard('S2',np.array([9.,4.],dtype=np.float32),0.,((9,4),))
        runtime.aux_guards = [shooter,helper]
        uid,vitals = shooter.guard_id,shooter.vitals
    else:
        shooter,helper = list(world.units.values())[:2]
        for u in world.units.values():
            u.alive = u in [shooter,helper]
        shooter.pos,helper.pos = np.array([4.,4.],dtype=np.float32),np.array([9.,4.],dtype=np.float32)
        shooter.grenade_count = 1
        uid,vitals = shooter.uid,runtime._unit_vitals[shooter.uid]
    assert not runtime._enemy_grenade(uid,shooter,vitals)
    assert shooter.grenade_count==1 and not runtime.explosives
    helper.pos = np.array([4.,9.],dtype=np.float32)
    assert runtime._enemy_grenade(uid,shooter,vitals)
    assert shooter.grenade_count==0 and len(runtime.explosives)==1
    assert vitals.stamina==pytest.approx(1-.3/vitals.stamina_capacity)


def test_a_fatal_grenade_marks_standalone_player_neutralized_immediately(swat):
    swat.core.map.grid[:] = 0
    swat.core.player_pos = np.array([4.,4.],dtype=np.float32)
    swat.player_vitals.health = swat.core.player_health = .2
    swat._throw_grenade(np.array([6.,4.]),swat.core.player_pos,'PLAYER')
    for _ in range(45):
        swat._step_explosives()
    assert swat.core.player_health == swat.player_vitals.health == 0
    assert swat.core.player_neutralized
