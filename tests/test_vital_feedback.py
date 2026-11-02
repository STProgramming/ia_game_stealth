import pytest
from shadow_ai.playable_vitals import PlayableVitals
from test_firearms_runtime import runtime
from shadow_ai.interactive_runtime import InputState


def step(vitals, seconds, **kwargs):
    for _ in range(round(seconds*30)):
        vitals.step(False,1/30,**kwargs)


def test_snack_reports_the_actual_instant_vigour_gain_and_does_not_heal():
    vitals=PlayableVitals(health=.7,stamina=.9)
    assert vitals.eat_snack()
    state=vitals.snapshot()
    assert state['stamina']==1 and state['health']==.7
    assert state['snack_stamina_gain']==pytest.approx(.1)
    assert state['snack_effect_remaining']==6
    step(vitals,6)
    assert vitals.snack_effect_remaining==pytest.approx(0,abs=1e-7)


def test_stillness_accelerates_recovery_and_walking_resets_the_bonus():
    resting=PlayableVitals(stamina=.1)
    walking=PlayableVitals(stamina=.1)
    step(resting,3)
    step(walking,3,moving=True)
    assert resting.stamina>walking.stamina
    assert resting.snapshot()['stamina_recovery_multiplier']>1
    resting.step(False,.1,moving=True)
    assert resting.still_seconds==0


def test_exhaustion_recovery_stays_at_normal_rate_even_after_snack():
    vitals=PlayableVitals(stamina=0,exhausted=True)
    step(vitals,2)
    assert vitals.stamina==pytest.approx(.22)
    assert vitals.eat_snack()
    before=vitals.stamina
    step(vitals,2)
    assert vitals.stamina-before==pytest.approx(.22)
    assert vitals.exhausted_recovery
    step(vitals,5)
    assert vitals.stamina==1 and not vitals.exhausted_recovery


def test_superficial_wound_clots_and_heals_gradually_after_a_delay():
    vitals=PlayableVitals()
    vitals.take_hit(.1)
    step(vitals,29)
    assert vitals.health==.9
    step(vitals,31)
    assert .9<vitals.health<1 and vitals.bleeding==0
    step(vitals,45)
    assert vitals.health==1 and vitals.wound_damage==0


@pytest.mark.parametrize('malus',[dict(injured=True),dict(fear=True),dict(exhausted=True,stamina=0),dict(smoke_exposure=.5),dict(fatigue=.1)])
def test_malus_blocks_superficial_healing(malus):
    vitals=PlayableVitals(**malus)
    vitals.take_hit(.1)
    # Test the wound gate directly so long rest does not clear the malus.
    vitals._step_superficial_wound(120,False)
    assert vitals.health==.9 and not vitals.superficial_healing


def test_deep_wounds_do_not_heal_spontaneously():
    vitals=PlayableVitals()
    vitals.take_hit(.3)
    step(vitals,180)
    assert vitals.health==.7


def test_minor_bleeding_does_not_permanently_block_its_own_natural_healing():
    vitals=PlayableVitals()
    vitals.take_hit(.1)
    for _ in range(3600):
        vitals.step(False,1/30)
        vitals.step_bleeding(1/30,0.)
    assert vitals.health==1 and vitals.bleeding==0


def test_shared_rule_is_applied_to_enemy_vitals_in_both_modes(runtime):
    if hasattr(runtime,'env'):
        guard=runtime._lead_patrol
        guard.vitals.take_hit(.08)
        guard.vitals.wound_rest_seconds=31
        before=guard.vitals.health
        runtime._guard_move(guard,None,0.)
        assert guard.vitals.health>before
    else:
        unit=next(iter(runtime.core.units.values()))
        vitals=runtime._unit_vitals[unit.uid]
        vitals.take_hit(.08); unit.health=vitals.health
        vitals.wound_rest_seconds=31
        runtime._unit_vitals_stepped.clear()
        runtime._world_after_ai()
        assert unit.health>.92
