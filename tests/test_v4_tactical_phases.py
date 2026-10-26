from shadow_ai import ShadowProtocolEnv, TacticalAction

def test_pursue_does_not_arrest_compliant_player():
    env=ShadowProtocolEnv(seed=1); env.reset(seed=1,options={"scenario":"arrest"})
    env.player_compliant=True; env.sense=env._sense(); env._update_knowledge(); d=env._decision_context()
    reward,term=env._terminal_events(TacticalAction.PURSUE,d)
    assert not term and reward==0.0

def test_arrest_requires_sustained_action():
    env=ShadowProtocolEnv(seed=2); env.reset(seed=2,options={"scenario":"arrest"})
    env.player_compliant=True; env.sense=env._sense(); env._update_knowledge(); d=env._decision_context()
    assert d["visual"]
    for _ in range(4):
        reward,term=env._terminal_events(TacticalAction.ARREST,d); assert not term
    reward,term=env._terminal_events(TacticalAction.ARREST,d); assert term and reward>0

def test_sound_reward_prefers_investigate():
    env=ShadowProtocolEnv(seed=3); env.reset(seed=3,options={"scenario":"sound"})
    env.sense.visual=False; env.sense.sound=True; env.sense.sound_rel=(-3.0,0.0); env.sense.sound_confidence=.8
    d=env._decision_context(); prev=env.guard_pos.copy()
    assert env._reward(TacticalAction.INVESTIGATE_SOUND,prev,d) > env._reward(TacticalAction.PATROL,prev,d)

def test_last_seen_prefers_go_last_seen_over_pursue():
    env=ShadowProtocolEnv(seed=4); env.reset(seed=4,options={"scenario":"lost_contact"})
    env.last_seen_pos=env.guard_pos.copy(); env.last_seen_pos[0]+=3.0; env.last_seen_age=1.0; env.sense.visual=False; env.sense.sound=False; env.alert=2
    d=env._decision_context(); prev=env.guard_pos.copy()
    assert env._reward(TacticalAction.GO_LAST_SEEN,prev,d) > env._reward(TacticalAction.PURSUE,prev,d)

def test_theft_rewards_reinforcement_call():
    env=ShadowProtocolEnv(seed=5); env.reset(seed=5,options={"scenario":"theft"})
    env.sense.visual=False; env.sense.sound=False; d=env._decision_context(); prev=env.guard_pos.copy()
    assert env._reward(TacticalAction.CALL_REINFORCEMENTS,prev,d) > env._reward(TacticalAction.PATROL,prev,d)
