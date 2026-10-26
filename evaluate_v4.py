from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from stable_baselines3 import PPO
from shadow_ai import ShadowProtocolEnv, TacticalAction

SCENARIOS=["routine","sound","visual","lost_contact","arrest","theft"]

def main():
    p=argparse.ArgumentParser(); p.add_argument("model",type=Path); p.add_argument("--episodes",type=int,default=50); a=p.parse_args()
    model=PPO.load(a.model,device="cpu")
    global_phases=defaultdict(Counter)
    for scenario in SCENARIOS:
        env=ShadowProtocolEnv(); actions=Counter(); phases=defaultdict(Counter); captures=escapes=truncs=0; total_reward=0.0
        for ep in range(a.episodes):
            obs,info=env.reset(seed=20_000+ep,options={"scenario":scenario}); er=0.0
            while True:
                phase=info["tactical_phase"]
                act,_=model.predict(obs,deterministic=True); ai=int(act); name=TacticalAction(ai).name
                actions[name]+=1; phases[phase][name]+=1; global_phases[phase][name]+=1
                obs,r,term,trunc,info=env.step(ai); er+=r
                if term or trunc:
                    if info["terminal_reward"]>0: captures+=1
                    elif info["terminal_reward"]<0: escapes+=1
                    else: truncs+=1
                    break
            total_reward+=er
        print(f"\n=== SCENARIO {scenario.upper()} ===")
        print(f"episodes={a.episodes}\nmean_reward={total_reward/a.episodes:.3f}\ncaptures={captures}\nescapes={escapes}\ntruncations={truncs}")
        total=sum(actions.values()) or 1
        for n,c in actions.most_common(): print(f"{n:<24} {c:>8} {c/total*100:6.2f}%")
        print("-- actual phases --")
        for ph in sorted(phases):
            pt=sum(phases[ph].values()) or 1
            print(f"{ph:<12} "+", ".join(f"{n}={c/pt*100:.1f}%" for n,c in phases[ph].most_common(5)))
    print("\n=== GLOBAL BY ACTUAL TACTICAL PHASE ===")
    for ph in sorted(global_phases):
        total=sum(global_phases[ph].values()) or 1
        print(f"\n[{ph}] n={total}")
        for n,c in global_phases[ph].most_common(): print(f"  {n:<24} {c:>8} {c/total*100:6.2f}%")

if __name__=="__main__": main()
