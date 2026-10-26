from __future__ import annotations
import argparse,time
from pathlib import Path
from stable_baselines3 import PPO
from shadow_ai import ShadowProtocolEnv,TacticalAction

def main():
    p=argparse.ArgumentParser(); p.add_argument("model",type=Path); p.add_argument("--scenario",choices=["routine","sound","visual","lost_contact","arrest","theft"],default="visual"); p.add_argument("--steps",type=int,default=160); a=p.parse_args()
    env=ShadowProtocolEnv(render_mode="ansi"); model=PPO.load(a.model,device="cpu"); obs,info=env.reset(seed=888,options={"scenario":a.scenario})
    for _ in range(a.steps):
        act,_=model.predict(obs,deterministic=True); ai=int(act); obs,r,term,trunc,info=env.step(ai)
        print("\033[2J\033[H",end=""); print(env.render()); print()
        print(f"scenario : {info['training_scenario']}")
        print(f"phase    : {info['tactical_phase']}")
        print(f"action   : {TacticalAction(ai).name}")
        print(f"reward   : {r:.3f}")
        print(f"alert    : {info['alert']}")
        print(f"visual   : {info['visual_contact']}")
        print(f"sound    : {info['sound_contact']}")
        print(f"memory   : {info['last_seen_age']}")
        print(f"arrest   : {info['arrest_progress']:.2f}/1.00")
        print(f"intel    : {info['intel_known_present']}")
        print(f"reinforce: {info['reinforcements_called']}")
        if term or trunc: print("\nEpisode ended."); break
        time.sleep(.12)

if __name__=="__main__": main()
