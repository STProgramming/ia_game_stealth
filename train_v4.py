from __future__ import annotations

import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from shadow_ai import ShadowProtocolEnv

def make_env():
    return Monitor(ShadowProtocolEnv())

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--timesteps",type=int,default=150_000)
    p.add_argument("--seed",type=int,default=45)
    p.add_argument("--out",type=Path,default=Path("models_v4"))
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    check_env(ShadowProtocolEnv(seed=a.seed),warn=True)
    train_env=DummyVecEnv([make_env]); eval_env=DummyVecEnv([make_env])
    callbacks=[
        CheckpointCallback(save_freq=50_000,save_path=str(a.out/"checkpoints"),name_prefix="shadow_guard_v4"),
        EvalCallback(eval_env,best_model_save_path=str(a.out/"best"),log_path=str(a.out/"eval"),eval_freq=25_000,n_eval_episodes=80,deterministic=True),
    ]
    model=PPO("MlpPolicy",train_env,verbose=1,seed=a.seed,learning_rate=2.5e-4,n_steps=2048,batch_size=256,gamma=0.995,gae_lambda=0.95,ent_coef=0.02,clip_range=0.20,policy_kwargs=dict(net_arch=dict(pi=[256,256],vf=[256,256])),tensorboard_log=str(a.out/"tensorboard"))
    model.learn(total_timesteps=a.timesteps,callback=callbacks,progress_bar=True)
    path=a.out/"shadow_guard_v4"; model.save(path); print(f"Saved: {path}.zip")

if __name__=="__main__": main()
