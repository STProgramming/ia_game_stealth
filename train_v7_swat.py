
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from shadow_ai import (
    SwatDoctrinePretrainEnv,
    SwatSharedPolicyEnv,
)


def make_doctrine():
    return Monitor(SwatDoctrinePretrainEnv(robust_context=True))


def make_team():
    return Monitor(SwatSharedPolicyEnv())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain", type=int, default=100_000)
    p.add_argument("--team", type=int, default=200_000)
    p.add_argument("--cycles", type=int, default=6)
    p.add_argument("--rehearsal", type=int, default=20_000)
    p.add_argument("--out", type=Path, default=Path("models_v7_swat"))
    p.add_argument("--seed", type=int, default=700)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    doctrine = DummyVecEnv([make_doctrine])
    team = DummyVecEnv([make_team])

    model = PPO(
        "MlpPolicy",
        doctrine,
        verbose=1,
        seed=args.seed,
        learning_rate=2.0e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.992,
        gae_lambda=0.95,
        ent_coef=0.02,
        clip_range=0.20,
        policy_kwargs=dict(
            net_arch=dict(pi=[384, 256], vf=[384, 256])
        ),
        tensorboard_log=str(args.out / "tensorboard"),
        device="cpu",
    )

    print("\n=== STAGE 1: SWAT ROLE DOCTRINE ===")
    model.learn(
        total_timesteps=args.pretrain,
        progress_bar=True,
    )
    model.save(args.out / "swat_doctrine_pretrained")

    per_cycle_team = max(1, args.team // args.cycles)
    best_marker = -1e9

    for cycle in range(1, args.cycles + 1):
        print(f"\n=== TEAM CYCLE {cycle}/{args.cycles} ===")
        model.set_env(team)
        model.learn(
            total_timesteps=per_cycle_team,
            reset_num_timesteps=False,
            progress_bar=True,
        )

        print("\n--- doctrine rehearsal ---")
        model.set_env(doctrine)
        model.learn(
            total_timesteps=args.rehearsal,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        model.save(args.out / f"swat_cycle_{cycle:02d}")

    model.save(args.out / "swat_shared_policy_v7")
    print("\nSaved:", args.out / "swat_shared_policy_v7.zip")


if __name__ == "__main__":
    main()
