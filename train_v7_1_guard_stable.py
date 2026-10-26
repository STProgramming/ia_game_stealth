
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from shadow_ai import BalancedGuardEnv, DoctrinePretrainEnv


def make_balanced():
    return Monitor(BalancedGuardEnv())


def make_doctrine():
    return Monitor(DoctrinePretrainEnv(robust_context=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--init-model",
        type=Path,
        default=Path("models_v6_3_2/best_stable/best_model.zip"),
    )
    p.add_argument("--cycles", type=int, default=6)
    p.add_argument("--world-steps", type=int, default=6000)
    p.add_argument("--doctrine-steps", type=int, default=3000)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("models_v7_1_guard"),
    )
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    world = DummyVecEnv([make_balanced])
    doctrine = DummyVecEnv([make_doctrine])

    model = PPO.load(args.init_model, env=world, device="cpu")
    model.verbose = 1
    model.tensorboard_log = str(args.out / "tensorboard")

    for cycle in range(1, args.cycles + 1):
        print(f"\n=== GUARD FINAL CYCLE {cycle}/{args.cycles} ===")
        model.set_env(world)
        model.learn(
            total_timesteps=args.world_steps,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        model.set_env(doctrine)
        model.learn(
            total_timesteps=args.doctrine_steps,
            reset_num_timesteps=False,
            progress_bar=True,
        )
        model.save(args.out / f"cycle_{cycle:02d}")

    model.save(args.out / "guard_v7_1_final")
    print("Saved:", args.out / "guard_v7_1_final.zip")


if __name__ == "__main__":
    main()
