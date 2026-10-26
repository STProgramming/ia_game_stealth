from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from shadow_ai import (
    DoctrinePretrainEnv,
    SKILLS,
    ShadowProtocolEnv,
)


def make_full_env():
    return Monitor(ShadowProtocolEnv())


def make_doctrine_env():
    return Monitor(DoctrinePretrainEnv())


def doctrine_accuracy(
    model: PPO,
    episodes_per_skill: int = 20,
) -> float:
    correct = 0
    total = 0

    for skill in SKILLS:
        env = DoctrinePretrainEnv(forced_skill=skill.name)

        for ep in range(episodes_per_skill):
            obs, _ = env.reset(seed=91_000 + ep)
            action, _ = model.predict(obs, deterministic=True)
            _, _, _, _, info = env.step(int(action))
            correct += int(bool(info["doctrine_correct"]))
            total += 1

    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "V6.3 stabilization: alternate full-world and doctrine "
            "training with a true step-level 70/30 ratio."
        )
    )
    parser.add_argument(
        "--init-model",
        type=Path,
        required=True,
        help=(
            "Compatible V6/V6.2 model. Recommended: "
            "models_v6_2\\finetune_best\\best_model.zip"
        ),
    )
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--full-steps", type=int, default=7000)
    parser.add_argument("--doctrine-steps", type=int, default=3000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models_v6_3"),
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--doctrine-eval-per-skill",
        type=int,
        default=20,
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cycles").mkdir(parents=True, exist_ok=True)
    (args.out / "best_full").mkdir(parents=True, exist_ok=True)
    (args.out / "best_doctrine").mkdir(parents=True, exist_ok=True)
    (args.out / "best_stable").mkdir(parents=True, exist_ok=True)

    full_train = DummyVecEnv([make_full_env])
    doctrine_train = DummyVecEnv([make_doctrine_env])

    model = PPO.load(
        args.init_model,
        env=full_train,
        device="cpu",
    )
    model.verbose = 1
    model.tensorboard_log = str(args.out / "tensorboard")

    best_full = -float("inf")
    best_doctrine = -float("inf")
    best_stable_full = -float("inf")

    print("\n=== V6.3 STABILIZATION ===")
    print(
        f"Each cycle: {args.full_steps} full steps + "
        f"{args.doctrine_steps} doctrine steps"
    )
    ratio = args.doctrine_steps / max(
        args.full_steps + args.doctrine_steps,
        1,
    )
    print(f"Doctrine rehearsal ratio by steps: {ratio:.1%}")

    for cycle in range(1, args.cycles + 1):
        print(f"\n========== CYCLE {cycle}/{args.cycles} ==========")

        print("\n--- FULL WORLD ---")
        model.set_env(full_train)
        model.learn(
            total_timesteps=args.full_steps,
            reset_num_timesteps=False,
            progress_bar=True,
        )

        print("\n--- DOCTRINE REHEARSAL ---")
        model.set_env(doctrine_train)
        model.learn(
            total_timesteps=args.doctrine_steps,
            reset_num_timesteps=False,
            progress_bar=True,
        )

        # Evaluate only after both halves of the cycle.
        full_eval = ShadowProtocolEnv()
        mean_reward, std_reward = evaluate_policy(
            model,
            full_eval,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            warn=False,
        )
        full_eval.close()

        d_acc = doctrine_accuracy(
            model,
            episodes_per_skill=args.doctrine_eval_per_skill,
        )

        print(
            f"\ncycle={cycle} "
            f"full_reward={mean_reward:.3f} +/- {std_reward:.3f} "
            f"doctrine_accuracy={d_acc * 100:.2f}%"
        )

        cycle_path = args.out / "cycles" / f"cycle_{cycle:02d}"
        model.save(cycle_path)

        if mean_reward > best_full:
            best_full = float(mean_reward)
            model.save(args.out / "best_full" / "best_model")
            print("New best FULL model.")

        if d_acc > best_doctrine:
            best_doctrine = float(d_acc)
            model.save(args.out / "best_doctrine" / "best_model")
            print("New best DOCTRINE model.")

        # "Stable" means doctrine remains >= 90%; among those checkpoints
        # prefer the highest full-world reward.
        if d_acc >= 0.90 and mean_reward > best_stable_full:
            best_stable_full = float(mean_reward)
            model.save(args.out / "best_stable" / "best_model")
            print("New best STABLE model (doctrine >= 90%).")

    final_path = args.out / "shadow_guard_v6_3"
    model.save(final_path)

    print("\n=== DONE ===")
    print(f"Final: {final_path}.zip")
    print(
        "Recommended checkpoint: "
        f"{args.out / 'best_stable' / 'best_model.zip'}"
    )
    print(
        f"best_full_reward={best_full:.3f} "
        f"best_doctrine_accuracy={best_doctrine * 100:.2f}% "
        f"best_stable_full_reward={best_stable_full:.3f}"
    )


if __name__ == "__main__":
    main()
