from __future__ import annotations

import argparse
from pathlib import Path

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


def make_robust_doctrine_env():
    return Monitor(DoctrinePretrainEnv(robust_context=True))


def doctrine_accuracy(
    model: PPO,
    *,
    robust: bool,
    episodes_per_skill: int,
) -> float:
    correct = 0
    total = 0

    for skill in SKILLS:
        env = DoctrinePretrainEnv(
            forced_skill=skill.name,
            robust_context=robust,
        )
        for ep in range(episodes_per_skill):
            obs, _ = env.reset(seed=120_000 + ep)
            action, _ = model.predict(obs, deterministic=True)
            _, _, _, _, info = env.step(int(action))
            correct += int(bool(info["doctrine_correct"]))
            total += 1

    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "V6.3.2 robustness stabilization: full-world training alternated "
            "with doctrine rehearsal containing realistic nuisance context."
        )
    )
    parser.add_argument(
        "--init-model",
        type=Path,
        required=True,
    )
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--full-steps", type=int, default=6000)
    parser.add_argument("--doctrine-steps", type=int, default=4000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models_v6_3_2"),
    )
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument(
        "--doctrine-eval-per-skill",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--min-clean-doctrine",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--min-robust-doctrine",
        type=float,
        default=0.92,
    )
    args = parser.parse_args()

    for folder in [
        args.out,
        args.out / "cycles",
        args.out / "best_full",
        args.out / "best_doctrine",
        args.out / "best_stable",
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    full_train = DummyVecEnv([make_full_env])
    doctrine_train = DummyVecEnv([make_robust_doctrine_env])

    model = PPO.load(
        args.init_model,
        env=full_train,
        device="cpu",
    )
    model.verbose = 1
    model.tensorboard_log = str(args.out / "tensorboard")

    best_full = -float("inf")
    best_robust = -float("inf")
    best_stable_full = -float("inf")

    total = args.full_steps + args.doctrine_steps
    print("\n=== V6.3.2 ROBUST STABILIZATION ===")
    print(
        f"Per cycle: full={args.full_steps}, "
        f"robust_doctrine={args.doctrine_steps}"
    )
    print(
        "Robust doctrine ratio by actual PPO steps: "
        f"{args.doctrine_steps / max(total, 1):.1%}"
    )

    for cycle in range(1, args.cycles + 1):
        print(f"\n========== CYCLE {cycle}/{args.cycles} ==========")

        model.set_env(full_train)
        print("\n--- FULL WORLD ---")
        model.learn(
            total_timesteps=args.full_steps,
            reset_num_timesteps=False,
            progress_bar=True,
        )

        model.set_env(doctrine_train)
        print("\n--- ROBUST DOCTRINE REHEARSAL ---")
        model.learn(
            total_timesteps=args.doctrine_steps,
            reset_num_timesteps=False,
            progress_bar=True,
        )

        full_eval = ShadowProtocolEnv()
        full_mean, full_std = evaluate_policy(
            model,
            full_eval,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            warn=False,
        )
        full_eval.close()

        clean_acc = doctrine_accuracy(
            model,
            robust=False,
            episodes_per_skill=args.doctrine_eval_per_skill,
        )
        robust_acc = doctrine_accuracy(
            model,
            robust=True,
            episodes_per_skill=args.doctrine_eval_per_skill,
        )

        print(
            f"\ncycle={cycle} "
            f"full_reward={full_mean:.3f} +/- {full_std:.3f} "
            f"clean_doctrine={clean_acc * 100:.2f}% "
            f"robust_doctrine={robust_acc * 100:.2f}%"
        )

        model.save(args.out / "cycles" / f"cycle_{cycle:02d}")

        if full_mean > best_full:
            best_full = float(full_mean)
            model.save(args.out / "best_full" / "best_model")
            print("New best FULL model.")

        if robust_acc > best_robust:
            best_robust = float(robust_acc)
            model.save(args.out / "best_doctrine" / "best_model")
            print("New best ROBUST-DOCTRINE model.")

        stable = (
            clean_acc >= args.min_clean_doctrine
            and robust_acc >= args.min_robust_doctrine
        )
        if stable and full_mean > best_stable_full:
            best_stable_full = float(full_mean)
            model.save(args.out / "best_stable" / "best_model")
            print("New best STABLE model.")

    model.save(args.out / "shadow_guard_v6_3_2")

    print("\n=== DONE ===")
    print(
        "Recommended checkpoint: "
        f"{args.out / 'best_stable' / 'best_model.zip'}"
    )
    print(
        f"best_full_reward={best_full:.3f} "
        f"best_robust_doctrine={best_robust * 100:.2f}% "
        f"best_stable_full_reward={best_stable_full:.3f}"
    )


if __name__ == "__main__":
    main()
