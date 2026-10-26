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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    raw_env = ShadowProtocolEnv(seed=args.seed)
    check_env(raw_env, warn=True)

    train_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    checkpoint = CheckpointCallback(
        save_freq=50_000,
        save_path=str(args.out / "checkpoints"),
        name_prefix="shadow_guard_ppo",
    )
    evaluator = EvalCallback(
        eval_env,
        best_model_save_path=str(args.out / "best"),
        log_path=str(args.out / "eval"),
        eval_freq=20_000,
        n_eval_episodes=20,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=args.seed,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.012,
        clip_range=0.20,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log=str(args.out / "tensorboard"),
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint, evaluator],
        progress_bar=True,
    )

    final_path = args.out / "shadow_guard_ppo"
    model.save(final_path)
    print(f"Saved: {final_path}.zip")


if __name__ == "__main__":
    main()
