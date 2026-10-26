from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from shadow_ai import DoctrinePretrainEnv, ShadowProtocolEnv


def make_doctrine_env():
    return Monitor(DoctrinePretrainEnv())


def make_full_env():
    return Monitor(ShadowProtocolEnv())


def build_model(env, seed: int, tensorboard_dir: Path) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        learning_rate=2.0e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.025,
        clip_range=0.20,
        policy_kwargs=dict(
            net_arch=dict(pi=[320, 256], vf=[320, 256])
        ),
        tensorboard_log=str(tensorboard_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V6.2 doctrine pretraining + full-environment fine-tuning."
    )
    parser.add_argument("--pretrain", type=int, default=150_000)
    parser.add_argument("--finetune", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models_v6_2"),
    )
    parser.add_argument(
        "--init-model",
        type=Path,
        default=None,
        help=(
            "Optional compatible 43-observation/16-action PPO model. "
            "If supplied, doctrine pretraining continues from it."
        ),
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip doctrine pretraining and run only full-environment fine-tuning.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw_doctrine = DoctrinePretrainEnv(seed=args.seed)
    raw_full = ShadowProtocolEnv(seed=args.seed)
    check_env(raw_doctrine, warn=True)
    check_env(raw_full, warn=True)

    doctrine_train_env = DummyVecEnv([make_doctrine_env])
    doctrine_eval_env = DummyVecEnv([make_doctrine_env])

    if args.init_model is not None:
        print(f"Loading compatible model: {args.init_model}")
        model = PPO.load(
            args.init_model,
            env=doctrine_train_env,
            device="cpu",
        )
        model.verbose = 1
        model.tensorboard_log = str(args.out / "tensorboard")
        reset_num_timesteps = False
    else:
        model = build_model(
            doctrine_train_env,
            args.seed,
            args.out / "tensorboard",
        )
        reset_num_timesteps = True

    if not args.skip_pretrain and args.pretrain > 0:
        doctrine_checkpoint = CheckpointCallback(
            save_freq=50_000,
            save_path=str(args.out / "pretrain_checkpoints"),
            name_prefix="shadow_guard_v6_2_doctrine",
        )
        doctrine_eval = EvalCallback(
            doctrine_eval_env,
            best_model_save_path=str(args.out / "pretrain_best"),
            log_path=str(args.out / "pretrain_eval"),
            eval_freq=20_000,
            n_eval_episodes=512,
            deterministic=True,
        )

        print("\n=== STAGE 1: DOCTRINE PRETRAINING ===")
        model.learn(
            total_timesteps=args.pretrain,
            callback=[doctrine_checkpoint, doctrine_eval],
            progress_bar=True,
            reset_num_timesteps=reset_num_timesteps,
        )
        doctrine_path = args.out / "doctrine_pretrained"
        model.save(doctrine_path)
        print(f"Doctrine model saved: {doctrine_path}.zip")

    # Same 43 observations / 16 actions, so PPO weights and optimizer transfer.
    full_train_env = DummyVecEnv([make_full_env])
    full_eval_env = DummyVecEnv([make_full_env])
    model.set_env(full_train_env)

    if args.finetune > 0:
        full_checkpoint = CheckpointCallback(
            save_freq=50_000,
            save_path=str(args.out / "finetune_checkpoints"),
            name_prefix="shadow_guard_v6_2_full",
        )
        full_eval = EvalCallback(
            full_eval_env,
            best_model_save_path=str(args.out / "finetune_best"),
            log_path=str(args.out / "finetune_eval"),
            eval_freq=25_000,
            n_eval_episodes=100,
            deterministic=True,
        )

        print("\n=== STAGE 2: FULL ENVIRONMENT FINE-TUNING ===")
        model.learn(
            total_timesteps=args.finetune,
            callback=[full_checkpoint, full_eval],
            progress_bar=True,
            reset_num_timesteps=False,
        )

    final_path = args.out / "shadow_guard_v6_2"
    model.save(final_path)
    print(f"\nFinal model saved: {final_path}.zip")
    print("Full-environment best model:")
    print(args.out / "finetune_best" / "best_model.zip")


if __name__ == "__main__":
    main()
