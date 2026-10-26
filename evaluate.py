from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from stable_baselines3 import PPO

from shadow_ai import ShadowProtocolEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/best/best_model.zip"),
    )
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()

    env = ShadowProtocolEnv()
    model = PPO.load(args.model, device="cpu")

    total_reward = 0.0
    captures = 0
    escaped = 0
    action_counts: Counter[str] = Counter()

    for episode in range(args.episodes):
        obs, info = env.reset(seed=1000 + episode)
        episode_reward = 0.0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            action_counts[info["action_name"]] += 1

            if terminated or truncated:
                if info["terminal_reward"] > 0:
                    captures += 1
                elif info["terminal_reward"] < 0:
                    escaped += 1
                break

        total_reward += episode_reward
        print(
            f"episode={episode:02d} "
            f"reward={episode_reward:8.3f} "
            f"style={info['player_style']} "
            f"alert={info['alert']}"
        )

    print("\n=== Evaluation ===")
    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {total_reward / args.episodes:.3f}")
    print(f"Captures: {captures}")
    print(f"Escapes: {escaped}")
    print("Actions:")
    for name, count in action_counts.most_common():
        print(f"  {name:<24} {count}")


if __name__ == "__main__":
    main()
