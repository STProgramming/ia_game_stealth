from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from stable_baselines3 import PPO

from shadow_ai import ShadowProtocolEnv, TacticalAction


SCENARIOS = [
    "routine",
    "sound",
    "visual",
    "lost_contact",
    "arrest",
    "theft",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=40)
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")

    grand_actions = Counter()

    for scenario in SCENARIOS:
        env = ShadowProtocolEnv()
        action_counts = Counter()
        captures = 0
        escapes = 0
        total_reward = 0.0

        for ep in range(args.episodes):
            obs, info = env.reset(
                seed=10_000 + ep,
                options={"scenario": scenario},
            )
            ep_reward = 0.0

            while True:
                action, _ = model.predict(obs, deterministic=True)
                action_i = int(action)
                name = TacticalAction(action_i).name

                action_counts[name] += 1
                grand_actions[name] += 1

                obs, reward, terminated, truncated, info = env.step(action_i)
                ep_reward += reward

                if terminated or truncated:
                    if info["terminal_reward"] > 0:
                        captures += 1
                    elif info["terminal_reward"] < 0:
                        escapes += 1
                    break

            total_reward += ep_reward

        total_actions = sum(action_counts.values()) or 1

        print(f"\n=== {scenario.upper()} ===")
        print(f"episodes={args.episodes}")
        print(f"mean_reward={total_reward / args.episodes:.3f}")
        print(f"captures={captures}")
        print(f"escapes={escapes}")

        for name, count in action_counts.most_common():
            print(
                f"{name:<24} "
                f"{count:>8} "
                f"{count / total_actions * 100:6.2f}%"
            )

    print("\n=== GLOBAL ACTIONS ===")
    total = sum(grand_actions.values()) or 1
    for name, count in grand_actions.most_common():
        print(
            f"{name:<24} "
            f"{count:>8} "
            f"{count / total * 100:6.2f}%"
        )


if __name__ == "__main__":
    main()
