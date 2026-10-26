from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from stable_baselines3 import PPO

from shadow_ai import ShadowProtocolEnv, TacticalAction


def context_name(obs) -> str:
    visual = obs[5] > 0.5
    sound = obs[10] > 0.5
    memory = obs[14] > 0.5
    alert = obs[4]

    if visual and obs[24] > 0.5:
        return "visual_compliant"
    if visual:
        return "visual_contact"
    if sound:
        return "sound_only"
    if memory:
        return "memory_only"
    if alert > 0.25:
        return "alert_no_contact"
    return "routine"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    env = ShadowProtocolEnv()
    model = PPO.load(args.model, device="cpu")

    actions = Counter()
    by_context = defaultdict(Counter)
    captures = 0
    escapes = 0
    truncations = 0
    total_reward = 0.0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=5000 + ep)
        ep_reward = 0.0

        while True:
            context = context_name(obs)
            action, _ = model.predict(obs, deterministic=True)
            action_i = int(action)
            name = TacticalAction(action_i).name

            actions[name] += 1
            by_context[context][name] += 1

            obs, reward, terminated, truncated, info = env.step(action_i)
            ep_reward += reward

            if terminated or truncated:
                if info["terminal_reward"] > 0:
                    captures += 1
                elif info["terminal_reward"] < 0:
                    escapes += 1
                else:
                    truncations += 1
                break

        total_reward += ep_reward

    print("=== SUMMARY ===")
    print(f"Episodes:      {args.episodes}")
    print(f"Mean reward:   {total_reward / args.episodes:.3f}")
    print(f"Captures:      {captures}")
    print(f"Escapes:       {escapes}")
    print(f"Truncations:   {truncations}")

    print("\n=== ALL ACTIONS ===")
    for name, count in actions.most_common():
        print(f"{name:<24} {count:>8}")

    print("\n=== ACTIONS BY CONTEXT ===")
    for context in sorted(by_context):
        total = sum(by_context[context].values())
        print(f"\n[{context}] n={total}")
        for name, count in by_context[context].most_common():
            pct = count / total * 100.0
            print(f"  {name:<24} {count:>7}  {pct:6.2f}%")


if __name__ == "__main__":
    main()
