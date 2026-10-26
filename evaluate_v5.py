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
    "search",
    "arrest",
    "theft",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")

    global_phases = defaultdict(Counter)

    for scenario in SCENARIOS:
        env = ShadowProtocolEnv()
        actions = Counter()
        phases = defaultdict(Counter)

        captures = 0
        escapes = 0
        truncations = 0
        calls = 0
        reinforcement_arrivals = 0
        helper_captures = 0
        search_visits = 0
        total_reward = 0.0

        for ep in range(args.episodes):
            obs, info = env.reset(
                seed=30_000 + ep,
                options={"scenario": scenario},
            )
            ep_reward = 0.0
            prev_called = False
            prev_active = False

            while True:
                phase = info["tactical_phase"]

                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )
                action_i = int(action)
                name = TacticalAction(action_i).name

                actions[name] += 1
                phases[phase][name] += 1
                global_phases[phase][name] += 1

                obs, reward, terminated, truncated, info = env.step(
                    action_i
                )
                ep_reward += reward

                if info["reinforcements_called"] and not prev_called:
                    calls += 1
                if info["reinforcements_active"] and not prev_active:
                    reinforcement_arrivals += 1

                prev_called = info["reinforcements_called"]
                prev_active = info["reinforcements_active"]
                search_visits = max(
                    search_visits,
                    info["search_waypoints_visited"],
                )

                if terminated or truncated:
                    if info["terminal_reward"] > 0:
                        captures += 1
                        if env.reinforcement_capture_event:
                            helper_captures += 1
                    elif info["terminal_reward"] < 0:
                        escapes += 1
                    else:
                        truncations += 1
                    break

            total_reward += ep_reward

        print(f"\n=== SCENARIO {scenario.upper()} ===")
        print(f"episodes={args.episodes}")
        print(f"mean_reward={total_reward / args.episodes:.3f}")
        print(f"captures={captures}")
        print(f"escapes={escapes}")
        print(f"truncations={truncations}")
        print(f"reinforcement_calls={calls}")
        print(f"reinforcement_arrivals={reinforcement_arrivals}")
        print(f"helper_captures={helper_captures}")
        print(f"max_search_visits={search_visits}")

        total = sum(actions.values()) or 1
        for name, count in actions.most_common():
            print(
                f"{name:<24} "
                f"{count:>8} "
                f"{count / total * 100:6.2f}%"
            )

        print("-- actual decision phases --")
        for phase in sorted(phases):
            total_phase = sum(phases[phase].values()) or 1
            rendered = ", ".join(
                f"{name}={count / total_phase * 100:.1f}%"
                for name, count in phases[phase].most_common(6)
            )
            print(f"{phase:<12} {rendered}")

    print("\n=== GLOBAL BY DECISION PHASE ===")
    for phase in sorted(global_phases):
        total = sum(global_phases[phase].values()) or 1
        print(f"\n[{phase}] n={total}")
        for name, count in global_phases[phase].most_common():
            print(
                f"  {name:<24} "
                f"{count:>8} "
                f"{count / total * 100:6.2f}%"
            )


if __name__ == "__main__":
    main()
