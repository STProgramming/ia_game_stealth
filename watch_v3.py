from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO

from shadow_ai import ShadowProtocolEnv, TacticalAction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--scenario",
        choices=[
            "routine",
            "sound",
            "visual",
            "lost_contact",
            "arrest",
            "theft",
        ],
        default="visual",
    )
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()

    env = ShadowProtocolEnv(render_mode="ansi")
    model = PPO.load(args.model, device="cpu")

    obs, info = env.reset(
        seed=777,
        options={"scenario": args.scenario},
    )

    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        action_i = int(action)

        obs, reward, terminated, truncated, info = env.step(action_i)

        print("\033[2J\033[H", end="")
        print(env.render())
        print()
        print(f"scenario : {info['training_scenario']}")
        print(f"action   : {TacticalAction(action_i).name}")
        print(f"reward   : {reward:.3f}")
        print(f"alert    : {info['alert']}")
        print(f"visual   : {info['visual_contact']}")
        print(f"sound    : {info['sound_contact']}")
        print(f"memory   : {info['last_seen_age']}")
        print(f"intel    : {info['intel_known_present']}")
        print(f"reinforce: {info['reinforcements_called']}")

        if terminated or truncated:
            print("\nEpisode ended.")
            break

        time.sleep(0.12)


if __name__ == "__main__":
    main()
