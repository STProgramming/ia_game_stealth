from __future__ import annotations

import argparse
import time
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
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="lost_contact",
    )
    parser.add_argument("--steps", type=int, default=180)
    args = parser.parse_args()

    env = ShadowProtocolEnv(render_mode="ansi")
    model = PPO.load(args.model, device="cpu")

    obs, info = env.reset(
        seed=999,
        options={"scenario": args.scenario},
    )

    for _ in range(args.steps):
        decision_phase = info["tactical_phase"]

        action, _ = model.predict(
            obs,
            deterministic=True,
        )
        action_i = int(action)

        obs, reward, terminated, truncated, info = env.step(
            action_i
        )

        print("\033[2J\033[H", end="")
        print(env.render())
        print()
        print(f"scenario       : {info['training_scenario']}")
        print(f"decision phase : {decision_phase}")
        print(f"action         : {TacticalAction(action_i).name}")
        print(f"result phase   : {info['result_phase']}")
        print(f"reward         : {reward:.3f}")
        print(f"alert          : {info['alert']}")
        print(f"visual         : {info['visual_contact']}")
        print(f"sound          : {info['sound_contact']}")
        print(f"memory age     : {info['last_seen_age']}")
        print(f"arrest         : {info['arrest_progress']:.2f}/1.00")
        print(
            "search         : "
            f"{info['search_waypoint_index']}/"
            f"{info['search_waypoint_count']} "
            f"visited={info['search_waypoints_visited']}"
        )
        print(
            "reinforcements : "
            f"called={info['reinforcements_called']} "
            f"active={info['reinforcements_active']} "
            f"count={info['reinforcement_count']} "
            f"timer={info['reinforcement_timer']:.1f}s"
        )
        print(
            f"radio contact  : "
            f"{info['reinforcement_radio_contact']}"
        )

        if terminated or truncated:
            print("\nEpisode ended.")
            break

        time.sleep(0.12)


if __name__ == "__main__":
    main()
