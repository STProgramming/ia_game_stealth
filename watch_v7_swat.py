
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from stable_baselines3 import PPO

from shadow_ai import SwatAction, SwatSharedPolicyEnv


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("--scenario", default="entry_clear")
    p.add_argument("--delay", type=float, default=0.08)
    args = p.parse_args()

    model = PPO.load(args.model, device="cpu")
    env = SwatSharedPolicyEnv(
        render_mode="ansi",
        scenario=args.scenario,
    )
    obs, info = env.reset()

    while True:
        acting = env.active_uid
        role = env.core.units[acting].role.name
        action, _ = model.predict(obs, deterministic=True)
        action_i = int(action)
        obs, reward, terminated, truncated, info = env.step(action_i)

        clear()
        print(env.render())
        print()
        print(
            f"agent={acting} role={role} "
            f"action={SwatAction(action_i).name} reward={reward:.3f}"
        )
        print(
            f"scenario={env.core.scenario} "
            f"contact={env.core.blackboard.contact_pos is not None} "
            f"contact_age={env.core.blackboard.contact_age:.1f} "
            f"cleared={len(env.core.blackboard.cleared_feature_ids)} "
            f"smoke={len(env.core.smoke_clouds)}"
        )
        time.sleep(args.delay)

        if terminated or truncated:
            print(
                "END:",
                "CAPTURE" if env.core.player_captured
                else "ESCAPE" if env.core.player_escaped
                else "TRUNCATED"
            )
            break


if __name__ == "__main__":
    main()
