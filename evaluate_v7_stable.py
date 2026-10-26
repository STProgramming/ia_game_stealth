
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from stable_baselines3 import PPO

from shadow_ai import (
    SWAT_SKILLS,
    SwatAction,
    SwatDoctrinePretrainEnv,
    SwatSharedPolicyEnv,
)


def doctrine_eval(model, episodes_per_skill):
    print("\n================ SWAT DOCTRINE ================")
    total_ok = 0
    total = 0
    for skill in SWAT_SKILLS:
        env = SwatDoctrinePretrainEnv(
            forced_skill=skill.name,
            robust_context=True,
        )
        counts = Counter()
        ok = 0
        for ep in range(episodes_per_skill):
            obs, _ = env.reset(seed=90000 + ep)
            action, _ = model.predict(obs, deterministic=True)
            action_i = int(action)
            counts[SwatAction(action_i).name] += 1
            _, _, _, _, info = env.step(action_i)
            ok += int(info["doctrine_correct"])
        total_ok += ok
        total += episodes_per_skill
        expected = "/".join(a.name for a in skill.expected)
        dist = ", ".join(
            f"{name}={count/episodes_per_skill*100:.1f}%"
            for name, count in counts.most_common(4)
        )
        print(
            f"{skill.name:<20} role={skill.role.name:<8} "
            f"accuracy={ok/episodes_per_skill*100:6.2f}% "
            f"expected={expected:<18} {dist}"
        )
    print(f"SWAT DOCTRINE TOTAL: {total_ok/max(total,1)*100:.2f}%")


def team_eval(model, episodes):
    scenarios = [
        "entry_clear",
        "corner_clear",
        "lost_contact",
        "confirmed_visual",
        "utility",
        "arrest",
        "objective_breach",
    ]
    print("\n================ SWAT TEAM WORLD ================")
    for scenario in scenarios:
        env = SwatSharedPolicyEnv(scenario=scenario)
        actions = Counter()
        role_actions = defaultdict(Counter)
        captures = escapes = trunc = 0
        cleared = radio = utility = 0
        reward_sum = 0.0

        for ep in range(episodes):
            obs, info = env.reset(seed=100000 + ep)
            ep_reward = 0.0
            while True:
                acting = env.active_uid
                role = env.core.units[acting].role.name
                action, _ = model.predict(obs, deterministic=True)
                action_i = int(action)
                name = SwatAction(action_i).name
                actions[name] += 1
                role_actions[role][name] += 1

                obs, reward, terminated, truncated, info = env.step(action_i)
                ep_reward += reward
                if terminated or truncated:
                    if env.core.player_captured:
                        captures += 1
                    elif env.core.player_escaped:
                        escapes += 1
                    else:
                        trunc += 1
                    cleared += len(env.core.blackboard.cleared_feature_ids)
                    radio += env.core.blackboard.radio_updates
                    utility += int(env.core.utility_effect_event)
                    break
            reward_sum += ep_reward

        print(f"\n=== {scenario.upper()} ===")
        print(
            f"mean_reward={reward_sum/episodes:.3f} "
            f"captures={captures} escapes={escapes} truncations={trunc}"
        )
        print(
            f"avg_cleared={cleared/episodes:.2f} "
            f"avg_radio_updates={radio/episodes:.2f}"
        )
        for role in sorted(role_actions):
            n = sum(role_actions[role].values()) or 1
            top = ", ".join(
                f"{name}={count/n*100:.1f}%"
                for name, count in role_actions[role].most_common(5)
            )
            print(f"{role:<8} {top}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("swat_model", type=Path)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--doctrine-episodes", type=int, default=100)
    args = p.parse_args()

    model = PPO.load(args.swat_model, device="cpu")
    doctrine_eval(model, args.doctrine_episodes)
    team_eval(model, args.episodes)


if __name__ == "__main__":
    main()
