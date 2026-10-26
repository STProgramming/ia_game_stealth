
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


def doctrine_eval(model, episodes):
    print("\n================ V7.1 SWAT DOCTRINE ================")
    total = 0
    correct = 0
    per_skill = {}

    for skill in SWAT_SKILLS:
        env = SwatDoctrinePretrainEnv(
            forced_skill=skill.name,
            robust_context=True,
        )
        counts = Counter()
        ok = 0
        for ep in range(episodes):
            obs, _ = env.reset(seed=171000 + ep)
            action, _ = model.predict(obs, deterministic=True)
            action_i = int(action)
            counts[SwatAction(action_i).name] += 1
            _, _, _, _, info = env.step(action_i)
            ok += int(info["doctrine_correct"])
        accuracy = ok / episodes
        per_skill[skill.name] = accuracy
        correct += ok
        total += episodes

        dist = ", ".join(
            f"{name}={count/episodes*100:.1f}%"
            for name, count in counts.most_common(4)
        )
        print(
            f"{skill.name:<26} {accuracy*100:6.2f}% "
            f"expected={'/'.join(a.name for a in skill.expected):<18} {dist}"
        )

    total_acc = correct / max(total, 1)
    print(f"SWAT DOCTRINE TOTAL: {total_acc*100:.2f}%")
    return total_acc, per_skill


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
    results = {}

    print("\n================ V7.1 SWAT TEAM WORLD ================")
    for scenario in scenarios:
        env = SwatSharedPolicyEnv(scenario=scenario)
        role_actions = defaultdict(Counter)
        captures = neutralized = escapes = trunc = 0
        cleared = radio = 0
        reward_sum = 0.0

        for ep in range(episodes):
            obs, _ = env.reset(seed=181000 + ep)
            ep_reward = 0.0
            while True:
                acting = env.active_uid
                role = env.core.units[acting].role.name
                action, _ = model.predict(obs, deterministic=True)
                action_i = int(action)
                role_actions[role][SwatAction(action_i).name] += 1
                obs, reward, terminated, truncated, _ = env.step(action_i)
                ep_reward += reward
                if terminated or truncated:
                    if env.core.player_captured:
                        captures += 1
                    elif env.core.player_neutralized:
                        neutralized += 1
                    elif env.core.player_escaped:
                        escapes += 1
                    else:
                        trunc += 1
                    cleared += len(env.core.blackboard.cleared_feature_ids)
                    radio += env.core.blackboard.radio_updates
                    break
            reward_sum += ep_reward

        results[scenario] = {
            "captures": captures,
            "neutralized": neutralized,
            "escapes": escapes,
            "truncations": trunc,
        }

        print(f"\n=== {scenario.upper()} ===")
        print(
            f"mean_reward={reward_sum/episodes:.3f} "
            f"captures={captures} neutralized={neutralized} "
            f"escapes={escapes} truncations={trunc}"
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

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--doctrine-episodes", type=int, default=100)
    args = p.parse_args()

    model = PPO.load(args.model, device="cpu")
    doctrine, skills = doctrine_eval(model, args.doctrine_episodes)
    team = team_eval(model, args.episodes)

    arrest_success = (
        team["arrest"]["captures"] / max(args.episodes, 1)
    )
    confirmed_stop = (
        team["confirmed_visual"]["captures"]
        + team["confirmed_visual"]["neutralized"]
    ) / max(args.episodes, 1)

    stable = (
        doctrine >= 0.95
        and skills.get("utility_smoke", 0.0) >= 0.90
        and skills.get("reported_contact_pursue", 0.0) >= 0.90
        and skills.get("reported_compliant_arrest", 0.0) >= 0.90
        and arrest_success >= 0.80
        and confirmed_stop >= 0.60
    )

    print("\n================ STABILITY GATE ================")
    print(f"doctrine >= 95%:                 {doctrine*100:.2f}%")
    print(f"utility_smoke >= 90%:            {skills.get('utility_smoke',0)*100:.2f}%")
    print(f"reported_contact_pursue >= 90%:  {skills.get('reported_contact_pursue',0)*100:.2f}%")
    print(f"reported_compliant_arrest >=90%: {skills.get('reported_compliant_arrest',0)*100:.2f}%")
    print(f"arrest capture >= 80%:           {arrest_success*100:.2f}%")
    print(f"confirmed contact stopped >=60%: {confirmed_stop*100:.2f}%")
    print("V7.1 STABLE:", "YES" if stable else "NO")


if __name__ == "__main__":
    main()
