from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from stable_baselines3 import PPO

from shadow_ai import (
    DoctrinePretrainEnv,
    SKILLS,
    ShadowProtocolEnv,
    TacticalAction,
)


SCENARIOS = [
    "routine",
    "peripheral",
    "attention",
    "repeat_visual",
    "sound",
    "repeat_sound",
    "visual",
    "lost_contact",
    "search",
    "arrest",
    "theft",
]


def evaluate_doctrine(model: PPO, episodes_per_skill: int) -> None:
    print("\n================ V6.3 DOCTRINE MICRO-EVAL ================")
    total_correct = 0
    total = 0

    for skill in SKILLS:
        env = DoctrinePretrainEnv(forced_skill=skill.name)
        actions = Counter()
        correct = 0
        phases = Counter()

        for ep in range(episodes_per_skill):
            obs, info = env.reset(seed=70_000 + ep)
            phases[info["tactical_phase"]] += 1
            action, _ = model.predict(obs, deterministic=True)
            action_i = int(action)
            name = TacticalAction(action_i).name
            actions[name] += 1

            _, _, _, _, result = env.step(action_i)
            correct += int(bool(result["doctrine_correct"]))

        total_correct += correct
        total += episodes_per_skill
        expected = "/".join(
            action.name for action in skill.expected_actions
        )
        most_common = ", ".join(
            f"{name}={count / episodes_per_skill * 100:.1f}%"
            for name, count in actions.most_common(5)
        )
        phase_text = ", ".join(
            f"{phase}={count}"
            for phase, count in phases.items()
        )
        print(
            f"{skill.name:<22} "
            f"accuracy={correct / episodes_per_skill * 100:6.2f}% "
            f"expected={expected:<34} "
            f"phase={phase_text}"
        )
        print(f"  actions: {most_common}")

    print(
        f"\nDOCTRINE TOTAL ACCURACY: "
        f"{total_correct / max(total, 1) * 100:.2f}%"
    )


def evaluate_full(model: PPO, episodes: int) -> None:
    print("\n================ V6.3 FULL ENVIRONMENT ================")
    global_phases = defaultdict(Counter)

    for scenario in SCENARIOS:
        env = ShadowProtocolEnv()
        actions = Counter()
        phases = defaultdict(Counter)
        captures = escapes = truncations = 0

        # Direct policy decisions: this fixes the old V6 evaluator bug.
        local_support_actions = 0
        radio_report_actions = 0
        reinforcement_call_actions = 0

        # Environmental consequences are tracked separately.
        buddy_support_activations = 0
        radio_state_activations = 0
        reinforcement_state_activations = 0

        peak_conf = 0.0
        max_exposures = 0
        total_reward = 0.0

        for ep in range(episodes):
            obs, info = env.reset(
                seed=80_000 + ep,
                options={"scenario": scenario},
            )
            ep_reward = 0.0
            prev_call = bool(info["reinforcements_called"])
            prev_support = bool(info["buddy_supporting"])
            prev_radio = bool(info["radio_reported"])

            while True:
                phase = info["tactical_phase"]
                action, _ = model.predict(obs, deterministic=True)
                action_i = int(action)
                action_enum = TacticalAction(action_i)
                name = action_enum.name

                actions[name] += 1
                phases[phase][name] += 1
                global_phases[phase][name] += 1

                if action_enum == TacticalAction.REQUEST_LOCAL_SUPPORT:
                    local_support_actions += 1
                elif action_enum == TacticalAction.RADIO_REPORT:
                    radio_report_actions += 1
                elif action_enum == TacticalAction.CALL_REINFORCEMENTS:
                    reinforcement_call_actions += 1

                obs, reward, terminated, truncated, info = env.step(action_i)
                ep_reward += reward
                peak_conf = max(
                    peak_conf,
                    float(info["visual_confidence"]),
                )
                max_exposures = max(
                    max_exposures,
                    int(info["visual_exposure_count"]),
                )

                if info["reinforcements_called"] and not prev_call:
                    reinforcement_state_activations += 1
                if info["buddy_supporting"] and not prev_support:
                    buddy_support_activations += 1
                if info["radio_reported"] and not prev_radio:
                    radio_state_activations += 1

                prev_call = bool(info["reinforcements_called"])
                prev_support = bool(info["buddy_supporting"])
                prev_radio = bool(info["radio_reported"])

                if terminated or truncated:
                    if info["terminal_reward"] > 0:
                        captures += 1
                    elif info["terminal_reward"] < 0:
                        escapes += 1
                    else:
                        truncations += 1
                    break

            total_reward += ep_reward

        print(f"\n=== {scenario.upper()} ===")
        print(f"mean_reward={total_reward / episodes:.3f}")
        print(
            f"captures={captures} escapes={escapes} "
            f"truncations={truncations}"
        )
        print(f"REQUEST_LOCAL_SUPPORT actions={local_support_actions}")
        print(f"RADIO_REPORT actions={radio_report_actions}")
        print(f"CALL_REINFORCEMENTS actions={reinforcement_call_actions}")
        print(f"buddy_support_activations={buddy_support_activations}")
        print(f"radio_state_activations={radio_state_activations}")
        print(
            "reinforcement_state_activations="
            f"{reinforcement_state_activations}"
        )
        print(f"peak_visual_confidence={peak_conf:.2f}")
        print(f"max_visual_exposures={max_exposures}")

        n_actions = sum(actions.values()) or 1
        for name, count in actions.most_common():
            print(
                f"{name:<24} {count:>8} "
                f"{count / n_actions * 100:6.2f}%"
            )

        print("-- phases --")
        for phase in sorted(phases):
            n = sum(phases[phase].values()) or 1
            top = ", ".join(
                f"{name}={count / n * 100:.1f}%"
                for name, count in phases[phase].most_common(8)
            )
            print(f"{phase:<18} {top}")

    print("\n=== GLOBAL BY PHASE ===")
    for phase in sorted(global_phases):
        n = sum(global_phases[phase].values()) or 1
        print(f"\n[{phase}] n={n}")
        for name, count in global_phases[phase].most_common():
            print(
                f"  {name:<24} {count:>8} "
                f"{count / n * 100:6.2f}%"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--doctrine-episodes",
        type=int,
        default=100,
    )
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    evaluate_doctrine(model, args.doctrine_episodes)
    evaluate_full(model, args.episodes)


if __name__ == "__main__":
    main()
