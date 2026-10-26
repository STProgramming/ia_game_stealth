from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from stable_baselines3 import PPO

from shadow_ai import ShadowProtocolEnv, TacticalAction


SCENARIOS = [
    'routine', 'peripheral', 'attention', 'repeat_visual',
    'sound', 'repeat_sound', 'visual', 'lost_contact',
    'search', 'arrest', 'theft',
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--episodes', type=int, default=50)
    args = parser.parse_args()

    model = PPO.load(args.model, device='cpu')
    global_phases = defaultdict(Counter)

    for scenario in SCENARIOS:
        env = ShadowProtocolEnv()
        actions = Counter()
        phases = defaultdict(Counter)
        captures = escapes = truncations = 0
        calls = local_support = radio = 0
        peak_conf = 0.0
        max_exposures = 0
        total_reward = 0.0

        for ep in range(args.episodes):
            obs, info = env.reset(
                seed=40_000 + ep,
                options={'scenario': scenario},
            )
            ep_reward = 0.0
            prev_call = prev_support = prev_radio = False

            while True:
                phase = info['tactical_phase']
                action, _ = model.predict(obs, deterministic=True)
                action_i = int(action)
                name = TacticalAction(action_i).name
                actions[name] += 1
                phases[phase][name] += 1
                global_phases[phase][name] += 1

                obs, reward, terminated, truncated, info = env.step(action_i)
                ep_reward += reward
                peak_conf = max(peak_conf, info['visual_confidence'])
                max_exposures = max(max_exposures, info['visual_exposure_count'])

                if info['reinforcements_called'] and not prev_call:
                    calls += 1
                if info['buddy_supporting'] and not prev_support:
                    local_support += 1
                if info['radio_reported'] and not prev_radio:
                    radio += 1
                prev_call = info['reinforcements_called']
                prev_support = info['buddy_supporting']
                prev_radio = info['radio_reported']

                if terminated or truncated:
                    if info['terminal_reward'] > 0:
                        captures += 1
                    elif info['terminal_reward'] < 0:
                        escapes += 1
                    else:
                        truncations += 1
                    break

            total_reward += ep_reward

        print(f'\n=== {scenario.upper()} ===')
        print(f'mean_reward={total_reward / args.episodes:.3f}')
        print(f'captures={captures} escapes={escapes} truncations={truncations}')
        print(f'local_support_requests={local_support}')
        print(f'radio_reports={radio}')
        print(f'reinforcement_calls={calls}')
        print(f'peak_visual_confidence={peak_conf:.2f}')
        print(f'max_visual_exposures={max_exposures}')

        total = sum(actions.values()) or 1
        for name, count in actions.most_common():
            print(f'{name:<24} {count:>8} {count / total * 100:6.2f}%')

        print('-- phases --')
        for phase in sorted(phases):
            n = sum(phases[phase].values()) or 1
            top = ', '.join(
                f'{name}={count / n * 100:.1f}%'
                for name, count in phases[phase].most_common(6)
            )
            print(f'{phase:<18} {top}')

    print('\n=== GLOBAL BY PHASE ===')
    for phase in sorted(global_phases):
        total = sum(global_phases[phase].values()) or 1
        print(f'\n[{phase}] n={total}')
        for name, count in global_phases[phase].most_common():
            print(f'  {name:<24} {count:>8} {count / total * 100:6.2f}%')


if __name__ == '__main__':
    main()
