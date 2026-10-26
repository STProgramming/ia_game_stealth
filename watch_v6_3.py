from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO

from shadow_ai import ShadowProtocolEnv, TacticalAction, VisualBand

SCENARIOS = [
    'routine', 'peripheral', 'attention', 'repeat_visual',
    'sound', 'repeat_sound', 'visual', 'lost_contact',
    'search', 'arrest', 'theft',
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--scenario', choices=SCENARIOS, default='attention')
    parser.add_argument('--steps', type=int, default=220)
    args = parser.parse_args()

    env = ShadowProtocolEnv(render_mode='ansi')
    model = PPO.load(args.model, device='cpu')
    obs, info = env.reset(seed=606, options={'scenario': args.scenario})

    for _ in range(args.steps):
        decision_phase = info['tactical_phase']
        action, _ = model.predict(obs, deterministic=True)
        action_i = int(action)
        obs, reward, terminated, truncated, info = env.step(action_i)

        print('\033[2J\033[H', end='')
        print(env.render())
        print()
        print(f"scenario       : {info['training_scenario']}")
        print(f"decision phase : {decision_phase}")
        print(f"action         : {TacticalAction(action_i).name}")
        print(f"result phase   : {info['result_phase']}")
        print(f"reward         : {reward:.3f}")
        print(f"visual conf    : {info['visual_confidence']:.3f}")
        try:
            band = VisualBand(info['visual_band']).name
        except Exception:
            band = str(info['visual_band'])
        print(f"visual band    : {band}")
        print(f"peripheral     : {info['visual_peripheral']}")
        print(f"spot rate      : {info['spot_rate']:.3f}/s")
        print(f"suspicion      : {info['personal_suspicion']:.3f}")
        print(f"visual repeats : {info['visual_exposure_count']}")
        print(f"audio repeats  : {info['audio_repeat_count']}")
        print(f"intruder known : {info['intruder_known']}")
        print(f"focus          : {info['focus_visual']}")
        print(f"flashlight     : {info['flashlight_active']}")
        print(f"night          : {info['is_night']}")
        print(f"buddy support  : {info['buddy_supporting']}")
        print(f"radio report   : {info['radio_reported']}")
        print(f"reinforcements : {info['reinforcements_called']}")

        if terminated or truncated:
            print('\nEpisode ended.')
            break
        time.sleep(0.12)


if __name__ == '__main__':
    main()
