from __future__ import annotations

from shadow_ai import ShadowProtocolEnv


def main() -> None:
    env = ShadowProtocolEnv(render_mode="ansi")
    obs, info = env.reset(seed=7)

    for step in range(80):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 10 == 0:
            print(f"\nstep={step} action={info['action_name']} reward={reward:.3f}")
            print(env.render())

        if terminated or truncated:
            print("\nEpisode ended:", info)
            break


if __name__ == "__main__":
    main()
