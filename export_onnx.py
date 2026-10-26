from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import onnx
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy


class OnnxablePPOPolicy(th.nn.Module):
    def __init__(self, policy: BasePolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        observation: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        return self.policy(observation, deterministic=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/best/best_model.zip"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("models/shadow_guard_policy.onnx"),
    )
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    wrapper = OnnxablePPOPolicy(model.policy).eval()

    obs_shape = model.observation_space.shape
    dummy = th.zeros((1, *obs_shape), dtype=th.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    th.onnx.export(
        wrapper,
        dummy,
        str(args.out),
        opset_version=17,
        input_names=["observation"],
        output_names=["action", "value", "log_prob"],
        dynamic_axes={"observation": {0: "batch"}},
    )

    onnx_model = onnx.load(str(args.out))
    onnx.checker.check_model(onnx_model)
    print(f"ONNX valid: {args.out}")


if __name__ == "__main__":
    main()
