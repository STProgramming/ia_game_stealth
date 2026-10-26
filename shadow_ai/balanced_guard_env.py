
from __future__ import annotations

from typing import Optional

import numpy as np

from .env import ShadowProtocolEnv


class BalancedGuardEnv(ShadowProtocolEnv):
    """
    Final full-world rehearsal for the normal guard.

    It uses the real environment, but deliberately oversamples the phases that
    remained contaminated in the V6.3.2 evaluation. No observation/action
    changes are introduced, so V6.x checkpoints remain compatible.
    """

    NAMES = (
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
    )

    WEIGHTS = np.array(
        [
            0.16,  # routine
            0.05,
            0.06,
            0.07,
            0.06,
            0.10,  # repeated sound/support
            0.16,  # reported/unreported visual transitions
            0.08,
            0.11,  # last_seen -> search boundary
            0.06,
            0.09,  # theft
        ],
        dtype=np.float64,
    )

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            # Parent reset will initialize its own RNG; this temporary RNG is
            # only for deterministic scenario choice.
            chooser = np.random.default_rng(seed + 771)
        else:
            chooser = self.rng

        weights = self.WEIGHTS / self.WEIGHTS.sum()
        scenario = str(chooser.choice(self.NAMES, p=weights))

        merged = dict(options or {})
        merged["scenario"] = scenario
        return super().reset(seed=seed, options=merged)
