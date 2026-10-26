
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def run(cmd):
    print("\n>", " ".join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], check=True)


def main():
    p = argparse.ArgumentParser(
        description="Train the complete V7 stable stack."
    )
    p.add_argument(
        "--guard-init",
        type=Path,
        default=Path("models_v6_3/best_stable/best_model.zip"),
    )
    p.add_argument("--guard-cycles", type=int, default=8)
    p.add_argument("--swat-pretrain", type=int, default=100_000)
    p.add_argument("--swat-team", type=int, default=200_000)
    args = p.parse_args()

    run([
        sys.executable,
        "train_v6_3_2.py",
        "--init-model", args.guard_init,
        "--cycles", args.guard_cycles,
        "--full-steps", 6000,
        "--doctrine-steps", 4000,
    ])

    run([
        sys.executable,
        "train_v7_swat.py",
        "--pretrain", args.swat_pretrain,
        "--team", args.swat_team,
    ])

    print("\nV7 training complete.")
    print("Guard:", "models_v6_3_2/best_stable/best_model.zip")
    print("SWAT:", "models_v7_swat/swat_shared_policy_v7.zip")


if __name__ == "__main__":
    main()
