
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def run(args):
    print("\n>", " ".join(str(x) for x in args))
    subprocess.run([str(x) for x in args], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--guard-init",
        type=Path,
        default=Path("models_v6_3_2/best_stable/best_model.zip"),
    )
    p.add_argument(
        "--swat-init",
        type=Path,
        default=Path("models_v7_swat/swat_shared_policy_v7.zip"),
    )
    args = p.parse_args()

    run([
        sys.executable,
        "train_v7_1_guard_stable.py",
        "--init-model",
        args.guard_init,
    ])
    run([
        sys.executable,
        "train_v7_1_swat_stable.py",
        "--init-model",
        args.swat_init,
    ])

    print("\nV7.1 final stabilization complete.")
    print("Guard: models_v7_1_guard/guard_v7_1_final.zip")
    print("SWAT:  models_v7_1_swat/swat_v7_1_final.zip")


if __name__ == "__main__":
    main()
