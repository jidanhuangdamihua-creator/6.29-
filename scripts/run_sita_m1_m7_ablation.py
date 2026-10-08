#!/usr/bin/env python3
"""Plan or execute one SITA M1–M7 condition.

The default is plan-only. Training requires ``--execute`` and exactly one
dataset, horizon, and seed. A plan file is never marked complete.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.transfer_methods.sita_m1_m7_ablation import (
    M7ContractError,
    GROUPS,
    default_output_root,
    execute_condition,
    parse_groups,
    render_plan,
    write_plan_file,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SITA M1–M7 ablation runner.")
    parser.add_argument("--execute", action="store_true", help="Train the selected condition. Omit to print a plan.")
    parser.add_argument("--resume", action="store_true", help="Continue an existing condition without repeating completed groups.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--groups", default=",".join(GROUPS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        groups = parse_groups(args.groups)
        output_root = args.output_root or default_output_root()
        if not args.execute:
            text = render_plan(
                dataset=args.dataset,
                horizon=args.horizon,
                seed=args.seed,
                groups=groups,
                output_root=output_root,
            )
            print(text, end="")
            if args.output_root is not None:
                write_plan_file(
                    args.output_root,
                    text,
                    dataset=args.dataset,
                    horizon=args.horizon,
                    seed=args.seed,
                    groups=groups,
                )
            return 0
        if args.dataset is None or args.horizon is None or args.seed is None:
            raise M7ContractError("--execute requires --dataset, --horizon, and --seed")
        result = execute_condition(
            dataset=args.dataset,
            horizon=int(args.horizon),
            seed=int(args.seed),
            output_root=output_root,
            groups=groups,
            resume=bool(args.resume),
        )
        print(f"CONDITION_STATUS={result['status']}")
        print(f"CONDITION_DIR={result['condition_dir']}")
        return 0 if result["status"] == "COMPLETE" else 2
    except M7ContractError as exc:
        print(str(exc), file=sys.stderr)
        print("TRAINING_CALLED=NO" if not args.execute else "CONDITION_STATUS=FAILED", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
